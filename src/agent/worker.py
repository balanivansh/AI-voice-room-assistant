import asyncio
import inspect
import json
import time
from typing import Callable, Dict, Optional
from livekit import api, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.stt import SpeechEventType
from livekit.agents.vad import VADEventType
from src.agent.personas import PersonaManager
from src.agent.queue_manager import TurnQueueManager
from src.audio.audio_publisher import BotAudioPublisher
from src.audio.stt import create_stt
from src.audio.transcription_handler import TranscriptionHandler, UtteranceNormalizer
from src.audio.tts import AzureSpeechSynthesizer, STATIC_FALLBACK_PCM
from src.audio.vad import create_vad
from src.config import settings
from src.logger import get_logger, log_structured_event
from src.state.graph import RoomStateGraphManager
from src.state.room_state import TurnPlan
from src.utils.fallback import safe_external_call

logger = get_logger("roxstar_voice_assistant.agent.worker")


class ParticipantAudioPipeline:
    """Isolated per-participant audio processing pipeline (Silero VAD + Deepgram STT)."""

    def __init__(
        self,
        participant_identity: str,
        track: rtc.RemoteAudioTrack,
        transcription_handler: TranscriptionHandler,
        speech_started_callback: Optional[Callable[[str], None]] = None,
        target_bot: str = "orchestrator",
    ) -> None:
        self.participant_identity = participant_identity
        self.track = track
        self.transcription_handler = transcription_handler
        self.speech_started_callback = speech_started_callback
        self.target_bot = target_bot
        self._processing_task: Optional[asyncio.Task] = None
        self._closed = False
        self.resampler: Optional[rtc.AudioResampler] = None
        self.resampler_input_rate: Optional[int] = None

    async def start(self) -> None:
        """Initialize VAD and STT components and launch stream reader task."""
        stt = await create_stt(
            model="nova-2",
            language="hi",
            target_bot=self.target_bot,
        )
        vad = create_vad()
        self._processing_task = asyncio.create_task(
            self._process_audio_stream(stt, vad)
        )
        logger.info(
            "participant_pipeline_started",
            participant=self.participant_identity,
            target_bot=self.target_bot,
        )

    async def _process_audio_stream(self, stt, vad) -> None:
        """Stream audio frames from track through VAD and STT."""
        audio_stream = rtc.AudioStream(self.track)
        stt_stream = stt.stream()
        vad_stream = vad.stream()

        async def _forward_frames():
            audio_iter = audio_stream.__aiter__()
            silence_frame = rtc.AudioFrame(
                data=b"\x00" * 640,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=320,
            )
            while not self._closed:
                try:
                    event = await asyncio.wait_for(audio_iter.__anext__(), timeout=0.05)
                    if self._closed:
                        break
                    frame = event.frame

                    # Dynamically adapt AudioResampler if input sample rate changes or is not initialized
                    if self.resampler is None or self.resampler_input_rate != frame.sample_rate:
                        self.resampler = rtc.AudioResampler(
                            input_rate=frame.sample_rate,
                            output_rate=16000,
                            num_channels=1,
                        )
                        self.resampler_input_rate = frame.sample_rate

                    resampled_frames = self.resampler.push(frame)
                    for resampled_frame in resampled_frames:
                        stt_stream.push_frame(resampled_frame)
                        vad_stream.push_frame(resampled_frame)
                except asyncio.TimeoutError:
                    if self._closed:
                        break
                    # Send 20ms silence frame every 50ms to prevent STT WebSocket 1006 audio starvation
                    stt_stream.push_frame(silence_frame)
                except StopAsyncIteration:
                    break

        async def _consume_vad_events():
            async for event in vad_stream:
                if self._closed:
                    break
                if event.type == VADEventType.START_OF_SPEECH:
                    logger.info(
                        "vad_speech_started_barge_in",
                        participant=self.participant_identity,
                    )
                    if self.speech_started_callback:
                        self.speech_started_callback(self.participant_identity)

        async def _consume_stt_events():
            async for event in stt_stream:
                if self._closed:
                    break
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    start_time = time.perf_counter()
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        latency_ms = (time.perf_counter() - start_time) * 1000.0
                        log_structured_event(
                            event_name="stt_final_transcript",
                            target_bot=self.target_bot,
                            latency_ms=latency_ms,
                            fallback_used=False,
                            speaker_identity=self.participant_identity,
                            source="voice",
                            text=text,
                        )
                        await self.transcription_handler.normalize_and_dispatch(
                            identity=self.participant_identity,
                            text=text,
                            source="voice",
                            target_bot=self.target_bot,
                        )

        try:
            await asyncio.gather(
                _forward_frames(), _consume_vad_events(), _consume_stt_events()
            )
        except Exception as exc:
            logger.error(
                "pipeline_stream_error",
                participant=self.participant_identity,
                error=str(exc),
            )
        finally:
            await stt_stream.aclose()
            await vad_stream.aclose()
            await audio_stream.aclose()

    async def stop(self) -> None:
        """Terminate the participant pipeline cleanly."""
        self._closed = True
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "participant_pipeline_stopped",
            participant=self.participant_identity,
        )


class VoiceAssistantWorker:
    """Manages LiveKit room connection, participant lifecycle, and message subscriptions."""

    def __init__(
        self,
        transcription_handler: Optional[TranscriptionHandler] = None,
        graph_manager: Optional[RoomStateGraphManager] = None,
        queue_manager: Optional[TurnQueueManager] = None,
        persona_manager: Optional[PersonaManager] = None,
        tts_synthesizer: Optional[AzureSpeechSynthesizer] = None,
        audio_publisher: Optional[BotAudioPublisher] = None,
        target_bot: str = "orchestrator",
    ) -> None:
        self.transcription_handler = transcription_handler or TranscriptionHandler()
        self.graph_manager = graph_manager or RoomStateGraphManager()
        self.queue_manager = queue_manager or TurnQueueManager()
        self.persona_manager = persona_manager or PersonaManager()
        self.tts_synthesizer = tts_synthesizer or AzureSpeechSynthesizer()
        
        self.dost_publisher = audio_publisher or BotAudioPublisher(
            bot_identity="AI Dost",
            gender="male",
        )
        self.sathi_publisher = BotAudioPublisher(
            bot_identity="AI Sathi",
            gender="female",
        )
        self.audio_publishers: Dict[str, BotAudioPublisher] = {
            "dost": self.dost_publisher,
            "sathi": self.sathi_publisher,
        }
        self.target_bot = target_bot
        self.pipelines: Dict[str, ParticipantAudioPipeline] = {}
        self.room: Optional[rtc.Room] = None
        self.sathi_room: Optional[rtc.Room] = None

        self.cancellation_event = asyncio.Event()

        # Sentence debounce state
        self._debounce_tasks: Dict[str, asyncio.Task] = {}
        self._transcript_buffers: Dict[str, List[str]] = {}

        # Attach callbacks
        self.transcription_handler.register_callback(self.on_utterance_normalized)
        self.queue_manager.set_executor(self.execute_turn_plan)

    @property
    def audio_publisher(self) -> BotAudioPublisher:
        """Dynamic getter for active/speaking audio publisher or default dost_publisher."""
        if self.sathi_publisher.is_speaking:
            return self.sathi_publisher
        return self.dost_publisher

    @audio_publisher.setter
    def audio_publisher(self, publisher: BotAudioPublisher) -> None:
        self.dost_publisher = publisher
        self.audio_publishers["dost"] = publisher

    def trigger_barge_in_interruption(self, interrupting_speaker: Optional[str] = None) -> None:
        """Trigger immediate barge-in or graceful yield on user speech start ONLY if bot is speaking."""
        is_any_speaking = any(p.is_speaking for p in self.audio_publishers.values())
        if not is_any_speaking:
            logger.debug("barge_in_skipped_bot_not_speaking")
            return

        active_speaker = self.graph_manager.state.get("current_speaker", "")

        # Distinct Speaker Handling (User B speaks while bot answers User A):
        # Immediate Audio Yield without destructive queue wipe
        if interrupting_speaker and active_speaker and interrupting_speaker != active_speaker:
            logger.info(
                "distinct_speaker_graceful_yield_triggered",
                interrupting_speaker=interrupting_speaker,
                active_speaker=active_speaker,
            )
            for pub in self.audio_publishers.values():
                pub.stop_current()
            self.graph_manager.record_interrupted_turn(active_speaker)
            log_structured_event(
                event_name="graceful_yield_halted_audio",
                target_bot=self.target_bot,
                active_speaker=active_speaker,
                interrupting_speaker=interrupting_speaker,
            )
            return

        # Same Speaker Handling (User A interrupts User A's bot turn) or unknown speaker:
        self.cancellation_event.set()
        for pub in self.audio_publishers.values():
            pub.stop_current()
        cleared = self.queue_manager.cancel_current_and_clear()
        log_structured_event(
            event_name="barge_in_interruption_triggered",
            target_bot=self.target_bot,
            cleared_turns_count=cleared,
            interrupting_speaker=interrupting_speaker,
        )

    async def execute_turn_plan(self, turn: TurnPlan) -> None:
        """Sequential turn execution worker for persona text generation, Azure TTS, and audio playout."""
        if self.cancellation_event.is_set():
            logger.info("skipping_turn_due_to_cancellation", target=turn.target, task=turn.task)
            return

        start_time = time.perf_counter()

        # 1. Generate Persona Reply Text via Groq
        text = await self.persona_manager.generate_reply(
            target=turn.target,
            task=turn.task,
            state=self.graph_manager.state,
            target_bot=turn.target,
        )

        if self.cancellation_event.is_set() or not text:
            return

        # Publish text reply back to room chat channel
        pub_room = self.sathi_room if (turn.target.lower() == "sathi" and self.sathi_room) else self.room
        if pub_room and getattr(pub_room, "local_participant", None):
            try:
                payload = f"[{turn.target.upper()}]: {text}".encode("utf-8")
                await pub_room.local_participant.publish_data(payload=payload, reliable=True, topic="chat")
            except Exception as exc:
                logger.debug("text_chat_publish_skipped", error=str(exc))

        # 2. Synthesize Speech via Azure Speech TTS
        pcm_bytes = await self.tts_synthesizer.synthesize(
            text=text,
            target_bot=turn.target,
            room=pub_room,
        )


        if self.cancellation_event.is_set() or not pcm_bytes:
            return

        if pcm_bytes == STATIC_FALLBACK_PCM and pub_room and getattr(pub_room, "local_participant", None):
            try:
                fb_payload = f"[{turn.target.upper()}]: Network issue ke karan audio delayed hai...".encode("utf-8")
                await pub_room.local_participant.publish_data(payload=fb_payload, reliable=True, topic="chat")
            except Exception:
                pass

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_structured_event(
            event_name="bot_turn_speaking",
            target_bot=turn.target,
            latency_ms=total_latency_ms,
            text=text,
            pcm_bytes_count=len(pcm_bytes),
        )

        # 3. Stream audio frames to persona's dedicated LiveKit track with 20ms pacing
        publisher = self.audio_publishers.get(turn.target.lower(), self.dost_publisher)
        completed = await publisher.play_audio(
            pcm_bytes=pcm_bytes,
            cancellation_event=self.cancellation_event,
            target_bot=turn.target,
        )

        if completed:
            # Append bot reply to room history
            turns = list(self.graph_manager.state.get("last_n_turns", []))
            if turns:
                turns[-1].setdefault("bot_replies", []).append({
                    "bot": turn.target,
                    "text": text,
                    "timestamp": time.time(),
                })

    async def handle_text_message(self, msg: str, sender: str) -> None:
        """Handle incoming room text chat message with JSON payload extraction and sliding-window deduplication."""
        if not msg:
            return

        cleaned_text = msg.strip()

        # Unwrap LiveKit chat JSON envelopes if payload is JSON
        if cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            try:
                data = json.loads(cleaned_text)
                if isinstance(data, dict):
                    extracted = data.get("message") or data.get("text")
                    if extracted and isinstance(extracted, str):
                        cleaned_text = extracted.strip()
            except Exception:
                pass

        if not cleaned_text:
            return

        # Sliding-window message deduplication (4.0 seconds)
        msg_key = f"{sender}:{cleaned_text.strip().lower()}"
        now = asyncio.get_event_loop().time()
        if hasattr(self, "_seen_messages"):
            # Purge entries older than 4.0 seconds
            self._seen_messages = {k: v for k, v in self._seen_messages.items() if now - v < 4.0}
        else:
            self._seen_messages = {}

        if msg_key in self._seen_messages:
            logger.info("ignoring_duplicate_text_message", key=msg_key)
            return
        self._seen_messages[msg_key] = now

        log_structured_event(
            event_name="utterance_received_text",
            speaker_identity=sender,
            source="text",
            text=cleaned_text,
            target_bot=self.target_bot,
        )
        await self.transcription_handler.normalize_and_dispatch(
            identity=sender,
            text=cleaned_text,
            source="text",
            target_bot=self.target_bot,
        )

    async def on_utterance_normalized(self, utterance: UtteranceNormalizer) -> None:
        """Process normalized utterance through the LangGraph router in real time."""
        if utterance.source == "text":
            await self._process_utterance_direct(utterance)
            return

        # Handle voice utterance with sentence debounce and fragment/connector trailing detection
        identity = utterance.identity
        self._transcript_buffers.setdefault(identity, []).append(utterance.text)

        if identity in self._debounce_tasks and not self._debounce_tasks[identity].done():
            self._debounce_tasks[identity].cancel()

        combined_draft = " ".join(self._transcript_buffers[identity]).strip()
        words = [w.strip(".,!?").lower() for w in combined_draft.split()]
        greetings = {"dost", "sathi", "ai dost", "ai sathi", "दोस्त", "साथी", "साथ ही", "सुनो", "hello", "hi", "hey"}

        # Incomplete words & conjunctions
        connector_phrases = [
            "aur phir", "aur saath hi", "batao aur", "dost", "sathi",
            "और फिर", "और साथ ही", "बताओ और", "दोस्त", "साथी"
        ]
        connector_words = [
            "aur", "phir", "ki", "hai", "saal", "and", "or", "then", "because", "so",
            "और", "फिर", "कि", "है", "साल", "या", "क्योंकि"
        ]

        draft_lower = combined_draft.lower().strip(".,!? ")

        is_short_fragment = len(words) < 3 and (
            any(draft_lower.endswith(cp) for cp in connector_phrases) or
            (len(words) > 0 and words[-1] in connector_words)
        )

        ends_with_connector = any(draft_lower.endswith(cp) for cp in connector_phrases) or (
            len(words) > 0 and words[-1] in connector_words
        )

        is_only_greeting = len(words) > 0 and all(w in greetings for w in words)

        # Debounce timing resolution:
        # - Isolated greeting (e.g. "AI Dost"): 2.50s
        # - Short fragment (< 3 words ending with incomplete word/conjunction): 0.60s (600ms hold)
        # - Compound trailing connector (>= 3 words ending with connector): 1.15s
        # - Short turn (<= 2 words): 0.60s
        # - Standard completed query: 0.35s
        if is_only_greeting:
            debounce_delay = 2.50
        elif is_short_fragment:
            debounce_delay = 0.60  # 600ms hold for < 3 words ending with incomplete word/conjunction
        elif ends_with_connector:
            debounce_delay = 1.15  # 1.15s delay for longer compound queries
        elif len(words) <= 2:
            debounce_delay = 0.60  # 600ms hold for short turns
        else:
            debounce_delay = 0.35  # Brief 350ms base debounce

        self._debounce_tasks[identity] = asyncio.create_task(
            self._process_debounced_utterance(identity, utterance.source, debounce_delay)
        )

    async def _process_debounced_utterance(
        self, identity: str, source: str, delay_sec: float = 1.20
    ) -> None:
        """Wait delay_sec seconds for settling silence before executing unified utterance turn plan."""
        try:
            await asyncio.sleep(delay_sec)
        except asyncio.CancelledError:
            return

        texts = self._transcript_buffers.pop(identity, [])
        if not texts:
            return

        combined_text = " ".join(texts)
        merged_utterance = UtteranceNormalizer(
            identity=identity,
            text=combined_text,
            source="voice",
            timestamp=time.time(),
        )
        await self._process_utterance_direct(merged_utterance)

    async def _process_utterance_direct(self, utterance: UtteranceNormalizer) -> None:
        """Direct execution of router decision for a complete utterance."""
        turn_plans = await self.graph_manager.process_utterance(utterance)
        active_topic = self.graph_manager.state.get("active_topic")

        # Check for halting/silence intents (no_reply or stop_command)
        has_stop = any(p.target == "no_reply" or p.reason == "stop_command" for p in turn_plans)
        if has_stop:
            logger.info(
                "halting_intent_detected_silence_enforced",
                speaker=utterance.identity,
                text=utterance.text,
            )
            self._transcript_buffers.pop(utterance.identity, None)
            self.cancellation_event.set()
            for pub in self.audio_publishers.values():
                pub.stop_current()
            self.queue_manager.cancel_current_and_clear()
            return

        # Trigger interruption on speaking bot for active turns
        self.trigger_barge_in_interruption(utterance.identity)
        self.cancellation_event.clear()

        logger.info(
            "live_turn_plan_generated",
            speaker=utterance.identity,
            source=utterance.source,
            text=utterance.text,
            turn_plans=[p.model_dump() for p in turn_plans],
            active_topic=active_topic,
        )

        # Enqueue generated turn plans for sequential execution
        if turn_plans:
            await self.queue_manager.add_turns(turn_plans)

        # Publish decision back to room data channel if connected
        if self.room and self.room.local_participant and turn_plans:
            summary_msg = f"[Router Decision] Topic: '{active_topic or 'None'}' | Turns: " + ", ".join(
                f"[{p.target}: {p.task}]" for p in turn_plans
            )
            try:
                await self.room.local_participant.publish_data(
                    summary_msg.encode("utf-8"),
                    reliable=True,
                )
            except Exception as exc:
                logger.debug("data_channel_publish_skipped", error=str(exc))

    async def run_job(self, ctx: JobContext) -> None:
        """LiveKit agent worker job runner."""
        logger.info(
            "agent_worker_connecting",
            url=settings.livekit_url,
            target_bot=self.target_bot,
        )
        await ctx.connect()
        if self.room is None and hasattr(ctx, "room"):
            self.room = ctx.room

        room_obj = getattr(self, "room", None) or getattr(ctx, "room", None)
        room_name = getattr(room_obj, "name", "") if room_obj else ""

        # For room_sid resolution:
        raw_sid = getattr(room_obj, "sid", "") if room_obj else ""
        if asyncio.iscoroutine(raw_sid) or inspect.isawaitable(raw_sid):
            room_sid = await raw_sid
        elif callable(raw_sid):
            res = raw_sid()
            room_sid = await res if inspect.isawaitable(res) else str(res)
        else:
            room_sid = str(raw_sid or "")

        logger.info(
            "agent_worker_connected",
            room_name=room_name,
            room_sid=room_sid,
        )

        # Explicitly set WebRTC participant display name in room roster
        if self.room and getattr(self.room, "local_participant", None):
            try:
                lp = self.room.local_participant
                if hasattr(lp, "update_name"):
                    res = lp.update_name("AI Dost")
                    if asyncio.iscoroutine(res) or inspect.isawaitable(res):
                        await res
            except Exception as exc:
                logger.debug("update_name_local_participant_failed", error=str(exc))

        if room_name and settings.livekit_api_key and settings.livekit_api_secret:
            try:
                lk_api = api.LiveKitAPI(settings.livekit_url, settings.livekit_api_key, settings.livekit_api_secret)
                local_ident = self.room.local_participant.identity if (self.room and hasattr(self.room, "local_participant") and self.room.local_participant) else ""
                if local_ident:
                    await lk_api.room.update_participant(
                        api.UpdateParticipantRequest(
                            room=room_name,
                            identity=local_ident,
                            name="AI Dost",
                        )
                    )
                    logger.info("primary_worker_display_name_updated", identity=local_ident, name="AI Dost")
                await lk_api.aclose()
            except Exception as exc:
                logger.debug("livekit_api_update_participant_name_failed", error=str(exc))

        # Lazily initialize and publish local bot audio tracks for both Dost and Sathi
        await self.dost_publisher.initialize_and_publish(self.room, track_name="dost_audio_track")

        # Connect secondary companion participant session for Sathi if credentials available
        if room_name:
            try:
                sathi_token = (
                    api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
                    .with_identity("AI Sathi")
                    .with_name("AI Sathi")
                    .with_grants(api.VideoGrants(room_join=True, room=room_name))
                    .to_jwt()
                )
                self.sathi_room = rtc.Room()
                await self.sathi_room.connect(settings.livekit_url, sathi_token)
                logger.info(
                    "sathi_companion_participant_connected",
                    identity="AI Sathi",
                    room=room_name,
                )
            except Exception as exc:
                logger.warning("sathi_companion_connection_failed", error=str(exc))

        sathi_target_room = self.sathi_room if self.sathi_room else self.room
        await self.sathi_publisher.initialize_and_publish(sathi_target_room, track_name="sathi_audio_track")

        def _is_bot(identity: Optional[str]) -> bool:
            if not identity:
                return False
            local_ident = self.room.local_participant.identity if (self.room and hasattr(self.room, "local_participant") and self.room.local_participant) else ""
            if local_ident and identity == local_ident:
                return True
            return identity in ["AI Dost", "AI Sathi", "Roxstar AI Dost", "Roxstar AI Sathi", "orchestrator"] or identity.startswith("AI ") or identity.startswith("Roxstar AI")

        # Register text stream handlers for topic 'lk.chat' and 'chat' to eliminate log warnings
        def _handle_lk_chat_stream(reader: rtc.TextStreamReader, participant_identity: str):
            async def _read_stream():
                try:
                    full_text = await reader.read_all()
                    if _is_bot(participant_identity):
                        return
                    asyncio.create_task(self.handle_text_message(full_text, participant_identity))
                except Exception as exc:
                    logger.debug("text_stream_read_failed", error=str(exc))
            asyncio.create_task(_read_stream())

        for topic_name in ["lk.chat", "chat"]:
            try:
                if hasattr(self.room, "register_text_stream_handler"):
                    self.room.register_text_stream_handler(topic_name, _handle_lk_chat_stream)
                if self.sathi_room and hasattr(self.sathi_room, "register_text_stream_handler"):
                    self.sathi_room.register_text_stream_handler(topic_name, _handle_lk_chat_stream)
            except Exception as exc:
                logger.debug("text_stream_handler_registration_skipped", error=str(exc))

        # Set up room event listeners
        @self.room.on("disconnected")
        def on_room_disconnected():
            if self.sathi_room:
                asyncio.create_task(self.sathi_room.disconnect())

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                if _is_bot(participant.identity):
                    logger.debug("ignoring_bot_self_audio_track", identity=participant.identity)
                    return
                logger.info(
                    "audio_track_subscribed",
                    participant=participant.identity,
                    track_sid=track.sid,
                )
                asyncio.create_task(
                    self.setup_participant_pipeline(participant.identity, track)
                )

        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(
                "participant_disconnected",
                participant=participant.identity,
            )
            asyncio.create_task(self.cleanup_participant_pipeline(participant.identity))

        @self.room.on("data_received")
        def on_data_received(data_packet: rtc.DataPacket):
            try:
                msg = data_packet.data.decode("utf-8")
                sender = data_packet.participant.identity if data_packet.participant else "Unknown"
                if _is_bot(sender):
                    return
                asyncio.create_task(self.handle_text_message(msg, sender))
            except Exception as e:
                logger.error("text_chat_failed", error=str(e))

        # Process existing participants already in the room
        for participant in self.room.remote_participants.values():
            if _is_bot(participant.identity):
                continue
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    await self.setup_participant_pipeline(
                        participant.identity, publication.track
                    )

    async def setup_participant_pipeline(
        self, identity: str, track: rtc.RemoteAudioTrack
    ) -> None:
        """Create and start an isolated pipeline for a human participant with VAD barge-in hook."""
        if identity in ["AI Dost", "AI Sathi", "Roxstar AI Dost", "Roxstar AI Sathi", "orchestrator"] or identity.startswith("AI ") or identity.startswith("Roxstar AI"):
            logger.debug("ignoring_bot_self_participant_pipeline", identity=identity)
            return

        if identity in self.pipelines:
            await self.cleanup_participant_pipeline(identity)

        pipeline = ParticipantAudioPipeline(
            participant_identity=identity,
            track=track,
            transcription_handler=self.transcription_handler,
            speech_started_callback=lambda p=identity: self.trigger_barge_in_interruption(p),
            target_bot=self.target_bot,
        )
        self.pipelines[identity] = pipeline
        await pipeline.start()

    async def cleanup_participant_pipeline(self, identity: str) -> None:
        """Stop and remove a participant audio pipeline."""
        if identity in self.pipelines:
            pipeline = self.pipelines.pop(identity)
            await pipeline.stop()


async def entrypoint(ctx: JobContext) -> None:
    """Top-level entrypoint for LiveKit worker jobs (lazy instantiation in child process)."""
    graph_manager = RoomStateGraphManager()
    worker_instance = VoiceAssistantWorker(graph_manager=graph_manager)
    await worker_instance.run_job(ctx)


def main():
    """CLI entrypoint for running the LiveKit worker process."""
    import sys
    import os

    target_room = os.getenv("LIVEKIT_ROOM", "room-voice-live")

    # Only inject --room when using the 'connect' command, never for 'dev' or 'start'
    if len(sys.argv) == 1:
        sys.argv.extend(["connect", "--room", target_room])
    elif len(sys.argv) > 1 and sys.argv[1] == "connect" and "--room" not in sys.argv:
        sys.argv.extend(["--room", target_room])

    options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="roxstar-voice-agent",
        ws_url=settings.livekit_url,
        api_key=settings.livekit_api_key,
        api_secret=settings.livekit_api_secret,
    )
    cli.run_app(options)



if __name__ == "__main__":
    main()
