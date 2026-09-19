import asyncio
import time
from typing import Optional
from livekit import rtc
from livekit.rtc import TrackPublishOptions, TrackSource
from src.logger import get_logger, log_structured_event

logger = get_logger("roxstar_voice_assistant.audio.audio_publisher")


class BotAudioPublisher:
    """Manages LiveKit local audio track publishing and 20ms frame streaming with interruption handling."""

    def __init__(
        self,
        sample_rate: int = 24000,
        num_channels: int = 1,
        bot_identity: str = "AI Dost",
        gender: str = "male",
    ) -> None:
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.bot_identity = bot_identity
        self.gender = gender
        self.source: Optional[rtc.AudioSource] = None
        self.track: Optional[rtc.LocalAudioTrack] = None
        self._published = False
        self._is_speaking = False
        self._stop_requested = False
        self._stop_event = asyncio.Event()
        self._keepalive_task: Optional[asyncio.Task] = None

    @property
    def is_speaking(self) -> bool:
        """Return True if bot is actively streaming audio frames."""
        return self._is_speaking

    def stop_current(self) -> None:
        """Immediately abort active audio playout and clear AudioSource queue."""
        self._stop_requested = True
        self._stop_event.set()
        if self.source:
            self.source.clear_queue()
        self._is_speaking = False

    def start_keepalive(self) -> None:
        """Start periodic background keepalive loop to maintain WebRTC/LiveKit connection during silences."""
        if self._keepalive_task is None or self._keepalive_task.done():
            self._keepalive_task = asyncio.create_task(self._run_keepalive())

    async def _run_keepalive(self) -> None:
        """Pump 20ms silence frame every 5s during idle periods to prevent LiveKit PONG timeouts."""
        silence_bytes = b"\x00" * 960  # 20ms of 24kHz 16-bit mono PCM
        samples_per_frame = int(self.sample_rate * 0.02)
        frame = rtc.AudioFrame(
            data=silence_bytes,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            samples_per_channel=samples_per_frame,
        )
        while True:
            try:
                await asyncio.sleep(5.0)
                if self.source and not self._is_speaking:
                    await self.source.capture_frame(frame)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("keepalive_silence_frame_skipped", error=str(exc))

    async def initialize_and_publish(self, room: rtc.Room, track_name: Optional[str] = None) -> None:
        """Lazily initialize AudioSource and LocalAudioTrack inside child job process after room connection."""
        if self._published:
            return

        t_name = track_name or f"{self.bot_identity.lower().replace(' ', '_')}_track"
        self.source = rtc.AudioSource(
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
        )
        self.track = rtc.LocalAudioTrack.create_audio_track(
            t_name,
            self.source,
        )

        if room and room.local_participant:
            options = TrackPublishOptions(source=TrackSource.SOURCE_MICROPHONE)
            await room.local_participant.publish_track(self.track, options)
            self._published = True
            self.start_keepalive()
            logger.info(
                "bot_audio_track_published",
                bot_identity=self.bot_identity,
                gender=self.gender,
                track_name=t_name,
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )

    async def play_audio(
        self,
        pcm_bytes: bytes,
        cancellation_event: asyncio.Event,
        target_bot: str = "dost",
    ) -> bool:
        """Stream raw 24kHz PCM bytes in 20ms chunks with cancellation monitoring and 20ms pacing.

        Args:
            pcm_bytes: 24kHz 16-bit mono PCM audio data bytes.
            cancellation_event: asyncio.Event set when barge-in interruption is triggered.
            target_bot: Bot identifier for logging context.

        Returns:
            True if audio playout completed cleanly, False if interrupted by barge-in.
        """
        if not self.source or not pcm_bytes:
            return False

        # 24000 Hz * 1 channel * 2 bytes/sample * 0.02 sec = 960 bytes per 20ms frame (480 samples)
        bytes_per_sample = 2
        samples_per_frame = int(self.sample_rate * 0.02)  # 480
        frame_bytes_count = samples_per_frame * self.num_channels * bytes_per_sample  # 960 bytes

        offset = 0
        total_bytes = len(pcm_bytes)
        start_time = time.perf_counter()

        logger.info(
            "bot_audio_playback_started",
            target_bot=target_bot,
            total_bytes=total_bytes,
        )

        self._is_speaking = True
        self._stop_requested = False
        self._stop_event.clear()

        try:
            while offset < total_bytes:
                # Check for immediate frame-level barge-in cancellation or stop_event before pushing frame
                if cancellation_event.is_set() or self._stop_event.is_set() or self._stop_requested:
                    if self.source:
                        self.source.clear_queue()
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    log_structured_event(
                        event_name="interruption_handled",
                        target_bot=target_bot,
                        elapsed_ms=elapsed_ms,
                        bytes_played=offset,
                        total_bytes=total_bytes,
                    )
                    return False

                chunk = pcm_bytes[offset : offset + frame_bytes_count]
                offset += len(chunk)

                # Pad final chunk if incomplete frame
                if len(chunk) < frame_bytes_count:
                    chunk = chunk + b"\x00" * (frame_bytes_count - len(chunk))

                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=self.sample_rate,
                    num_channels=self.num_channels,
                    samples_per_channel=samples_per_frame,
                )

                await self.source.capture_frame(frame)

                # Strict 20ms pacing to match audio buffer playback rate
                await asyncio.sleep(0.02)
        finally:
            self._is_speaking = False

        duration_sec = total_bytes / (self.sample_rate * bytes_per_sample)
        logger.info(
            "bot_audio_playback_completed",
            target_bot=target_bot,
            duration_sec=round(duration_sec, 2),
        )

        return True
