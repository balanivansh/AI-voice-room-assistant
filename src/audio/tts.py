import asyncio
import inspect
import json
import math
import os
import struct
import time
from typing import Any, Dict, Optional
import azure.cognitiveservices.speech as speechsdk
from src.config import settings
from src.logger import get_logger, log_structured_event
from src.utils.fallback import safe_external_call

logger = get_logger("roxstar_voice_assistant.audio.tts")

VOICE_MAPPING: Dict[str, str] = {
    "dost": "hi-IN-MadhurNeural",
    "sathi": "hi-IN-SwaraNeural",
}


def _generate_acknowledgment_cue_pcm() -> bytes:
    """Generate a 1.0s soft dual-tone acknowledgment audio cue (24kHz 16-bit mono raw PCM, 48000 bytes)."""
    sample_rate = 24000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    buf = bytearray()
    for i in range(num_samples):
        t = i / float(sample_rate)
        # Tone 1 (0-0.25s @ 587.33Hz D5), Tone 2 (0.3-0.55s @ 880Hz A5), Silence (0.55-1.0s)
        if t < 0.25:
            freq = 587.33
            env = math.sin(math.pi * (t / 0.25))
        elif 0.3 <= t < 0.55:
            t_rel = t - 0.3
            freq = 880.00
            env = math.sin(math.pi * (t_rel / 0.25))
        else:
            freq = 0.0
            env = 0.0
        val = int(3500.0 * env * math.sin(2.0 * math.pi * freq * t))
        val = max(-32768, min(32767, val))
        buf.extend(struct.pack("<h", val))
    return bytes(buf)


ACKNOWLEDGMENT_CUE_PCM: bytes = _generate_acknowledgment_cue_pcm()
STATIC_FALLBACK_PCM: bytes = ACKNOWLEDGMENT_CUE_PCM


async def _synthesize_azure_speech(
    speech_key: str,
    speech_region: str,
    voice_name: str,
    text: str,
    target_bot: str = "orchestrator",
) -> bytes:
    """Async helper function to invoke Azure Speech SDK synthesis without blocking the event loop."""
    if os.getenv("SIMULATE_TTS_FAILURE", "").lower() in ["true", "1"]:
        log_structured_event(
            event_name="provider_failure_simulated",
            provider="azure_tts",
            target_bot=target_bot,
        )
        raise RuntimeError("Simulated Azure TTS network failure triggered via SIMULATE_TTS_FAILURE env var.")

    def _sync_synthesize_once() -> bytes:
        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key,
            region=speech_region,
        )
        speech_config.speech_synthesis_voice_name = voice_name
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        else:
            cancellation = result.cancellation_details
            error_details = (
                cancellation.error_details if cancellation else "Unknown synthesis failure"
            )
            raise RuntimeError(f"Azure TTS synthesis failed: {cancellation.reason if cancellation else 'Cancelled'} - {error_details}")

    def _sync_synthesize() -> bytes:
        try:
            return _sync_synthesize_once()
        except Exception as exc:
            logger.warning(
                "azure_tts_connection_retry",
                error=str(exc),
                target_bot=target_bot,
            )
            return _sync_synthesize_once()

    try:
        return await asyncio.wait_for(asyncio.to_thread(_sync_synthesize), timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("azure_tts_synthesis_timed_out", timeout=15.0, target_bot=target_bot)
        raise RuntimeError("Azure TTS synthesis timed out after 15.0s")


class AzureSpeechSynthesizer:
    """Synthesizes text to 24kHz 16-bit mono raw PCM audio using Azure Speech SDK."""

    def __init__(
        self,
        speech_key: Optional[str] = None,
        speech_region: Optional[str] = None,
        room: Optional[Any] = None,
    ) -> None:
        self.speech_key = speech_key or settings.azure_speech_key
        self.speech_region = speech_region or settings.azure_speech_region
        self.room = room

    async def _post_text_to_chat(
        self,
        text: str,
        target_bot: str = "dost",
        room: Optional[Any] = None,
    ) -> None:
        """Post complete intended response text directly to LiveKit room text chat (lk.chat)."""
        target_room = room or self.room
        if not target_room or not getattr(target_room, "local_participant", None):
            return

        formatted_msg = f"[{target_bot.upper()}]: {text}"
        payload_bytes = json.dumps({
            "message": formatted_msg,
            "text": formatted_msg,
            "sender": target_bot.upper(),
            "topic": "lk.chat",
        }).encode("utf-8")

        try:
            res = target_room.local_participant.publish_data(
                payload=payload_bytes,
                reliable=True,
                topic="lk.chat",
            )
            if asyncio.iscoroutine(res) or inspect.isawaitable(res):
                await res
            logger.info(
                "tts_fallback_text_posted_to_chat",
                target_bot=target_bot,
                text_length=len(text),
            )
        except Exception as exc:
            logger.warning("tts_fallback_chat_post_failed", error=str(exc))

    async def synthesize(
        self,
        text: str,
        target_bot: str = "dost",
        room: Optional[Any] = None,
    ) -> bytes:
        """Synthesize input text into raw PCM bytes with persona voice mapping.

        Args:
            text: Spoken response text string.
            target_bot: Bot identifier ('dost' or 'sathi').
            room: Optional LiveKit room instance for text chat fallback.

        Returns:
            Raw 24kHz 16-bit mono PCM bytes object.
        """
        voice_name = VOICE_MAPPING.get(target_bot.lower(), "hi-IN-MadhurNeural")
        start_time = time.perf_counter()

        logger.info(
            "azure_tts_synthesizing",
            target_bot=target_bot,
            voice=voice_name,
            text_length=len(text),
        )

        pcm_bytes: bytes = await safe_external_call(
            f"azure_tts_{target_bot}",
            _synthesize_azure_speech,
            self.speech_key,
            self.speech_region,
            voice_name,
            text,
            target_bot,
            fallback_response=STATIC_FALLBACK_PCM,
            target_bot=target_bot,
        )

        if not pcm_bytes or pcm_bytes == STATIC_FALLBACK_PCM:
            pcm_bytes = STATIC_FALLBACK_PCM
            logger.warning(
                "azure_tts_fallback_triggered",
                target_bot=target_bot,
                reason="timeout_or_network_error",
                text=text,
            )
            await self._post_text_to_chat(text, target_bot, room=room)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_structured_event(
            event_name="azure_tts_completed",
            target_bot=target_bot,
            voice=voice_name,
            latency_ms=latency_ms,
            bytes_count=len(pcm_bytes),
        )

        return pcm_bytes

