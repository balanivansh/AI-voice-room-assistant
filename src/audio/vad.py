from typing import Any
from livekit.agents.vad import VAD
from livekit.plugins import silero
from src.logger import get_logger

logger = get_logger("roxstar_voice_assistant.audio.vad")


def create_vad(
    min_speech_duration: float = 0.05,
    min_silence_duration: float = 1.75,
    prefix_padding_duration: float = 0.5,
    max_buffered_speech: float = 60.0,
    activation_threshold: float = 0.5,
    sample_rate: int = 16000,
    force_cpu: bool = True,
    **kwargs: Any,
) -> VAD:
    """Initialize and configure Silero VAD (Voice Activity Detection).

    Args:
        min_speech_duration: Minimum duration of speech to trigger VAD event (seconds).
        min_silence_duration: Minimum duration of silence to end VAD event (seconds).
        prefix_padding_duration: Duration of audio prepended to speech start (seconds).
        max_buffered_speech: Maximum buffer limit for speech audio (seconds).
        activation_threshold: Threshold probability (0.0 to 1.0) to activate speech.
        sample_rate: Audio sampling rate (8000 or 16000 Hz).
        force_cpu: Force CPU execution for ONNX model.
        **kwargs: Additional options passed to Silero VAD.load().

    Returns:
        Configured livekit.agents.vad.VAD instance.
    """
    logger.info(
        "initializing_silero_vad",
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        activation_threshold=activation_threshold,
        sample_rate=sample_rate,
    )
    return silero.VAD.load(
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        prefix_padding_duration=prefix_padding_duration,
        max_buffered_speech=max_buffered_speech,
        activation_threshold=activation_threshold,
        sample_rate=sample_rate,  # type: ignore
        force_cpu=force_cpu,
        **kwargs,
    )
