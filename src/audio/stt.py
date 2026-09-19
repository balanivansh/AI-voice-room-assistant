from typing import Any, Optional
from livekit.plugins import deepgram
from src.config import settings
from src.logger import get_logger
from src.utils.fallback import safe_external_call

logger = get_logger("roxstar_voice_assistant.audio.stt")


def _init_deepgram_stt(
    model: str,
    language: str,
    api_key: str,
    interim_results: bool,
    punctuate: bool,
    smart_format: bool,
    filler_words: bool,
    **kwargs: Any,
) -> deepgram.STT:
    """Internal factory to create a Deepgram STT instance."""
    return deepgram.STT(
        model=model,
        language=language,
        api_key=api_key,
        interim_results=interim_results,
        punctuate=punctuate,
        smart_format=smart_format,
        filler_words=filler_words,
        **kwargs,
    )


async def create_stt(
    model: str = "nova-2",
    language: str = "hi",
    api_key: Optional[str] = None,
    interim_results: bool = True,
    punctuate: bool = True,
    smart_format: bool = True,
    filler_words: bool = True,
    target_bot: str = "orchestrator",
    **kwargs: Any,
) -> deepgram.STT:
    """Initialize and configure Deepgram streaming STT wrapped with safe execution logic.

    Args:
        model: Deepgram model name (defaults to 'nova-2' for multi-language code-switching).
        language: Language code ('hi' targeting Hindi/English code-switching).
        api_key: Deepgram API key (defaults to settings.deepgram_api_key).
        interim_results: Whether to emit interim partial transcription results.
        punctuate: Enable automatic punctuation.
        smart_format: Enable Deepgram smart formatting.
        filler_words: Include filler words in transcription output.
        target_bot: Bot identifier for logging context ('dost', 'sathi', 'orchestrator').
        **kwargs: Additional parameters passed to Deepgram STT initializer.

    Returns:
        Configured livekit.plugins.deepgram.STT instance.
    """
    key = api_key or settings.deepgram_api_key

    logger.info(
        "initializing_deepgram_stt",
        model=model,
        language=language,
        target_bot=target_bot,
    )

    stt_instance = await safe_external_call(
        service_name="deepgram_stt_init",
        func=_init_deepgram_stt,
        model=model,
        language=language,
        api_key=key,
        interim_results=interim_results,
        punctuate=punctuate,
        smart_format=smart_format,
        filler_words=filler_words,
        fallback_response=None,
        target_bot=target_bot,
        **kwargs,
    )

    if stt_instance is None:
        # If external initialization failed, create standard fallback instance
        logger.warning("using_fallback_stt_instance", target_bot=target_bot)
        stt_instance = deepgram.STT(
            model=model,
            language=language,
            api_key=key,
            interim_results=interim_results,
        )

    return stt_instance
