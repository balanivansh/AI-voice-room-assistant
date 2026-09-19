import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Literal, Optional, List
from src.logger import get_logger, log_structured_event

logger = get_logger("roxstar_voice_assistant.audio.transcription_handler")

UtteranceSource = Literal["voice", "text"]


@dataclass(frozen=True)
class UtteranceNormalizer:
    """Dataclass encapsulating a normalized user utterance from voice STT or chat text."""

    identity: str
    text: str
    source: UtteranceSource
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert utterance to dictionary representation."""
        return {
            "identity": self.identity,
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp,
        }


UtteranceCallback = Callable[[UtteranceNormalizer], Optional[Awaitable[None]]]


def normalize_transcript(text: str) -> str:
    """Normalize common STT mis-transcriptions for assistant names."""
    if not text:
        return text
    # Replace Deepgram's 'साथ ही' with 'साथी' when addressing assistant Sathi
    return text.replace("साथ ही", "साथी")


class TranscriptionHandler:
    """Normalizes, queues, and dispatches multi-speaker transcription & chat events."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[UtteranceNormalizer] = asyncio.Queue()
        self._callbacks: List[UtteranceCallback] = []

    def register_callback(self, callback: UtteranceCallback) -> None:
        """Register a callback hook to receive normalized utterances."""
        self._callbacks.append(callback)

    async def normalize_and_dispatch(
        self,
        identity: str,
        text: str,
        source: UtteranceSource,
        timestamp: Optional[float] = None,
        target_bot: str = "orchestrator",
    ) -> Optional[UtteranceNormalizer]:
        """Sanitize text, instantiate UtteranceNormalizer, log event, and dispatch.

        Args:
            identity: Speaker or sender participant identity string.
            text: Raw input text from speech transcription or data channel message.
            source: Source of transcript ('voice' or 'text').
            timestamp: Timestamp in seconds (defaults to current time.time()).
            target_bot: Bot identifier for logging context.

        Returns:
            Created UtteranceNormalizer instance, or None if text is empty/whitespace.
        """
        sanitized_text = text.strip() if text else ""

        # Unwrap LiveKit chat protocol JSON strings if source is text
        if source == "text" and sanitized_text.startswith("{") and sanitized_text.endswith("}"):
            try:
                data = json.loads(sanitized_text)
                if isinstance(data, dict):
                    extracted = data.get("message") or data.get("text")
                    if extracted and isinstance(extracted, str):
                        sanitized_text = extracted.strip()
            except Exception:
                pass

        # Normalize STT voice transcript errors (e.g. 'साथ ही' -> 'साथी')
        sanitized_text = normalize_transcript(sanitized_text)

        if not sanitized_text:
            logger.debug(
                "ignoring_empty_utterance",
                identity=identity,
                source=source,
                target_bot=target_bot,
            )
            return None

        ts = timestamp if timestamp is not None else time.time()
        utterance = UtteranceNormalizer(
            identity=identity,
            text=sanitized_text,
            source=source,
            timestamp=ts,
        )

        # Log event with speaker metadata
        log_structured_event(
            event_name=f"utterance_received_{source}",
            target_bot=target_bot,
            fallback_used=False,
            speaker_identity=identity,
            source=source,
            text_length=len(sanitized_text),
        )

        # Enqueue for downstream consumption
        await self._queue.put(utterance)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                res = cb(utterance)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:
                logger.error(
                    "utterance_callback_error",
                    error=str(exc),
                    identity=identity,
                    target_bot=target_bot,
                )

        return utterance

    async def get_next_utterance(self) -> UtteranceNormalizer:
        """Dequeue the next normalized utterance asynchronously."""
        return await self._queue.get()

    @property
    def queue_size(self) -> int:
        """Return current size of utterance queue."""
        return self._queue.qsize()
