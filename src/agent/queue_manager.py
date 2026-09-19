import asyncio
import time
from typing import Awaitable, Callable, List, Optional
from src.logger import get_logger, log_structured_event
from src.state.room_state import TurnPlan

logger = get_logger("roxstar_voice_assistant.agent.queue_manager")

TurnExecutor = Callable[[TurnPlan], Awaitable[None]]


class TurnQueueManager:
    """Manages persistent sequential execution of queued bot turns with event-driven execution and immediate handoff."""

    def __init__(
        self,
        executor: Optional[TurnExecutor] = None,
        ttl_seconds: float = 45.0,
    ) -> None:
        self.executor = executor
        self.ttl_seconds = ttl_seconds
        self._queue: asyncio.Queue[TurnPlan] = asyncio.Queue()
        self._current_turn: Optional[TurnPlan] = None
        self._active_turn_task: Optional[asyncio.Task] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._queue_updated = asyncio.Event()
        self._shutdown_event = asyncio.Event()

    def set_executor(self, executor: TurnExecutor) -> None:
        """Register turn executor callback function."""
        self.executor = executor

    async def add_turns(self, turns: List[TurnPlan]) -> None:
        """Add new turn plans to the execution queue and wake the worker loop."""
        for turn in turns:
            if turn.target == "no_reply":
                logger.info("skipping_no_reply_turn", reason=turn.reason)
                continue
            await self._queue.put(turn)
            log_structured_event(
                event_name="turn_enqueued",
                target_bot=turn.target,
                task=turn.task,
                queue_size=self._queue.qsize(),
            )

        # Launch persistent worker task if not running
        if self._worker_task is None or self._worker_task.done():
            self._shutdown_event.clear()
            self._worker_task = asyncio.create_task(self._queue_worker_loop())

        self._queue_updated.set()

    def get_next_pending_turn(self) -> Optional[TurnPlan]:
        """Fetch the next pending turn plan from the queue if available."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def _queue_worker_loop(self) -> None:
        """Persistent background worker loop processing queued turns sequentially."""
        while not self._shutdown_event.is_set():
            turn = self.get_next_pending_turn()
            if turn is not None:
                self._current_turn = turn
                log_structured_event(
                    event_name="turn_execution_started",
                    target_bot=turn.target,
                    task=turn.task,
                    reason=turn.reason,
                )

                try:
                    if self.executor:
                        self._active_turn_task = asyncio.create_task(self.executor(turn))
                        await self._active_turn_task
                except asyncio.CancelledError:
                    logger.info("active_turn_task_cancelled", target_bot=turn.target)
                except Exception as exc:
                    logger.error(
                        "turn_execution_error",
                        target_bot=turn.target,
                        error=str(exc),
                    )
                finally:
                    self._current_turn = None
                    self._active_turn_task = None
                    # Continue immediately to next turn in queue without sleeping
                    continue

            # Wait for next item to be enqueued or an interruption trigger
            self._queue_updated.clear()
            try:
                await self._queue_updated.wait()
            except asyncio.CancelledError:
                break

    def cancel_active_turn(self) -> None:
        """Cancel active turn playback task and trigger worker loop check."""
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
        self._queue_updated.set()

    def cancel_current_and_clear(self) -> int:
        """Interruption handler: cancel active turn task, clear queued turns, and wake worker loop.

        Returns:
            Count of cleared queued turns.
        """
        cleared_count = 0

        # Empty pending queued turns
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break

        # Cancel active turn execution task if running
        active_cancelled = False
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
            active_cancelled = True

        if cleared_count > 0 or active_cancelled or self._current_turn is not None:
            logger.info(
                "turn_queue_interrupted_and_cleared",
                cleared_count=cleared_count,
                active_turn_cancelled=active_cancelled or self._current_turn is not None,
            )

        self._queue_updated.set()
        return cleared_count

    def handle_interruption(self) -> int:
        """Alias for cancel_current_and_clear."""
        return self.cancel_current_and_clear()

    async def stop(self) -> None:
        """Shutdown queue manager worker loop cleanly."""
        self._shutdown_event.set()
        self._queue_updated.set()
        if self._active_turn_task and not self._active_turn_task.done():
            self._active_turn_task.cancel()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    @property
    def is_busy(self) -> bool:
        """Check if a turn is currently being executed or queued."""
        return self._current_turn is not None or not self._queue.empty()

    @property
    def queue_size(self) -> int:
        """Current number of items in the queue."""
        return self._queue.qsize()

