"""In-process job runner. Isolated from request handlers.

Real torch training is CPU/GPU-bound, so it runs in a thread executor (never on
the event loop). Threads can't be force-killed, so cancellation is cooperative:
each run gets a threading.Event the training loop polls between steps.
"""

import asyncio
import logging
import threading

from training.char_train import train_char_model

logger = logging.getLogger("runner")

# run_id -> cancel event for the live training thread.
_cancels: dict[int, threading.Event] = {}


def launch_training(run_id: int) -> str:
    """Start the trainer in a worker thread and return an opaque job id."""
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    _cancels[run_id] = cancel

    future = loop.run_in_executor(None, train_char_model, run_id, cancel)

    def _cleanup(fut: asyncio.Future) -> None:
        _cancels.pop(run_id, None)
        exc = fut.exception()
        if exc is not None:
            logger.error("training for run %s failed", run_id, exc_info=exc)

    future.add_done_callback(_cleanup)
    return f"task-{run_id}"


def cancel_training(run_id: int) -> bool:
    """Signal a live training thread to stop. Returns True if one was running."""
    cancel = _cancels.get(run_id)
    if cancel is not None and not cancel.is_set():
        cancel.set()
        return True
    return False
