"""Async helpers for Graphviz rendering with deferred logging."""

import atexit
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Tuple

_ENV_FLAGS = ("DEEPFLOW_VISUALIZE_GRAPHS", "DEEPFLOW_PERSIST_ARTIFACT_VIZ")


def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


class _GraphVizAsyncManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._tasks: List[Tuple[str, Future[Any], Optional[str]]] = []
        self._completed_messages: List[str] = []
        self._enabled: Optional[bool] = None
        atexit.register(self.wait_for_all)

    def _compute_enabled(self) -> bool:
        return any(_env_truthy(flag) for flag in _ENV_FLAGS)

    def is_enabled(self) -> bool:
        if self._enabled is None:
            self._enabled = self._compute_enabled()
        return self._enabled

    def _ensure_executor(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="graphviz")

    def submit(
        self,
        description: str,
        fn: Callable[..., Any],
        *args: Any,
        print_message: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Future[Any]]:
        if not self.is_enabled():
            fn(*args, **kwargs)
            if print_message:
                with self._lock:
                    self._completed_messages.append(print_message)
            return None

        self._ensure_executor()

        future = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._tasks.append((description, future, print_message))
        return future

    def wait_for_all(self) -> None:
        with self._lock:
            tasks = list(self._tasks)
            self._tasks = []

        completed: List[str] = []


        for description, future, message in tasks:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - best effort logging only
                print(f"[WARN] Graph visualization task '{description}' failed: {exc}")
            else:
                if message:
                    completed.append(message)

        with self._lock:
            if self._completed_messages:
                completed.extend(self._completed_messages)
                self._completed_messages = []

        if completed:
            print("[Debug] Graph visualizations generated:")
            for entry in completed:
                print(entry)

    def clear_cached_state(self) -> None:
        with self._lock:
            self._tasks = []
        self._enabled = None


_manager = _GraphVizAsyncManager()


def is_enabled() -> bool:
    return _manager.is_enabled()


def submit(description: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Optional[Future[Any]]:
    print_message = kwargs.pop("print_message", None)
    return _manager.submit(description, fn, *args, print_message=print_message, **kwargs)


def wait_for_all() -> None:
    _manager.wait_for_all()


def reset_for_tests() -> None:
    _manager.clear_cached_state()
