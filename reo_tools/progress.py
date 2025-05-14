import inspect
import sys
from collections.abc import Callable
from functools import wraps
from typing import Any

from alive_progress import alive_bar


def progress_bar(func: Callable[..., Any] = None, *, enabled: bool | Callable[[Any], bool] = True):
    """
    Wrap a function so that:
      • each source line steps the bar (bar())
      • each print(...) updates bar.text *and* prints below the bar
      • you can turn it on/off with a bool or a lambda(self)->bool
    """
    if func is None:
        return lambda f: progress_bar(f, enabled=enabled)

    src = inspect.getsource(func).splitlines()
    total = len([line for line in src if line.strip() and not line.strip().startswith("#")])

    @wraps(func)
    def wrapper(*args, **kwargs):
        is_enabled = not enabled(args[0]) if callable(enabled) else not enabled

        with alive_bar(
            total, enrich_print=False, monitor="{percent:.0%}", spinner=None, stats=False, disable=is_enabled
        ) as bar:

            def tracer(frame, event, arg):
                if event == "line" and frame.f_code is func.__code__:
                    bar()
                return tracer

            sys.settrace(tracer)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
                remaining = total - bar.current
                if remaining > 0:
                    bar(remaining)

    return wrapper
