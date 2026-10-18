"""Bounded "chooser" hooks: a plugin may pick one value for a decision the core would otherwise take
from config (``delegation_tier_chooser``, ``busy_input_chooser``).

Contract shared by every chooser hook:
- no plugin registered for the hook -> ``None`` at once (no thread, zero cost for unconfigured profiles);
- callbacks run on a daemon worker under a hard deadline; on timeout the worker is abandoned (never
  joined) and the caller gets ``None``;
- the first non-``None`` result that ``accept`` turns into a value wins; invalid answers are skipped;
- a raising callback is isolated by ``invoke_hook`` itself; anything else that goes wrong is ``None``.

``None`` always means "use the configured default", so a broken or slow plugin can only ever fall
back to today's behaviour.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def first_plugin_choice(
    hook_name: str, *, timeout_s: float, accept: Callable[[Any], Any], **kwargs: Any,
) -> Any:
    """Return the first accepted non-``None`` answer of *hook_name* within *timeout_s*, else ``None``."""
    try:
        from hermes_cli import plugins
        if not plugins.has_hook(hook_name):
            return None
    except Exception:
        logger.debug("chooser hook %s: plugin lookup failed", hook_name, exc_info=True)
        return None

    box: dict = {}
    done = threading.Event()
    ctx = contextvars.copy_context()  # profile/secret scope follows the caller

    def _work() -> None:
        try:
            box["results"] = ctx.run(plugins.invoke_hook, hook_name, **kwargs)
        except BaseException:  # noqa: BLE001 — a chooser can never break the caller
            logger.debug("chooser hook %s failed", hook_name, exc_info=True)
        finally:
            done.set()

    threading.Thread(target=_work, name=f"hook-{hook_name}", daemon=True).start()
    if not done.wait(timeout=max(0.0, float(timeout_s))):
        logger.info("chooser hook %s timed out after %.0f ms; using the configured default",
                    hook_name, float(timeout_s) * 1000)
        return None
    for result in box.get("results") or ():
        try:
            value = accept(result)
        except Exception:
            value = None
        if value is not None:
            return value
    return None
