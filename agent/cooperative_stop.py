"""Cooperative cancellation: stop admission of new actions without cancelling in-flight work.

Cron installs the concrete durable check. Polls happen only at tool admission and
between iterations, never in a tool waiter or watchdog.
"""
def install_cooperative_stop(agent, check):
    agent._cooperative_stop_check = check


def stop_requested(agent):
    check = getattr(agent, "_cooperative_stop_check", None)
    if check is None:
        return False
    # Failure to establish continuing authority fails closed at this boundary.
    try:
        return bool(check())
    except Exception:
        return True


def stop_at_iteration_boundary(agent):
    if stop_requested(agent):
        agent._interrupt_requested = True
        agent._interrupt_message = "Owner requested stopping; no further actions were started."
        return True
    return False
