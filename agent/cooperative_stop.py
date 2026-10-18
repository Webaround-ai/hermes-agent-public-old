"""Cooperative cancellation: stop admission of new actions without cancelling in-flight work.

Cron installs the concrete durable check. Polls happen only at tool admission and
between iterations, never in a tool waiter or watchdog.
"""
STOPPED_BEFORE_ACTION = "Owner withdrawal confirmed before any action started."


def install_cooperative_stop(agent, check):
    agent._cooperative_stop_check = check
    agent._cooperative_action_started = False
    agent._cooperative_stop_observed = False


def note_action_started(agent):
    if vars(agent).get("_cooperative_stop_check") is not None:
        agent._cooperative_action_started = True


def stopped_before_action(agent):
    return (getattr(agent, "_cooperative_stop_observed", False) is True
            and getattr(agent, "_cooperative_action_started", True) is False)


def stop_requested(agent):
    check = vars(agent).get("_cooperative_stop_check")
    if check is None:
        return False
    # Failure to establish continuing authority fails closed at this boundary.
    try:
        requested = bool(check())
        if requested:
            agent._cooperative_stop_observed = True
        return requested
    except Exception:
        return True


def stop_at_iteration_boundary(agent):
    if stop_requested(agent):
        agent._interrupt_requested = True
        agent._interrupt_message = "Execution stopped; no further actions were started."
        return True
    return False
