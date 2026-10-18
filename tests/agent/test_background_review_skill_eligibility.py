"""Current-turn evidence drives learning without a workflow classifier."""
from types import SimpleNamespace

import pytest

from agent.background_review import should_review_skills


def _result(name, content='{}'):
    return [
        {"role": "assistant", "tool_calls": [{"id": name, "function": {"name": name}}]},
        {"role": "tool", "tool_call_id": name, "content": content},
    ]


@pytest.mark.parametrize("tool", ["terminal", "browser_click", "search_answer", "execute_code"])
def test_short_actual_tool_work_reviews_but_history_and_chatter_do_not(tool):
    agent = SimpleNamespace(_skill_nudge_interval=10, _iters_since_skill=1, valid_tool_names={"skill_manage"})
    current = [{"role": "user", "content": "Do the task"}] + _result(tool)
    assert should_review_skills(agent, current)
    assert agent._iters_since_skill == 0
    assert not should_review_skills(agent, current + [{"role": "user", "content": "Thanks"}])
    assert not should_review_skills(agent, [{"role": "user"}] + _result("skill_view"))


@pytest.mark.parametrize("success", [True, False])
def test_only_successful_foreground_skill_write_suppresses_review(success):
    agent = SimpleNamespace(_skill_nudge_interval=10, _iters_since_skill=12, valid_tool_names={"skill_manage"})
    messages = [{"role": "user"}] + _result("terminal") + _result("skill_manage", '{"success": ' + str(success).lower() + '}')
    assert should_review_skills(agent, messages) is not success
    agent._skill_nudge_interval = 0
    assert not should_review_skills(agent, [{"role": "user"}] + _result("terminal"))


@pytest.mark.parametrize("flag", [
    "_verification_stop_synthetic", "_pre_verify_synthetic", "_kanban_stop_synthetic",
    "_empty_recovery_synthetic", "_dropped_toolcall_nudge", "_todo_snapshot_synthetic",
])
def test_runtime_nudge_keeps_short_tool_evidence_until_genuine_new_user_turn(flag):
    agent = SimpleNamespace(_skill_nudge_interval=10, _iters_since_skill=1, valid_tool_names={"skill_manage"})
    messages = [{"role": "user", "content": "Complete this task"}] + _result("terminal") + [
        {"role": "user", "content": "Verify before finishing", flag: True},
        {"role": "assistant", "content": "Verified; done"},
    ]
    assert should_review_skills(agent, messages)
    messages.append({"role": "user", "content": "What is the weather?"})
    assert not should_review_skills(agent, messages)
