"""The dock's Browser and agent-browser resolve to one identity: same executable, same user-data-dir —
and when a human opened that browser first, the agent attaches to it instead of launching a second one
(Chromium's profile singleton would forward the launch and kill it without a DevTools endpoint)."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from tools.bot_desktop import browser, runtime


def test_dock_and_agent_share_browser_identity(tmp_path, monkeypatch):
    exe = tmp_path / "chrome"
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(exe))
    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")

    dock_exe, dock_profile = browser.dock_launch()
    agent_env = browser.env_for_agent({})

    assert dock_exe == agent_env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(exe)
    assert dock_profile == agent_env["AGENT_BROWSER_PROFILE"] == str(tmp_path / "bot-desktop" / "browser-profile")


def test_user_pinned_profile_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", str(tmp_path / "mine"))
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    assert browser.profile_dir() == tmp_path / "mine"


def test_pinned_profile_honours_tilde_and_resolves_relative_paths_against_hermes_home(tmp_path, monkeypatch):
    """Regression for #110029: the docs say setting AGENT_BROWSER_PROFILE pins your own user-data-dir, but only
    an absolute value was honoured — `~/pin` and `pin` silently fell back to the default and the human's dock
    browser and the agent's browser could end up on different jars. A relative path is anchored where the rest
    of this profile's screen state lives (its HERMES_HOME), so two profiles never share one 'pin'."""
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(runtime, "get_hermes_home", lambda: home)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "user")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))

    monkeypatch.setenv("AGENT_BROWSER_PROFILE", "~/pin")
    assert browser.profile_dir() == tmp_path / "user" / "pin"
    monkeypatch.setenv("AGENT_BROWSER_PROFILE", "pin")
    assert browser.profile_dir() == home / "pin"
    assert browser.env_for_agent({})["AGENT_BROWSER_PROFILE"] == str(home / "pin"), "agent-browser gets the resolved path"


def test_dock_browser_advertises_a_devtools_port():
    """A human-started instance must be attachable, or the agent can never drive it afterwards."""
    assert "--remote-debugging-port=" in browser.dock_argv("/opt/chrome", "/p/dir")[2]


def test_dock_browser_caps_its_disk_cache():
    """The persistent profile lives on the gateway's disk (6 GB on a hosted instance); the HTTP cache
    must not be allowed to grow without bound there."""
    argv = browser.dock_argv("/opt/chrome", "/p/dir")
    cap = next(a for a in argv if a.startswith("--disk-cache-size="))
    assert 0 < int(cap.split("=", 1)[1]) <= 512 * 1024 * 1024


def _fake_running_instance(user_data_dir, pid: int, port: int) -> None:
    (user_data_dir / "DevToolsActivePort").write_text(f"{port}\n/devtools/browser/abc\n", encoding="utf-8")
    os.symlink(f"host-{pid}", user_data_dir / "SingletonLock")


def test_running_instance_port_requires_live_pid_and_open_port(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        _fake_running_instance(tmp_path, os.getpid(), port)
        assert browser.running_instance_cdp_port(str(tmp_path)) == port

        # Both files outlive a closed Chromium: a dead pid must not be trusted.
        os.unlink(tmp_path / "SingletonLock")
        os.symlink("host-2147483000", tmp_path / "SingletonLock")
        assert browser.running_instance_cdp_port(str(tmp_path)) is None
    finally:
        listener.close()
    # Live pid, port no longer accepting: still not attachable.
    os.unlink(tmp_path / "SingletonLock")
    os.symlink(f"host-{os.getpid()}", tmp_path / "SingletonLock")
    assert browser.running_instance_cdp_port(str(tmp_path)) is None
    assert browser.running_instance_cdp_port(str(tmp_path / "missing")) is None


def test_agent_attaches_to_human_started_browser(monkeypatch):
    """With a live dock instance on the shared profile the local argv carries ``--cdp <port>``; without one
    it stays a plain ``--session`` launch."""
    from tools import browser_tool_session as session

    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    monkeypatch.setattr(session._cloud, "_get_browser_engine", lambda: "auto")
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: False)
    monkeypatch.setattr(session, "_agent_browser_argv", lambda cmd: [cmd])
    argvs: list = []

    def spawn(task_id, session_info, cmd_parts, *rest):
        argvs.append(cmd_parts)
        return {"success": True}

    monkeypatch.setattr(session, "_spawn_and_collect", spawn)
    monkeypatch.setattr(session._lp, "_lightpanda_fallback_reason", lambda *a: None)
    info = {"session_name": "h_abc", "cdp_url": None, "features": {"local": True}}

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: 41234)
    session._dispatch_browser_command("t", info, "agent-browser", "open", ["https://x"], 10, None)
    assert argvs[-1][:5] == ["agent-browser", "--session", "h_abc", "--cdp", "41234"]

    monkeypatch.setattr(browser, "running_instance_cdp_port", lambda d, **kw: None)
    session._dispatch_browser_command("t", info, "agent-browser", "open", ["https://x"], 10, None)
    assert "--cdp" not in argvs[-1] and argvs[-1][:3] == ["agent-browser", "--session", "h_abc"]


def _install_browsers(tmp_path, monkeypatch, *, playwright: bool, system: bool):
    """A Playwright build under a private PLAYWRIGHT_BROWSERS_PATH and/or a system chromium on PATH."""
    monkeypatch.delenv("AGENT_BROWSER_EXECUTABLE_PATH", raising=False)
    roots = tmp_path / "pw"
    roots.mkdir(parents=True)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(roots))
    monkeypatch.setattr("tools.browser_tool_install._chromium_search_roots", lambda: [str(roots)])
    pw_exe = roots / "chromium-1200" / "chrome-linux" / "chrome"
    if playwright:
        pw_exe.parent.mkdir(parents=True)
        pw_exe.write_text("#!/bin/sh\n", encoding="utf-8")
        pw_exe.chmod(0o755)
    sys_exe = tmp_path / "bin" / "chromium"
    if system:
        sys_exe.parent.mkdir(parents=True)
        sys_exe.write_text("#!/bin/sh\n", encoding="utf-8")
        sys_exe.chmod(0o755)
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: str(sys_exe) if system and name == "chromium" else None)
    return str(pw_exe), str(sys_exe)


def test_unprivileged_user_under_apparmor_userns_restriction_gets_the_system_browser(tmp_path, monkeypatch):
    """Playwright's bundled Chromium has no setuid chrome_sandbox; with
    kernel.apparmor_restrict_unprivileged_userns=1 it dies 'FATAL: No usable sandbox!' for a non-root user,
    so the dock icon is dead. A distro chromium (which ships the sandbox helper) must win there — and
    the Playwright build stays the answer when it is the only one, started with exactly the flags
    agent-browser starts it with on that host (one sandbox policy for the human's and the bot's browser)."""
    pw_exe, sys_exe = _install_browsers(tmp_path, monkeypatch, playwright=True, system=True)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: True)
    assert browser.executable() == sys_exe

    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)
    assert browser.executable() == pw_exe  # unrestricted host: Playwright's build as before

    _install_browsers(tmp_path / "only-pw", monkeypatch, playwright=True, system=False)
    monkeypatch.setattr(browser, "_userns_restricted", lambda: True)
    exe = browser.executable()
    assert exe and exe.endswith("chrome-linux/chrome")
    from tools import browser_tool_session as session

    agent_env: dict = {}
    session._apply_chromium_sandbox_args(agent_env)
    agent_flags = set(agent_env.get("AGENT_BROWSER_ARGS", "").split(",")) - {""}
    assert agent_flags <= set(browser.dock_argv(exe, "/p/dir"))
    assert ("--no-sandbox" in browser.dock_argv(exe, "/p/dir")) == ("--no-sandbox" in agent_flags)


def test_root_dock_browser_starts_with_the_same_sandbox_args_as_the_agents_browser(monkeypatch):
    """Chromium refuses to start as root without --no-sandbox; agent-browser gets that flag from one
    policy, and the dock icon (same binary, same profile) must get the very same flags or the human's
    click dies while the agent's launch works."""
    from tools import browser_tool_session as session

    monkeypatch.setattr(session.os, "geteuid", lambda: 0)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 0)
    agent_env: dict = {}
    session._apply_chromium_sandbox_args(agent_env)
    agent_flags = set(agent_env["AGENT_BROWSER_ARGS"].split(","))
    assert agent_flags, "root must inject sandbox flags for agent-browser"
    assert agent_flags <= set(browser.dock_argv("/opt/chrome", "/p/dir"))

    # Same policy for the NON-root container case (the official image runs the gateway as uid 10000 in
    # Docker): agent-browser bypasses the sandbox there, and the dock icon must too or it dies on click.
    monkeypatch.setattr(session.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(browser.os, "geteuid", lambda: 10000)
    monkeypatch.setattr(session._install, "_running_in_docker", lambda: True)
    docker_env: dict = {}
    session._apply_chromium_sandbox_args(docker_env)
    assert set(docker_env["AGENT_BROWSER_ARGS"].split(",")) <= set(browser.dock_argv("/opt/chrome", "/p/dir"))
    monkeypatch.setattr(session._install, "_running_in_docker", lambda: False)
    monkeypatch.setattr(session, "apparmor_restricts_unprivileged_userns", lambda: False)
    assert "--no-sandbox" not in browser.dock_argv("/opt/chrome", "/p/dir")  # seated non-root host: sandboxed


def test_status_reports_the_headed_browser_or_its_absence(monkeypatch):
    """The official image ships only chromium_headless_shell: executable() is None and the dock silently
    has no Browser icon. Status must say so instead of leaving the pane to guess. The subject is the
    browser field, not host policy: status() gates it on is_supported_host(), which is pinned True so the
    test means the same thing on every CI lane."""
    monkeypatch.setattr(runtime, "is_supported_host", lambda: True)
    monkeypatch.setattr(runtime, "_launcher_pid", lambda: None)
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    monkeypatch.setattr(runtime, "geometry", lambda: "1440x900")
    monkeypatch.setattr(browser, "executable", lambda: None)
    assert runtime.status().as_dict()["browser"] is None
    monkeypatch.setattr(browser, "executable", lambda: "/usr/bin/chromium")
    assert runtime.status().browser == "/usr/bin/chromium"


def test_dock_exec_line_survives_spaces_in_the_executable_and_profile_paths():
    """The launcher used to split the shell line on the first space to find the executable, so a
    Chromium under '/opt/Google Chrome/' or a profile under a spaced HERMES_HOME broke the dock icon.
    Exec= follows the Desktop Entry spec: each argument double-quoted, with the reserved characters
    backslash-escaped inside the quotes."""
    exe = "/opt/Google Chrome/chrome"
    profile = '/home/a b/.hermes/browser "x"/profile'
    line = browser.dock_exec_line(exe, profile)
    assert line.startswith('Exec="/opt/Google Chrome/chrome" ')
    assert r'"--user-data-dir=/home/a b/.hermes/browser \\"x\\"/profile"' in line  # spec: \" quoted, then \ string-escaped
    assert "--remote-debugging-port=0" in line


def test_headless_shell_override_is_not_a_headed_browser(tmp_path, monkeypatch):
    """The official Docker image ships only Playwright's chrome-headless-shell and its boot hook exports it
    as AGENT_BROWSER_EXECUTABLE_PATH. That binary cannot open a window: taken at face value the dock's
    Browser icon would point at it and status would claim a headed browser exists. Live in the image:
    status.browser named the headless shell while the dock had no working Browser."""
    shell = tmp_path / "shell" / "chromium_headless_shell-1243" / "chrome-headless-shell-linux64" / "chrome-headless-shell"
    shell.parent.mkdir(parents=True)
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)
    _install_browsers(tmp_path, monkeypatch, playwright=False, system=False)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(shell))
    assert browser.executable() is None

    _, sys_exe = _install_browsers(tmp_path / "with-sys", monkeypatch, playwright=False, system=True)
    monkeypatch.setenv("AGENT_BROWSER_EXECUTABLE_PATH", str(shell))
    assert browser.executable() == sys_exe  # a real headed browser elsewhere still wins over the override


@pytest.mark.parametrize("engine, headed, starts", [("chrome", True, 1), ("chrome", False, 0), ("lightpanda", True, 0)])
def test_headed_chromium_spawn_asks_the_screen_to_start_but_the_env_builder_never_does(tmp_path, monkeypatch, engine, headed, starts):
    """Regression for #110050 at the right boundary: a real browser command that forks a headed Chromium daemon
    starts the profile's screen (bot_desktop.auto_start) like computer_use dispatch does; `_build_browser_env()`
    itself stays pure — it also serves the npx cache warmer, the Chromium auto-installer and the Lightpanda
    engine, none of which may block on Xvnc+Xfce coming up."""
    from tools import browser_tool as bt
    from tools import browser_tool_cloud as cloud
    from tools import browser_tool_session as session

    calls: list = []
    monkeypatch.setattr(runtime, "ensure_started_for_tool", lambda: calls.append(1))
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    monkeypatch.setattr(cloud, "_is_headed_mode", lambda: headed)
    bt._build_browser_env()
    assert calls == []

    class _Done:
        returncode = 0
        def wait(self, timeout=None): return 0
    def _fake_popen(argv, env, socket_dir, tag, stdin_payload=None):
        for slot in ("stdout", "stderr"):
            Path(socket_dir, f"_{slot}_{tag}").write_text("{}" if slot == "stdout" else "")
        return _Done()
    monkeypatch.setattr(session, "_popen_agent_browser", _fake_popen)
    monkeypatch.setattr(session, "_prepare_session_socket_dir", lambda name: str(tmp_path))
    session._spawn_and_collect("t", {"session_name": "h_x"}, ["agent-browser"], "open", engine, 5)
    assert len(calls) == starts


def test_janitor_keeps_the_shared_browser_alive_while_a_human_holds_the_lease(monkeypatch):
    """#110064: the agent goes idle BECAUSE the human took over to log in; the janitor must count the human's
    lease as activity for the shared local browser, and reap it again once the lease is handed back."""
    from tools import browser_tool_lifecycle as lifecycle, browser_tool_session as session
    from tools.bot_desktop import lease

    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    # The HUMAN exemption also protects local sessions outside headed desktop ownership.
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: False)
    holder = {"human": True}
    monkeypatch.setattr(lease, "human_holds", lambda *a, **k: holder["human"])
    reaped: list = []
    monkeypatch.setattr(lifecycle, "cleanup_browser", lambda task_id: reaped.append(task_id))
    monkeypatch.setattr(lifecycle._bt, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 1)
    monkeypatch.setattr(lifecycle._bt, "_active_sessions", {
        "bot": {"session_name": "h_bot", "features": {"local": True}},
        "cloud": {"session_name": "c_1", "bb_session_id": "bb", "features": {}},
    })
    monkeypatch.setattr(lifecycle._bt, "_session_last_activity", {"bot": 0.0, "cloud": 0.0})
    monkeypatch.setattr(lifecycle._bt, "_session_owner_homes", {})

    lifecycle._cleanup_inactive_browser_sessions()
    assert reaped == ["cloud"], "only the browser the human is not typing into may be reaped"
    assert lifecycle._bt._session_last_activity["bot"] > 0, "the human's lease refreshed the bot browser's activity"

    holder["human"] = False
    lifecycle._bt._session_last_activity["bot"] = 0.0
    lifecycle._cleanup_inactive_browser_sessions()
    assert reaped == ["cloud", "bot"]
    assert session.human_holds_shared_browser({"features": {"local": True}}) is False


def test_daemon_idle_timer_defers_to_the_janitor_only_for_the_shared_headed_browser(monkeypatch):
    """The agent-browser daemon cannot see the lease, so on the Bot Desktop its self-termination timer steps
    back and the desktop owns the browser's idle lifetime; headless browsing keeps the mirror."""
    from tools import browser_tool_session as session

    monkeypatch.setattr(session._bt, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 120)
    monkeypatch.setattr(runtime, "published_env", lambda: {"DISPLAY": ":37"})
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: True)
    assert session._daemon_idle_timeout_seconds() > 120
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: False)
    assert session._daemon_idle_timeout_seconds() == 120
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: True)
    monkeypatch.setattr(runtime, "published_env", lambda: {})
    assert session._daemon_idle_timeout_seconds() == 120


def test_a_headless_shell_pin_is_replaced_while_a_screen_is_up(tmp_path, monkeypatch):
    """The boot hook exports a chrome-headless-shell path; leaving it would put the agent and the dock on
    two binaries over one --user-data-dir, where the singleton swallows the dock's launch."""
    shell = tmp_path / "chrome-headless-shell"
    shell.write_text("#!/bin/sh\n", encoding="utf-8")
    shell.chmod(0o755)
    headed = tmp_path / "chrome"
    headed.write_text("#!/bin/sh\n", encoding="utf-8")
    headed.chmod(0o755)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    monkeypatch.setattr(browser, "_playwright_executable", lambda: str(headed))
    monkeypatch.delenv("AGENT_BROWSER_PROFILE", raising=False)
    # Unpinned, the ubuntu runner (non-root, userns-restricted) flips executable() to
    # its own /usr/bin/google-chrome; host policy is not the subject here.
    monkeypatch.setattr(browser, "_userns_restricted", lambda: False)

    agent_env = browser.env_for_agent({"AGENT_BROWSER_EXECUTABLE_PATH": str(shell)})
    dock_exe, _ = browser.dock_launch()
    assert agent_env["AGENT_BROWSER_EXECUTABLE_PATH"] == dock_exe == str(headed), \
        "the agent and the dock must share one binary once a screen is up"


def test_a_real_user_pin_is_still_honoured(tmp_path, monkeypatch):
    """Only a headless-shell pin is overridden; a human's own headed browser stays put."""
    mine = tmp_path / "my-chrome"
    mine.write_text("#!/bin/sh\n", encoding="utf-8")
    mine.chmod(0o755)
    other = tmp_path / "chrome"
    other.write_text("#!/bin/sh\n", encoding="utf-8")
    other.chmod(0o755)
    monkeypatch.setattr(runtime, "state_dir", lambda: tmp_path / "bot-desktop")
    monkeypatch.setattr(browser, "_playwright_executable", lambda: str(other))

    env = browser.env_for_agent({"AGENT_BROWSER_EXECUTABLE_PATH": str(mine)})
    assert env["AGENT_BROWSER_EXECUTABLE_PATH"] == str(mine)


@pytest.fixture
def desktop_browser_lifecycle(tmp_path, monkeypatch):
    from tools import browser_tool_lifecycle as lifecycle, browser_tool_session as session
    from tools.bot_desktop import lease

    for attr in ("_active_sessions", "_session_last_activity", "_session_owner_homes",
                 "_cleanup_failures", "_last_active_session_key", "_suspect_browser_sessions"):
        monkeypatch.setattr(lifecycle._bt, attr, {})
    monkeypatch.setattr(lifecycle._bt, "_recording_sessions", set())
    monkeypatch.setattr(lifecycle._bt, "BROWSER_SESSION_INACTIVITY_TIMEOUT", 120)
    monkeypatch.setattr(lifecycle._bt, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(lifecycle._bt, "_socket_safe_tmpdir", lambda: str(tmp_path))
    monkeypatch.setattr(lifecycle._bt, "_maybe_stop_recording", lambda *_a: None)
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: True)
    monkeypatch.setattr(session._cloud, "_get_cloud_provider", lambda: None)
    monkeypatch.setattr(session._real_profile, "_real_profile_cdp", lambda: (None, None))
    monkeypatch.setattr(lifecycle._bt, "_is_browser_use_cli_mode", lambda: False)
    monkeypatch.setattr(lifecycle, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(session._cdp, "_stop_cdp_supervisor", lambda *_a: None)
    monkeypatch.setattr(lease, "human_holds", lambda: False)
    now = [1000.0]
    monkeypatch.setattr(lifecycle.time, "time", lambda: now[0])
    screen = {"DISPLAY": ":37"}
    monkeypatch.setattr(runtime, "published_env", lambda: dict(screen))
    closed = []
    monkeypatch.setattr(session, "_run_browser_command",
                        lambda task_id, *a, **kw: closed.append(task_id) or {"success": True})
    return lifecycle, session, now, screen, closed


def test_shared_headed_browser_outlives_agent_idle_and_cleans_after_desktop_stop(desktop_browser_lifecycle):
    lifecycle, session, now, screen, closed = desktop_browser_lifecycle
    # Cloud bootstrap registers its local session BEFORE starting the screen.
    screen.clear()
    key = "iollo-browser-lifecycle::local"
    original = session._get_session_info(key)
    screen["DISPLAY"] = ":37"
    lifecycle._bt._active_sessions["private"] = {
        "session_name": "private", "bb_session_id": None, "features": {"cdp_override": True}}
    lifecycle._bt._session_last_activity["private"] = now[0]
    now[0] += 15 * 60  # approval wait well past the 120 second browser timeout
    lifecycle._cleanup_inactive_browser_sessions()
    assert closed == ["private"]
    assert lifecycle._bt._active_sessions[key] is original
    assert lifecycle._bt._session_last_activity[key] == 1000.0  # no fake heartbeat
    lifecycle._cleanup_inactive_browser_sessions()
    assert closed == ["private"]

    screen.clear()  # desktop stop/death releases lifetime ownership
    lifecycle._cleanup_inactive_browser_sessions()
    assert closed == ["private", key]
    assert key not in lifecycle._bt._active_sessions
    assert key not in lifecycle._bt._session_last_activity
    screen["DISPLAY"] = ":37"  # display reuse does not restore the old session
    replacement = session._get_session_info(key)
    assert replacement["session_name"] != original["session_name"]
    lifecycle.cleanup_browser(key)  # explicit close still works while the screen runs
    assert key not in lifecycle._bt._active_sessions
    assert closed == ["private", key, key]


@pytest.mark.parametrize("info,headed", [
    ({"features": {"local": True, "lightpanda": True}}, True),
    ({"features": {"local": True, "real_profile": True}}, True),
    ({"features": {"local": True}, "cdp_url": "ws://independent"}, True),
    ({"features": {"local": True}, "bb_session_id": "cloud"}, True),
    ({"features": {"cdp_override": True}}, True),
    ({"features": {"local": True}}, False),
])
def test_running_desktop_does_not_own_other_browser_idle_lifetimes(
        desktop_browser_lifecycle, monkeypatch, info, headed):
    lifecycle, session, now, _screen, _closed = desktop_browser_lifecycle
    monkeypatch.setattr(session._cloud, "_is_headed_mode", lambda: headed)
    assert session.desktop_owns_browser_lifetime(info) is False
    lifecycle._bt._active_sessions["other"] = info
    lifecycle._bt._session_last_activity["other"] = now[0]
    cleaned = []
    monkeypatch.setattr(lifecycle, "cleanup_browser", lambda key: cleaned.append(key))
    now[0] += 121
    lifecycle._cleanup_inactive_browser_sessions()
    assert cleaned == ["other"]
