#!/usr/bin/env python3
"""Boot the relocated gateway and its real HTTP adapter with isolated user state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def smoke(root):
    python = root / "python/bin/python3"
    launcher = root / "bin/hermes"
    # Darwin's sockaddr_un is short; the default /var/folders/... TMPDIR can
    # overflow it once Hermes adds its control-socket basename.
    with tempfile.TemporaryDirectory(prefix="iollo-smoke-", dir="/tmp") as scratch:
        work = Path(scratch)
        home = work / "home"
        state = work / "state"
        home.mkdir()
        state.mkdir()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = {
            "model": {"provider": "openai", "default": "iollo-smoke", "base_url": "http://127.0.0.1:1/v1"},
            "gateway": {"multiplex_profiles": False},
            "agent": {"gateway_startup_warmup_timeout": 0},
            "platforms": {"api_server": {"enabled": True, "host": "127.0.0.1", "port": port}},
        }
        # JSON is also YAML; no additional tooling is needed to prepare the probe.
        (state / "config.yaml").write_text(json.dumps(config), encoding="utf-8")
        key = secrets.token_hex(32)
        env = {"HOME": str(home), "HERMES_HOME": str(state), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
               "LANG": "C.UTF-8", "API_SERVER_KEY": key, "OPENAI_API_KEY": "sk-iollo-smoke-unused",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
        def run(args):
            subprocess.run([str(arg) for arg in args], env=env, cwd=work, check=True, timeout=120)

        # hermes-agent is a distribution name, not an import named hermes_agent.
        run([python, "-I", "-B", "-c", "import hermes_cli, run_agent, cli, agent, gateway, "
             "gateway.run, gateway.platforms.api_server, hermes_state, hermes_constants, "
             "model_tools, toolsets, tools, cron, tui_gateway, acp_adapter, plugins, providers; "
             "import iollo_envelope.runtime, iollo_envelope.producer; "
             "import aiohttp, fastapi, uvicorn, mcp, telegram, discord, slack_bolt; "
             "from pathlib import Path; import sys; "
             "assert Path(run_agent.__file__).resolve().is_relative_to(Path(sys.prefix).parent); "
             "print('Bundled imports OK:', sys.executable)"])
        run([launcher, "--version"])
        run([launcher, "gateway", "--help"])
        run([launcher, "gateway", "run", "--help"])
        run([python, "-I", "-B", "-m", "gateway.run", "--help"])
        run([python, "-I", "-m", "pip", "check"])
        # Exercises rewritten console-script shebangs after relocation, including spaces.
        run([root / "python/bin/pip3", "--version"])
        log_path = work / "gateway.log"
        with log_path.open("w+") as log:
            process = subprocess.Popen([str(launcher), "gateway", "run"], env=env, cwd=work,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 120
                request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                                                 headers={"Authorization": f"Bearer {key}"})
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"Gateway exited with {process.returncode}")
                    try:
                        with urllib.request.urlopen(request, timeout=2) as response:
                            payload = json.load(response)
                            assert response.status == 200 and payload["object"] == "list" and payload["data"]
                            status_path = state / "gateway_state.json"
                            if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8")).get("gateway_state") == "running":
                                print("GET /v1/models: HTTP 200", json.dumps(payload), flush=True)
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(0.25)
                else:
                    raise RuntimeError("Gateway did not answer /v1/models within 120 seconds")
                for headers, expected in (({}, 401), ({"Authorization": f"Bearer {key}"}, 404)):
                    probe = urllib.request.Request(f"http://127.0.0.1:{port}/v1/runs/fake/envelope",
                                                   headers=headers)
                    try:
                        urllib.request.urlopen(probe, timeout=5)
                    except urllib.error.HTTPError as error:
                        assert error.code == expected, (error.code, expected)
                    else:
                        raise AssertionError(f"Envelope probe should return {expected}")
            except BaseException:
                log.flush()
                print(log_path.read_text(encoding="utf-8", errors="replace"), file=sys.stderr)
                raise
            finally:
                if process.poll() is None:
                    # Upstream deliberately returns 1 for an unplanned SIGTERM.
                    # SIGINT is the documented foreground gateway stop path.
                    os.killpg(process.pid, signal.SIGINT)  # windows-footgun: ok — macOS-only bundle smoke
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok — macOS-only bundle smoke
                        process.wait(timeout=10)
            if process.returncode != 0:
                raise RuntimeError(f"Gateway shutdown failed: {process.returncode}\n{log_path.read_text()}")
        print("Relocated gateway/API smoke passed; temporary user state removed.")


if __name__ == "__main__":
    smoke(Path(sys.argv[1]).resolve())
