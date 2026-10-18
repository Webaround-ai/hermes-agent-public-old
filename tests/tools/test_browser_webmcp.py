"""Exercise the shipped helper's JavaScript against a controlled page API."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.browser_webmcp import install

NODE = shutil.which("node") or next((str(p) for p in [Path.home() / "actions-runner-iollo-mac/externals/node24/bin/node"] if p.exists()), None)
pytestmark = pytest.mark.skipif(not NODE, reason="Node required for controlled page API")


class Page:
    def __init__(self):
        self.mode = "normal"
        self.version = 155
        self.calls = []
        self.tools = [{"name": "ARGUMENT", "description": "test", "inputSchema": {"type": "object"}}]
        self.api_calls = []
        self.namespace = {"current_tab": lambda: {"targetId": "target-a"}, "cdp": self.cdp}
        install(self.namespace)

    def cdp(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method == "Browser.getVersion":
            return {"product": f"Chrome/{self.version}.0"}
        if method == "Target.attachToTarget":
            return {"sessionId": "pinned"}
        if method == "Target.detachFromTarget":
            return {}
        assert method == "Runtime.evaluate"
        script = r"""
const [expression, mode, descriptors] = JSON.parse(process.argv[1]);
const calls=[]; global.window={}; global.location={origin:'https://example.test'};
const tools=descriptors.map(t=>({...t,window,origin:location.origin}));
if(mode==='foreign') tools.push({...tools[0],name:'foreign',origin:'https://foreign.test'});
if(mode==='other_window') tools.push({...tools[0],name:'other_window',window:{}});
global.document=mode==='absent'?{}:{modelContext:{
 getTools:async()=>{if(mode==='error')throw Error('discovery failed');if(mode==='hang')return new Promise(()=>{});return mode==='empty'?[]:tools;},
 executeTool:async(tool,args)=>{calls.push({name:tool.name,args});if(mode==='null')return null;
   if(mode==='execute_error')throw Error('execution failed');return {ok:true,args};}
}};
(async()=>{try {const value=await eval(expression);console.log(JSON.stringify({value,calls}));}
catch(e){console.log(JSON.stringify({error:String(e),calls}));}})();
"""
        result = subprocess.run([NODE, "-e", script, json.dumps([kwargs["expression"], self.mode, self.tools])],
                                capture_output=True, text=True, check=True, timeout=4)
        answer = json.loads(result.stdout)
        self.api_calls.extend(answer["calls"])
        if "error" in answer:
            return {"exceptionDetails": {"text": answer["error"]}}
        return {"result": {"value": answer["value"]}}


@pytest.mark.parametrize("mode,status", [("normal", "available"), ("absent", "no_api"), ("empty", "no_tools"),
                                         ("error", "discovery_error"), ("hang", "discovery_error")])
def test_discovery_states(mode, status):
    page = Page()
    page.mode = mode
    result = page.namespace["webmcp_tools"]()
    assert result["status"] == status
    assert not page.api_calls


@pytest.mark.parametrize("mode", ["foreign", "other_window"])
def test_discovery_ignores_other_origins_and_windows(mode):
    page = Page()
    page.mode = mode
    assert [t["name"] for t in page.namespace["webmcp_tools"]()["tools"]] == ["ARGUMENT"]


@pytest.mark.parametrize("version,expected", [(153, '{"SCHEMA_ONLY": "ARGUMENT"}'), (154, '{"SCHEMA_ONLY": "ARGUMENT"}'),
                                             (155, {"SCHEMA_ONLY": "ARGUMENT"})])
def test_calls_once_with_version_format_and_literal_input(version, expected):
    page = Page()
    page.version = version
    result = page.namespace["webmcp_call"]("ARGUMENT", {"SCHEMA_ONLY": "ARGUMENT"}, expected_origin="https://example.test")
    assert result["status"] == "returned"
    assert page.api_calls == [{"name": "ARGUMENT", "args": expected}]
    assert all(params["session_id"] == "pinned" for method, params in page.calls if method == "Runtime.evaluate")


def test_origin_and_refreshed_tools_prevent_stale_execution():
    page = Page()
    assert page.namespace["webmcp_tools"]()["status"] == "available"
    with pytest.raises(RuntimeError, match="origin changed"):
        page.namespace["webmcp_call"]("ARGUMENT", {}, expected_origin="https://changed.test")
    page.tools = []
    with pytest.raises(RuntimeError, match="unavailable"):
        page.namespace["webmcp_call"]("ARGUMENT", {}, expected_origin="https://example.test")
    assert not page.api_calls


def test_schema_refresh_and_bounds():
    page = Page()
    schema = page.namespace["webmcp_schema"]("ARGUMENT", expected_origin="https://example.test")
    assert schema["inputSchema"] == {"type": "object"}
    page.tools[0]["inputSchema"] = {"description": "a" * 33000}
    with pytest.raises(RuntimeError, match="output bound"):
        page.namespace["webmcp_schema"]("ARGUMENT", expected_origin="https://example.test")
    assert not page.api_calls


@pytest.mark.parametrize("mode", ["null", "execute_error"])
def test_uncertain_or_failed_execution_never_retries(mode):
    page = Page()
    page.mode = mode
    if mode == "null":
        assert page.namespace["webmcp_call"]("ARGUMENT", {}, expected_origin="https://example.test")["status"] == "unknown"
    else:
        with pytest.raises(RuntimeError, match="execution failed"):
            page.namespace["webmcp_call"]("ARGUMENT", {}, expected_origin="https://example.test")
    assert len(page.api_calls) == 1


def test_completion_stays_with_existing_purchase_gate():
    page = Page()
    with pytest.raises(RuntimeError, match="commit_purchase"):
        page.namespace["webmcp_call"]("complete_checkout", {}, expected_origin="https://example.test")
    assert not page.calls


def test_discovery_output_is_bounded():
    page = Page()
    page.tools = [{"name": str(n), "description": "x" * 10000} for n in range(50)]
    result = page.namespace["webmcp_tools"]()
    assert len(result["tools"]) == 40 and result["truncated"]
    assert len(result["tools"][0]["description"]) == 320
