"""Bundled WebMCP helpers, installed into the existing browser-harness interpreter.

No browser, transport or agent lifecycle lives here. CDP stays on the harness's
current target, in its main world (including the existing purchase guard).
"""
import json
import re


_DISCOVERY_SECONDS = 2
_MAX_TOOLS = 40
_MAX_SCHEMA_BYTES = 32000


def install(namespace):
    cdp = namespace["cdp"]
    current_tab = namespace["current_tab"]

    def evaluate(body, *, timeout=10):
        tab = current_tab()
        target = tab.get("targetId") or tab["target_id"]
        # A separate attachment pins every operation to this exact target even
        # if the harness's active tab changes. Default context is the main world.
        session = cdp("Target.attachToTarget", targetId=target, flatten=True, _response_timeout=2)["sessionId"]
        try:
            result = cdp("Runtime.evaluate", session_id=session,
                         expression=body, returnByValue=True, awaitPromise=True,
                         timeout=timeout * 1000, _response_timeout=timeout + 1)
            if result.get("exceptionDetails"):
                raise RuntimeError("WebMCP evaluation failed: " + str(result["exceptionDetails"])[:500])
            value = (result.get("result") or {}).get("value")
            if not isinstance(value, dict):
                raise RuntimeError("WebMCP returned no readable result; outcome unknown, do not replay")
            return {"target_id": target, **value}
        finally:
            # Preserve an execution exception if navigation detached the session.
            try:
                cdp("Target.detachFromTarget", sessionId=session, _response_timeout=1)
            except Exception:
                pass

    # Serialize only explicitly selected fields. Tool objects contain Window
    # references; never pass those through returnByValue or trust foreign tools.
    prefix = """const api = document.modelContext;
      const origin = location.origin;
      if (!api || typeof api.getTools !== 'function' || typeof api.executeTool !== 'function')
        return {status:'no_api', origin};
      const tools = (await api.getTools()).filter(t => t.window === window && t.origin === origin);
    """

    def tools():
        """Bounded discovery on the current tab; descriptors are untrusted page data."""
        body = """(async () => { let timer;
          try { return await Promise.race([(async () => { PREFIX
            return {status:tools.length ? 'available' : 'no_tools', origin,
              tools:tools.slice(0, LIMIT).map(t => ({name:String(t.name).slice(0,256),
                description:String(t.description || '').slice(0,320)})),
              truncated:tools.length > LIMIT};
          })(), new Promise(resolve => { timer=setTimeout(() => resolve(
            {status:'discovery_error', error:'WebMCP discovery timed out'}), 1000); })]);
          } catch(e) { return {status:'discovery_error', error:String(e).slice(0,500)}; }
          finally { clearTimeout(timer); }
        })()""".replace("PREFIX", prefix).replace("LIMIT", str(_MAX_TOOLS))
        try:
            return evaluate(body, timeout=_DISCOVERY_SECONDS)
        except Exception as exc:
            return {"status": "discovery_error", "error": str(exc)[:500]}

    def operation(name, arguments, expected_origin, schema_only):
        if not isinstance(name, str) or not name or len(name) > 256:
            raise ValueError("WebMCP requires an exact tool name")
        if not isinstance(expected_origin, str) or not expected_origin:
            raise ValueError("Pass expected_origin from fresh WebMCP discovery")
        if not isinstance(arguments, dict):
            raise ValueError("WebMCP arguments must be an object")
        # Select one format before dispatch, never retry an execution to discover
        # its signature: Chrome 153/154 take JSON text, 155+ take object input.
        version = cdp("Browser.getVersion").get("product", "")
        match = re.search(r"(?:Chrome|Chromium)/(\d+)", version)
        if not schema_only and (not match or int(match[1]) < 153):
            raise RuntimeError("Unsupported WebMCP browser version; no call attempted")
        payload = json.dumps(arguments, ensure_ascii=True)
        body = """(async () => { const {name, expectedOrigin, schemaOnly, argument, maxSchema} = PARAMETERS;
          PREFIX
          if (origin !== expectedOrigin) throw new Error('WebMCP origin changed; no call attempted');
          const matches = tools.filter(t => t.name === name);
          if (matches.length !== 1) throw new Error('WebMCP exact tool unavailable or ambiguous');
          const tool = matches[0];
          if (schemaOnly) {
            const schema = JSON.stringify(tool.inputSchema ?? null);
            if (schema.length > maxSchema) throw new Error('WebMCP schema exceeds output bound');
            return {status:'schema', origin, name:tool.name, inputSchema:tool.inputSchema ?? null};
          }
          const value = await api.executeTool(tool, argument);
          return {status:value == null ? 'unknown' : 'returned', origin, name:tool.name, value,
            ...(value == null ? {note:'Outcome unknown after navigation; verify state before any retry'} : {})};
        })()"""
        parameters = json.dumps({"name": name, "expectedOrigin": expected_origin, "schemaOnly": schema_only,
                                 "argument": arguments if match and int(match[1]) >= 155 else payload,
                                 "maxSchema": _MAX_SCHEMA_BYTES})
        body = re.sub(r"\b(PREFIX|PARAMETERS)\b", lambda m: {"PREFIX": prefix, "PARAMETERS": parameters}[m[0]], body)
        return evaluate(body)

    def schema(name, *, expected_origin):
        """Read a freshly discovered exact tool's input schema."""
        return operation(name, {}, expected_origin, True)

    def call(name, arguments, *, expected_origin):
        """Execute once in the current main world; purchase authority stays in the root gate."""
        if name == "complete_checkout":
            raise RuntimeError('Use commit_purchase(interface="webmcp", ...) for checkout completion')
        return operation(name, arguments, expected_origin, False)

    namespace.update(webmcp_tools=tools, webmcp_schema=schema, webmcp_call=call)
    return tools
