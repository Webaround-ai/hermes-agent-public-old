# Iollo runtime integration

Source review 2026-10-04: Hermes `e3682503cf` (`iollo-2026.9.24-rc13`, `origin/iollo` HEAD); cloud
`0e62700`. Tags rc11, rc12 and rc13 are PRs #22, #23 and #24: per-run `tool_profile` (incl. `read_tools` globs and
`hosted_tools`), opt-in `ends_turn` with commentary promotion on the Responses path, `tool.generating` / `run.heartbeat`
events, and `skip_background_review`. The earlier review (2026-10-03) was of rc10 at cloud `af8ee4c`. This file is the fork entrypoint, not a parallel product architecture.

Read cloud [ARCHITECTURE](https://github.com/Webaround-ai/iollo/blob/main/docs/ARCHITECTURE.md),
[WEBAROUND-ROUTER](https://github.com/Webaround-ai/iollo/blob/main/docs/WEBAROUND-ROUTER.md),
[CONVERSATIONS](https://github.com/Webaround-ai/iollo/blob/main/docs/CONVERSATIONS.md) and
[DATA-RETENTION](https://github.com/Webaround-ai/iollo/blob/main/docs/DATA-RETENTION.md).
These are the canonical cloud guides; check their dated source and live evidence before operations.
For native TypeSafe transport and shim retirement, read cloud
[JEV](https://github.com/Webaround-ai/iollo/blob/main/docs/JEV.md).

Iollo runs the personal agent on one Linux machine/volume per owner. Web is a device client; Mac is
menu bar/local tools. iOS and Desktop Pro are parked. Upstream room, desktop, messaging or local-agent
features are not evidence that Iollo exposes them. Sessions/delegates share owner filesystem, memory,
browser and services; distinct session IDs do not isolate resources for multiple people.

Cloud `profile_policy.py` owns managed models/persona/permissions; `sandbox/boxsync.py` installs them
under root-owned `/etc/hermes`. Provider master keys stay at Bifrost; the managed runtime receives a
per-user virtual key. Do not tell operators to use `hermes model` as a persistent policy override.
The cloud repo's `sandbox/hermes-base.txt` pins rc13 at this review (rc10 at the 2026-10-03 review); `sandbox/Dockerfile`
has the same fallback and the sandbox workflow selects an explicit train override or that pin.
Record the final sandbox image/digest and rollout separately: this fork's branch HEAD or tag alone
does not identify a deployed image. See cloud
[DEPLOY](https://github.com/Webaround-ai/iollo/blob/main/docs/DEPLOY.md) and
[RELEASES](https://github.com/Webaround-ai/iollo/blob/main/docs/RELEASES.md).

| Runtime seam | Source | Integration rule |
|---|---|---|
| Durable runs | `gateway/platforms/api_server_runs.py` | Preserve declared session key, idempotency, provider/effort, envelope context and prompt version. Each synchronous API-run worker executes with an isolated copy of its submitting task's `ContextVar` context, so trusted request provenance remains available to tools without leaking between runs. |
| Tool profile / turn-ending tools | `agent/tool_profile.py`, `agent/turn_tool_round.py`, `agent/turn_api_request.py` | Per-run `tool_profile` projects requests only (agent, stored prompt, pin stay full); `ends_turn` ends a succeeded text+tool round; rc12: globs and `read_tools`; rc13: `hosted_tools` (annotated read-only MCP tools, budgeted) and per-run `skip_background_review`; see [IOLLO-FORK](../IOLLO-FORK.md); recovery source `d4e61c0be5` materializes explicitly selected deferred tools from the scoped raw catalog for each request, preserves the stored tool pin, and grants direct dispatch only for that projection. Catalog resolution errors fail visibly. |
| Successful execution handoff | `tools/registry.py`, `agent/tool_executor.py`, `agent/turn_tool_round.py` | Source on `codex/purchase-execution-ownership`, based on `bd5fd42fccba`: a trusted `execution_handoff(args, result)` callback closes an accepted immediate transfer after durable result recording and before later sibling actions; broker identity and skipped-call pairing remain canonical. Independent of ordinary reply `ends_turn`; no browser lease or extra planner. See [IOLLO-FORK](../IOLLO-FORK.md#native-execution-handoff--source-change-2026-10-18). |
| Prompt refresh | `agent/prompt_version.py`, `agent/conversation_loop.py` | Changed declared version rebuilds stored prompt and re-pins current tools once; transcript/session retained; compression inherits version |
| Sessions | `hermes_state.py`, API server session handlers | Full owner-box transcript, distinct from control-plane retained copies |
| Delegation | `tools/delegate_tool_config.py`, related delegate/escalate modules | Policy supplies tiers and caps; escalation directly launches a read-only advisor without a routing gate or parent reasoning update; no secret or budget bypass |
| Provider adapters | `agent/` provider transports | Preserve selected transport and cache prefix; OpenAI configuration updates are appended and validated |
| Native text stream | `gateway/platforms/api_server_runs.py` | `message.delta` events carry `stream_id` captured on the producing thread; consumers replace retried drafts; rc12 branch: Responses commentary beside turn-ending calls becomes the visible reply |
| Iollo task reports | `hermes_state_messages.py::append_iollo_task_report`, `gateway/platforms/api_server.py` | Source review 2026-10-09 on rc15 (`2d2068f3ce54b5974629220a81c451d17aabbd2b`): authenticated `POST /api/sessions/{session_id}/task-reports` stores a bounded, explicitly framed model-facing completion event in the exact owner-box session and projects its literal text as an assistant row in native history. Retries dedupe by task id + generation through compression lineage; active turn leases refuse the append for retry. `GET .../messages` pages native deduped display history across compression lineage, and a watermark-cloned report keeps its original public session/row identity for envelope bindings. No prior control-plane transcript is imported. |
| Run progress events | `gateway/platforms/api_server_run_progress.py` | rc12: `tool.generating {tool}` (name only, ≤ every 10 s) and `run.heartbeat` (≤ every 12 s while the agent is alive and silent); status `last_event` moves, never during an approval, never in transcripts |
| Recorded voice | `gateway/platforms/api_server_audio.py`, [voice API](voice-api.md) | Opt-in authenticated transcription/read-aloud; bounded temporary audio and busy accounting; no agent turn |

Read the relevant nested AGENTS.md before touching each runtime area. When changing an integration
seam, update this pointer and the cloud specialist guide/tests with a pinned source revision. Never
claim an upstream feature is deployed without image/config evidence. Preserve private history and
credentials; document only sanitized behavior and credential names.

Distribution and upstream-upgrade details live in [IOLLO-FORK](../IOLLO-FORK.md);
trace data handling lives in [IOLLO-RUN-TRACE](../IOLLO-RUN-TRACE.md). rc7–rc13 are
cloud runtime tags without a published GitHub runtime release as of this review. The "Iollo release" workflow is disabled, so
tags do not build images; the base image is built by hand (cloud `docs/DEPLOY.md`, "Building the Hermes base by hand").
A historical train manifest or native runtime bundle must not override the recorded cloud pin.


## Cooperative task withdrawal (source change on base `a58f99cc94c4`, 2026-10-18)

Authenticated `POST /api/jobs/{id}/stop` durably pauses future fires and records `stop_requested` on the job.
Native cron agents check that withdrawal at tool admission and between iterations. Already executing calls are
allowed to return; no process, payment or tool waiter is killed by this request. The cloud tasks concern keeps
`stopping` until native execution ends, retains late outcomes, and never retries a withdrawn unknown execution.
A returned native run confirms zero started actions explicitly before automatic continuation is eligible;
the box durably claims that handoff once and never automatically retries ambiguous acceptance.
The task-report append accepts `cancelled` and retains its existing task/generation idempotency. This source
requires a new Hermes image and matching sandbox concern; no rollout or deployed verification is claimed.

Native `GET /api/sessions/{id}/task-output` authenticates and reads the exact session, returning `{text,silent}`
using `gateway.response_filters` autonomous semantics. The task concern applies suppression only for successful
executions; failures keep cleaned prose. This API boundary lets cloud-only tests run without importing Hermes
or maintaining another silence matcher. Raw native execution history remains intact. Delegated children inherit
cooperative withdrawal; explicit native resume clears it for future work. Owner task cancellation supersedes a
previously accepted direction, and only a native `stopped_before_action` execution-ledger proof permits a
single claimed automatic continuation.
