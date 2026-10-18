# Iollo Hermes distribution

Read [the Iollo integration guide](iollo/README.md) first. Active development is web and the
Mac menu bar/local tools, supported by the personal cloud boxes. iOS and Desktop Pro are parked.
The Mac companion does not run the conversational agent.

Cloud uses the pinned fork base image in `registry.fly.io/instinct-sandboxes`, then builds
its own sandbox layer. At the 2026-10-04 source review the pin is **iollo-2026.9.24-rc13** (Hermes `e3682503cf`; PRs #22/#23/#24 are
rc11/rc12/rc13). At the 2026-10-03 review it was **iollo-2026.9.24-rc10**,
Hermes commit **a4976548790078dee821983d2bef6cc3c728fb62**. Cloud
`sandbox/hermes-base.txt`, the Dockerfile fallback and an explicit build override determine
that base; the final sandbox image and successful rollout determine what owners receive.

The packaging sections below describe the existing full release workflow and its historical
Mac Python runtime artifacts. A matching fork tag does not prove that workflow ran or that a
GitHub release exists: rc7–rc13 were cloud runtime updates without published full runtime
releases as of this review. The "Iollo release" workflow is disabled (2026-10-04), so tags do not build images; the base
image is built by hand on Fly's remote builder (cloud `docs/DEPLOY.md`, "Building the Hermes base by hand"). Do not refresh parked clients or their runtime pins as part of
routine cloud/menu bar work. Neither publishing a base nor creating a tag deploys boxes.
Iollo managed boxes use cloud policy and deployment controls rather than the upstream
`hermes update` command.

## Branches and the verified starting point

- `main` tracks `NousResearch/hermes-agent:main`, without Iollo changes.
- `iollo` is an upstream release tag plus our patches: distribution machinery isolated in
  `scripts/iollo/`, this document, one workflow and its tests; the additive
  runs-event trace fields documented in [IOLLO-RUN-TRACE.md](IOLLO-RUN-TRACE.md); and the
  vendored `iollo_envelope/` producer (see its `VENDORED` file) that the runs API serves as
  envelope revisions; and one approval fix in `tools/approval_context.py`: an `api_server` session
  with a registered approval listener (every `/v1/runs` run) counts as attended, so it asks instead of
  refusing as an unattended platform; one browser fix in `tools/browser_tool_session.py`
  (upstream candidate): a CDP-endpoint session whose agent-browser daemon lost its browser link
  ("CDP response channel closed", e.g. the browser behind `browser.cdp_url` restarted) drops that
  daemon generation and retries once against the same URL, as a CDP timeout already does; and fork
  brief 041, all off unless configured: `delegation.tiers` / `default_tier` / `tier_chooser_timeout_ms`
  (per-child model + reasoning effort for `delegate_task`, `tools/delegate_tool_tiers.py`), the
  advice-only `escalate` tool behind `delegation.escalate` (`tools/escalate_tool.py`), and two chooser
  hooks (`delegation_tier_chooser`, `busy_input_chooser`; `hermes_cli/plugin_choices.py`).
  Fork brief 043 adds an optional `prompt_version` on `POST /v1/runs` (see "Prompt version" below).
- The current base is upstream **v2026.9.24**, package version **0.21.5**, commit
  **f97608f178d1ffeca59860195ab7da295f7c8e5f** (the peeled annotated tag), rebased from
  v2026.9.14 (0.21.3) on 2026-09-25. That rebase tagged on targeted tests only (owner's call);
  four upstream-native test failures also fail on a clean v2026.9.24 checkout. The run stream
  keeps upstream's tool-output preview off (`_EMIT_TOOL_COMPLETED_PREVIEW` in
  `gateway/platforms/api_server_runs.py`).

```sh
git remote -v
git rev-parse 'v2026.9.24^{commit}'
git show v2026.9.24:pyproject.toml
```

## Prompt version (fork brief 043)

A session keeps the system prompt it stored on its first turn, so prompt-text changes (SOUL,
plugin prompt sections) never reached an existing session unless the control plane rotated it.
`POST /v1/runs` now accepts an optional string `prompt_version` (trimmed, at most 200 characters;
blank counts as absent; any other type is a 400 `invalid_prompt_version`). It is stored as
`prompt_version` in the session's `model_config` (no schema change). When a turn declares a version
that differs from the stored one, or the session has none stored, the stored prompt is rebuilt once
(`invalidate_system_prompt`, then the normal build and `_persist_system_prompt`) and the new version
is stored. Same version, or no `prompt_version` sent: the stored bytes are reused as before. The
transcript and session id never change, and `on_session_start` is not re-fired. On that
one refresh turn the runtime re-pins `tools[]` to the current build, so new configured tools
reach an existing session. Subsequent turns reuse the stored prompt and tool prefix.
Compression children inherit the version through `_session_init_model_config`. Code:
`agent/prompt_version.py`, `_restore_or_build_system_prompt` in `agent/conversation_loop.py`;
tests: `tests/agent/test_prompt_version.py` and the `prompt_version` cases in
`tests/gateway/test_api_server_runs.py`. A turn a live Desktop Bot Chat owner executes (mailbox
path) is not affected.

## Per-run tool profile and turn-ending tools (2026-10-04, rc11, PR #22, merged; base image built by hand)

`POST /v1/runs` accepts an optional `tool_profile` object: `name` (short identifier), `tools` (1-200
tool names), `skills` (boolean, default true) and `note` (at most 600 characters, appended to each
profile-lifting tool's description in that run's requests, e.g. the capabilities only the full set
has); anything else is a 400 `invalid_tool_profile`. The
agent is built exactly as without it: the stored system prompt, the session tool pin, compression
rebuilds and every persisted byte stay the full set. Only each provider request is projected
(`build_api_request` in `agent/turn_api_request.py`, before cache decoration): `tools[]` keeps the
named tools in the session's order plus tools registered `lifts_tool_profile=True`, and with
`skills: false` the `## Skills` index block is cut from the system message (the identity, guidance
and caller-context text before it stay byte-identical). Without a profile, `lifts_tool_profile` tools
are never sent. A call to such a tool, or to any tool of the agent the profile left out, lifts the
profile for the rest of the turn (the call runs as usual; the next request carries every tool and the
skills index). Without `tools` (and `skills` absent or true) the run is only named: nothing is
projected. The completed run reports `tool_profile: {name, lifted, api_calls}` only when one was
requested, so a caller can compare input tokens per call across profiles.
A provider switch between profiles changes `tools[]`, which most providers cache ahead of or with the
system text, so the first request after a switch re-reads the prompt uncached; each profile's prefix
is cached on its own.

A tool may be registered `ends_turn=True` (or `ends_turn=predicate(args, result)`), also through
`PluginContext.register_tool`. It only takes effect on a run that sends `ends_turn: true` on `/v1/runs`
(boolean, default false, else 400 `invalid_ends_turn`), and never after a stop was requested during
the round. On such a run, when every call of a tool round names such a tool, each has a result
`agent.display._detect_tool_failure` does not flag (and its predicate says yes), and the assistant
message carried visible text, the turn ends with that text (`turn_exit_reason` `tool_ended_turn`)
instead of one more model call. Otherwise the loop continues as before, so a refused call reaches the
model. Each call keeps its tool result; no duplicate closing assistant row is written
(`assistant(tool_calls) → tool → user` is legal on every provider path), and the gateway's
auto-continue does not treat such a tail as interrupted (`transcript_tail_ended_by_tool`). Code:
`agent/tool_profile.py`, `agent/turn_tool_round.py`, `agent/turn_finalizer.py`,
`gateway/platforms/api_server_runs.py`; tests: `tests/agent/test_iollo_tool_profile_and_ended_turn.py`
and the `tool_profile` cases in `tests/gateway/test_api_server_runs.py`.

### Connector read tools and machine-run review (2026-10-04, rc12, PR #23, merged; `hosted_tools` in rc13, PR #24)

Base `iollo-2026.9.24-rc11` (`5403c1749b`); rc12 is `25690aa809`, rc13 `e3682503cf` (adds `hosted_tools`: provider-executed
tools such as the Responses `web_search`, appended to a run's requests while its profile holds). Code evidence; the
cloud docs record which image the boxes run.

- `tool_profile.tools` entries may be globs (`*` only, literal prefix of at least 4 characters, e.g.
  `mcp__notion__*`), resolved against the session's real tools on every request; a glob that matches
  nothing is ignored. New `read_tools` (at most 20 names/globs; only with `tools`): adds the matching
  tools that are MCP tools whose discovery-time `readOnlyHint` is exactly true (the
  `_tool_read_only_hints` record the call-time trust gate reads) and whose own raw MCP name carries no
  write verb (`_WRITE_WORDS`: create, update, delete, send, …, also camelCase). No annotation, a
  utility tool or a built-in never counts. Budget per request: at most `MAX_READ_TOOLS` (12) read tools
  and `MAX_READ_CHARS` (16,000 schema characters, about 4k tokens); over either, none is sent (all or
  nothing, so the request stays stable). Calling a sent read tool keeps the profile; calling any other
  tool lifts it as before. The report adds `read_tools: {status: ok|none|over_budget, tools, chars,
  patterns: {pattern: count}}` only when `read_tools` was asked. Calls run through the same handlers,
  so approvals and MCP trust gates are unchanged. Stored prompt and tool pin are untouched.
- `skip_background_review` (boolean, default false, else 400 `invalid_skip_background_review`) on
  `POST /v1/runs` sets `agent.skip_background_review` for that run only (the flag cron already uses):
  no post-turn memory/skill review fork. The codex app-server runtime now honours the flag too (it
  ignored it before, cron included). Other end-of-turn work is unchanged: external memory sync
  (`iollo_notes` has no turn sync), context-engine notification, micro-compaction, persistence.
  Session auto-titling (turn start, `title_generation` side model) still runs for a new api_server
  session. The curator is a time-based gateway housekeeping job, not per run.
- Turn-ending on the OpenAI Responses path (gpt-6-luna): the reply the model writes beside
  `show_widget`/`ask_owner` arrives as a `phase=commentary` message, which the adapter files under
  reasoning, so the tool-call message had no content and no turn ever ended (production box: 0 of 31
  `show_widget` and 0 of 6 `ask_owner` calls had content beside them). Now, on an `ends_turn` run whose
  message has no content and whose calls all name turn-ending tools, `stage_tool_call_message`
  (`agent/turn_tool_round.py` `_promote_commentary_reply`) makes that commentary the row's visible
  content before the row is persisted and removes its flattened copy from `reasoning`; the exact
  `codex_message_items` stay, so the Responses replay is unchanged (content is not replayed beside
  them). If every call then succeeds, the turn ends with that text as `final_response` and it is
  streamed once as `message.delta`; otherwise the loop continues as for a Claude message with text
  beside tool calls. Claude paths carry no commentary items and are unchanged.
- Progress events on `/v1/runs` (`gateway/platforms/api_server_run_progress.py`): `tool.generating
  {tool}` when the model starts producing a tool call (the agent's `tool_gen_callback`, now wired for
  runs; the name only, never arguments), repeated at most every 10 s while that call is still being
  generated; `run.heartbeat` (no fields) at most every 12 s while the run is `running`, nothing else was
  emitted, no tool is executing and the agent's activity clock moved (a streaming or waiting model call,
  reasoning, compaction). A hung agent stays silent. Both update status `updated_at`/`last_event` like
  any event (not persisted), go to the SSE queue only while a stream is open, never reach the
  transcript or reply text, and are never sent while the run waits for an approval (a caller reads a
  later `last_event` as the box having moved on). Cost: one sleeping asyncio task per active run.
  Consumers that filter by event type (the iollo control plane's `_watch_tools`,
  `_stream_registered_partials`) ignore both.
- rc11 and older read the body with `.get`, so they ignore `read_tools` inside a profile they accept
  (a glob in `tools` fails their name check: 400) and ignore `skip_background_review`.
- Tests: `tests/agent/test_iollo_tool_profile_and_ended_turn.py` (connector section),
  `tests/gateway/test_api_server_runs.py` (`skip_background_review`, globs/read_tools),
  `tests/agent/test_codex_app_server_integration.py::TestSkipBackgroundReview`,
  `tests/gateway/test_iollo_run_progress.py`, and the `responses_commentary` cases in
  `tests/agent/test_iollo_tool_profile_and_ended_turn.py`. Known unrelated
  failure on the base: `TestCodexToolProgressBridge::test_session_wired_with_on_event_that_fires_tool_progress`.

## Local memory and native approvals (source change, 2026-10-18)

The retired optional permissions plugin and its judge are removed. Native manual approvals,
including API-server approval listeners, rule-specific wait windows and fail-closed unanswered
requests, remain available. Cloud payment rules and deny globs remain cloud policy.
Historical payment-gate implementation is retained in Git history.

The notes provider performs retrieval and fact filing through its local index, without a
retrieve/rerank/route decision service or decision configuration. It keeps typed notes,
provenance, profile injection, local FTS/vector search and connected Mac recall. The cloud
image adapter still removes private profile sections from default prompt injection.
This moves the existing cloud image's effective memory behavior into fork source; it does
not establish a new deployment. Tests cover local storage/search across A→B→A profile homes
without network access, connected Mac recall and native approval behavior.

Owner-requested note corrections/removals use the existing `memory_note` tool. Read the exact note
first; `correct` and `remove` require its id, full `expected_content` and exact prior fact;
`correct` also requires `replacement`. `remove_note` requires id and full expected content.
A changed note or ambiguous fact refuses the edit. Unrelated facts and handwritten profiles remain.
Generated profile projections are invalidated for normal maintenance to rebuild; existing
conversation prompts may still contain earlier context. When the owner's scope includes learned
procedures, inspect the relevant skill and use native `skill_manage` to patch/remove the scoped
content. Do not delete conversations or originals as an implied part of a note correction.

The receipt identifies the changed scope and retained history, skills and sources. This is not
complete cross-store forgetting: cloud maintenance watermarks skip already-processed conversations,
but old unprocessed retained evidence can be learned again. Moving a global watermark on one
removal would discard unrelated learning, so this source change does not do that. Permanent
exclusion across retained historical input needs an explicit policy and separate implementation.

## Taking an upstream release

Use an isolated clean worktree and record the old base before a deliberate upstream upgrade.
Do not switch or reset a shared primary checkout. This procedure describes upgrade work; a
stability pass does not authorize changing the upstream base or building parked client releases.
Never move or
reuse a published Iollo tag. Branch protection should permit a coordinated rebase
of `iollo`, while protecting release tags from modification/deletion.

```sh
git fetch upstream main --tags
git switch main
git merge --ff-only upstream/main
git push origin main

git switch iollo
old_base=v2026.9.24
new_base=v2026.10.1                 # example; choose an actual reviewed upstream tag
git rebase --onto "$new_base" "$old_base" iollo
# Resolve conflicts in distribution files, preserving upstream runtime behavior.
uv sync --frozen --extra dev --extra messaging --extra mcp --extra web
scripts/run_tests.sh               # full upstream suite, never bare pytest
brew install zstd actionlint shellcheck
actionlint .github/workflows/iollo-release.yml
shellcheck scripts/iollo/*.sh
scripts/iollo/build-mac-bundle.sh "iollo-${new_base#v}-rc1" arm64 /tmp/iollo-candidate
git push --force-with-lease origin iollo
# Review the arm64 PR dry-run before tagging a candidate/release.
git tag -a "iollo-${new_base#v}" -m "Iollo runtime based on $new_base"
git push origin "iollo-${new_base#v}"
```

For a rebuild/packaging revision use `iollo-2026.9.14-1`, then `-2`, etc. Candidate
tags end in `-rcN`, for example `iollo-2026.9.14-1-rc1`, and become prereleases.
Tags without that suffix become normal releases. An upstream `v` prefix is also
accepted (`iollo-v2026.9.14`), but prefer the forms above consistently.
The workflow resolves the upstream tag to its full commit and verifies that it is
an ancestor of the tagged checkout; it does not fetch/build upstream HEAD.

Review upstream packaging, extras, Python support and launch paths on every rebase.
Update `python-standalone.json` deliberately when upgrading Python: use exact
python-build-standalone asset URLs, verify their SHA256s against the publisher's
release digests, and update both architectures together. Currently both are
CPython **3.12.14**, standalone release **20260901**, unstripped `install_only`
archives. The local and CI builder are the same script.

## Release layout

The workflow `.github/workflows/iollo-release.yml` runs on `iollo-*` tag pushes.
Publishing is restricted to this fork. It produces:

```text
registry.fly.io/instinct-sandboxes:hermes-base-<tag>   # the box base image
ghcr.io/webaround-ai/hermes-agent:<tag>
ghcr.io/webaround-ai/hermes-agent:<tag>-<12-character-checkout-sha>

GitHub release assets:
  hermes-runtime-<tag>-macos-arm64.tar.zst
  hermes-runtime-<tag>-macos-x86_64.tar.zst   # best effort
  release.json
  SHA256SUMS
```

The shared reply producer is vendored unchanged under `iollo_envelope/`, with its
source commit in `iollo_envelope/VENDORED`. Package discovery and catalog package
data include it in both the Mac wheel and box image. The bounded install hook at
the end of `gateway/platforms/api_server_runs.py` adds envelope SSE and inputs
routes and enriches run status with the same locally stored envelope. It retains
the existing runs authorization and callback seams. Mac callers may provide
`envelope_context` with `surface: mac` and a public integer `conversation_id`.
The producer uses `HERMES_HOME/state.db`; it does not import the control plane.
The build and relocated smoke verify imports, catalog data, revision identity,
and envelope route authentication. The scripted integration test is
`tests/gateway/test_iollo_envelope_runtime.py`.

The box uses the **unchanged root Dockerfile**, built from the tagged checkout,
and pushes with `GITHUB_TOKEN` and job-scoped `packages: write`. It is **linux/amd64
only**: Fly boxes do not need a multi-architecture image. `release.json.image` is
the SHA-suffixed image tag; deployments can additionally record the GHCR digest.
Neither an upstream `latest` image nor upstream release assets are used.

Mac builds are native: `macos-15` for arm64 and **`macos-15-large` for x86_64**.
The Intel larger runner requires owner access/billing configuration. Set repository
variable `IOLLO_MACOS_X86_RUNNER=macos-15-intel` to use GitHub's standard native Intel
runner when the larger runner is unavailable. There is no
claimed cross-build fallback: python-build-standalone publishes separate Darwin
architecture archives, and a universal interpreter alone would not make all
third-party native extension wheels universal. Native builds let CI actually boot
both artifacts. Select/enable the Intel runner before tagging; do not silently
ship an untested cross-build.
See GitHub's [standard runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [larger runner labels](https://docs.github.com/en/actions/reference/runners/larger-runners).

Each archive extracts into one directory named like the archive without `.tar.zst`:

```text
bin/hermes                # relative launcher; preserves the caller's HERMES_HOME
python/                   # complete standalone Python + installed packages
hermes/                   # complete tracked checkout, no .git or untracked secrets
VERSION                   # exact Iollo release tag
MANIFEST.json             # regular-file hashes/sizes/modes and symlink hashes/targets
PYTHON-STANDALONE.json     # reviewed interpreter URLs and SHA256s
requirements.lock.txt     # selected upstream uv.lock closure, with hashes
DEPENDENCIES-REPORT.json   # pip's dependency installation receipt
INSTALL-REPORT.json       # pip's installation receipt for this checkout
```

The builder exports `uv.lock` using a temporary **uv 0.11.6** environment, installs
the dependency closure with pip's hash checking, then runs `pip install --no-deps`
on this checkout with **messaging, mcp, web, iollo-memory** extras. `aiohttp` in messaging powers
the `api_server` gateway adapter; MCP and the web/serve stack are also included.
There is no standalone `api_server --help`: it is an adapter enabled through gateway
configuration. CI checks the gateway CLI help, the `gateway.run` module help and
the actual HTTP server in a foreground gateway.

The `iollo_notes` memory provider (brief 003) needs the **iollo-memory** extra
(onnxruntime, sqlite-vec, tokenizers; onnxruntime has no macOS x86_64 wheel, so Intel
runtimes search with FTS only) and its embedding model, all-MiniLM-L6-v2 int8 ONNX
(about 23 MB). The model is never committed and never downloaded at runtime:
`scripts/iollo/fetch-embedding-model.py` fetches `model.onnx` and `tokenizer.json` at
release-build time and fails the build unless both match the size and SHA-256 pinned in
`plugins/memory/iollo_notes/model.json`. It lands in
`hermes-runtime-<tag>-macos-<arch>/hermes/plugins/memory/iollo_notes/model/` in the Mac
tarball (covered by `MANIFEST.json` and the archive hash; the smoke boot loads it) and at
`/opt/hermes/plugins/memory/iollo_notes/model/` in the box image (the box job fetches it
into the build context; the unchanged Dockerfile's `COPY . .` and `--extra all`, which now
includes iollo-memory, do the rest). The provider loads it from there
(`IOLLO_EMBED_MODEL_DIR` overrides); a missing or mismatched file means FTS-only search.

Upstream's `setup.py` blocks normal wheel builds because wheels omit source-relative
assets. The builder uses the existing **HERMES_NIX_BUILD=1** switch only for the pip
build and retains the tracked source tree beside the installed wheel. A relative
`.pth` entry makes that complete tree importable from the bundled interpreter,
including child Python processes. No flag is set at runtime to pretend this is Nix.
No package contents, lazy modules, skills, plugins, locales or stdlib modules are
stripped. Python console-script shebangs are made relative as part of packaging.
The only source-tree cleanup removes pip-generated build/egg-info directories.

`bin/hermes` uses only `python/bin/python3`, prevents host Python path/user-site
injection, and exports the supplied `HERMES_HOME`; if absent, it asks Hermes's own
`get_hermes_home()` for the default. It preserves the working directory and arguments.
Hermes's own `--version` remains the upstream version (0.21.5); `VERSION` is the Iollo product release tag.
The distribution is called `hermes-agent`; this checkout has **no import package
named `hermes_agent`**. Smoke tests import its real top-level modules instead.

This is the Python gateway/API runtime, not a Mac app/DMG, bundled JS dashboard,
Node runtime, browser download, or every optional provider/tool's external binary.
Those optional integrations retain upstream's setup/lazy-install behavior. The app
and box orchestrator own the product update policy and any additional tool provisioning.

`MANIFEST.json` lists every regular file and symlink except itself (a self-hash is
impossible). Symlinks hash the literal link-target bytes and must stay inside the
runtime. The archive's outer SHA-256 covers the manifest too.

`release.json` is the release train manifest, schema 1:

```json
{"schema": 1, "tag": "<tag>", "commit": "<40-hex tagged checkout>",
 "upstream_version": "v2026.9.14", "upstream_commit": "<40-hex>",
 "image": "ghcr.io/webaround-ai/hermes-agent:<tag>-<12-hex>",
 "box_base_image": "registry.fly.io/instinct-sandboxes:hermes-base-<tag>",
 "box_base_digest": "sha256:<64 hex> or null",
 "assets": [{"name": "hermes-runtime-<tag>-macos-arm64.tar.zst", "arch": "arm64",
             "sha256": "<64 hex>", "size": 0,
             "url": "https://github.com/Webaround-ai/hermes-agent/releases/download/<tag>/<name>"}]}
```

`box_base_digest` is the `box` job's pushed image digest; local builds write `null`.
Releases up to `iollo-2026.9.14-rc4` have the older shape (no `schema`, `commit`,
`box_base_*` or `url`; a `signature_name` per asset). `check-archive` accepts both.
`SHA256SUMS` covers the archives and `release.json`. After publishing, the `release`
job downloads its own `release.json` and runs `release.py check-train` on it, which
prints one line per problem and fails the job.

## Signing and trust

We do not sign anything. There is no minisign key, signature or public key. The Mac
app embeds the arm64 tarball at build time, checks it against the SHA-256 pinned in
its own repository, and Apple's signature and notarization of the app cover the
result. Boxes take `box_base_image` for the same tag. `SHA256SUMS` and the per-asset
SHA-256 are integrity checks only. After this change the old repository secrets
`IOLLO_MINISIGN_SECRET_KEY` and `IOLLO_MINISIGN_PASSWORD` are unused; the owner may
delete them.

## Local and CI verification

Prerequisites: native macOS for the chosen architecture, git, curl, Python 3.9+
for the packaging helpers, and zstd. No developer
Python environment is used to run Hermes. Keep the output directory outside the
checkout (or under the already-ignored `dist/`). A successful build automatically
relocates to a path containing spaces, runs imports and help checks, boots the
gateway with a temporary `HOME` and `HERMES_HOME`, uses a random fake API key to
GET `/v1/models`, waits for running state, and stops it with the upstream SIGINT
foreground shutdown contract. It then creates and verifies the archive.

```sh
brew install zstd
scripts/iollo/build-mac-bundle.sh iollo-2026.9.14 arm64 /tmp/iollo-runtime
scripts/iollo/verify-bundle.sh /tmp/iollo-runtime/hermes-runtime-iollo-2026.9.14-macos-arm64.tar.zst

# For a downloaded release: put release.json and the archive together.
scripts/iollo/verify-bundle.sh /path/to/archive.tar.zst
python3 scripts/iollo/release.py check-train /path/to/release.json

# To repeat the full smoke against an already extracted runtime:
/path/to/runtime/python/bin/python3 scripts/iollo/smoke-bundle.py /path/to/runtime
scripts/run_tests.sh tests/scripts/iollo/test_release.py
```

The verifier first checks size/SHA256 against adjacent `release.json`, then
rejects path escapes/special files, checks the complete manifest and embedded tag,
and boots `bin/hermes --version` with empty environment, temporary user state and a
different working directory. Version ordering and rollback belong to the Mac app,
which pins one tag.

PRs targeting `iollo` get an arm64 dry-run with the same builder and smoke checks;
the archive, checksum list and metadata are workflow artifacts for seven
days. Tag jobs verify each architecture natively, publish the box image, and only
then stage a GitHub draft containing all assets before publishing the release.
If final publication fails, inspect/delete the incomplete draft before rerunning
that job; never overwrite a release consumers have already downloaded.

## Historical runtime-bundle integration

Enable Actions and select the Intel runner as above, allow the jobs' scoped token permissions,
and ensure this repository can write its GHCR package. If the package already exists,
grant it Actions access to this repository. Choose public package visibility if Fly
pulls anonymously; otherwise provision Fly's registry pull credentials. GitHub's
repository/package visibility settings are separate.

Historical integration decisions, answered 2026-09-24 (superseded for the active menu bar
companion by the cloud-agent architecture above): the Mac app pins one fork tag and
embeds that tag's arm64 tarball, checked against the SHA-256 in its own repository
(iollo-mac brief 028); the cloud promote command rolls boxes onto
`hermes-base-<tag>` for the same tag (iollo brief 030). Still open: an Intel runner
and GHCR visibility. These consumer changes live in those repositories.

## Box image on the Fly registry

The `box` job also pushes the same image to `registry.fly.io/instinct-sandboxes:hermes-base-<tag>`,
using the repository secret `FLY_API_TOKEN` (a Fly deploy token scoped to the `instinct-sandboxes`
app). Cloud's sandbox image adds its integration layer on the selected base; existing cloud-only
base builds may publish directly to this registry without a full GitHub release. Pin the exact
fork revision and registry digest in the build evidence. GHCR publication is an additional
artifact of the full release workflow, not a required marker of cloud fleet convergence.

### rc15: explicit final reply publication (2026-10-08)

Built on rc14, this release lets a trusted successful turn-ending callback return final reply text.
The runtime appends and persists a closing assistant message after tool results before streaming it,
without another model call. Failed, mixed, interrupted and ambiguous publication rounds keep running.
The control plane opts in per run; full stored prompts and tool pins remain intact. Cloud coordinates
plugin, guide and prompt rollout. Focused runtime/relay checks passed remotely (43); no local suite ran.
## Managed shared profiles (2026-10-06, dedicated shared runtime deployed)

`agent.managed_allowed_tools` is an optional operator tool ceiling, enforced in
schema selection and before inline/registry dispatch. Absence preserves personal
agents; an empty/malformed value denies all. On dedicated shared hosts, the root
profile sets `agent.managed_tool_host: true`, so missing per-profile policy also
denies tools. These files must be operator-owned. This is application policy,
not an OS sandbox or a protection against arbitrary host code execution.

`GET /p/<profile>/v1/runs` returns only that authenticated scope's live run IDs,
including runs whose creation acknowledgement was lost. The shared runtime uses
it to quiesce a profile before removing its storage. No input/output is returned.
Cloud's `docs/SHARED-AGENTS.md` owns deployment and group-membership semantics.

## Native execution handoff — source change, 2026-10-18

Based on `bd5fd42fccba`, trusted plugins may register
`execution_handoff=predicate(actual_args, result)`. The callback returns a nonempty acknowledgement
of at most 2,000 characters when a successful tool transferred execution, or `None`/`False` to
continue. The integration supplies acknowledgement text in the owner's language. A task integration
uses this only for accepted immediate work, including an accepted existing task on retry; an actual
refusal or a future schedule continues normally. This metadata is not a model argument or a magic
field an untrusted tool result can assert.

Native dispatch resolves broker calls to the canonical tool, persists the successful tool result,
then evaluates the registered callback before admitting another sibling action. Unstarted siblings
receive persisted no-effect results with their original call IDs. Potential handoffs are sequential
barriers, including callers that directly request concurrent dispatch; earlier completed calls remain
recorded. Segmented batches do not launch later segments after transfer. The native turn publishes
the supplied acknowledgement and completes normally through `tool_ended_turn`, even without model
prose or the reply-tool `ends_turn` opt-in. A later owner turn starts with no inherited handoff state.
Ordinary reply-tool turn-ending behavior, stored prompts, model policy and tool pins are unchanged.

A failed canonical result append stops the caller through the existing persistence-failure path.
A broken registered callback stops further actions with an outcome-check diagnostic rather than
retrying the completed action. Timeout/unknown dispositions retain the existing native semantics;
this mechanism does not make task creation itself idempotent or resolve an ambiguous external action.
The cloud task adapter remains responsible for determining whether the same task accepted the work.
No shared browser lease, resource isolation or additional planner is introduced by this change.

Focused native regressions cover real dispatch and SQLite transcript persistence for direct/broker
calls, mixed and explicitly concurrent/segmented batches, accepted existing tasks, refused/future
starts, persistence failure and subsequent owner turns. Deployment evidence belongs in cloud STATE;
source presence does not identify an installed runtime.


## Shared headed browser idle lifetime — source change, 2026-10-18

The browser inactivity janitor defers plain local headed Chromium to the running Bot Desktop's
lifetime. This keeps the shared screen browser alive through an approval wait even while input is
agent-owned. The check uses existing session backend provenance, headed policy and the currently
published desktop; it also covers bootstrap that registers the local session before starting the
screen. It does not refresh activity. Once the desktop is down, an already idle session can be
cleaned normally. Explicit shutdown, backend recovery, session tracking and orphan cleanup remain
active. Cloud, independent CDP, real-profile, Lightpanda, Camofox and headless sessions retain their
existing idle cleanup; the HUMAN takeover exemption remains unchanged. No heartbeat, timer increase,
lease or additional lifecycle is introduced. This source change is not deployment evidence.

## Direct advice escalation — source change 2026-10-18

`escalate` launches its configured read-only advisor directly, subject to the existing caps.
The escalation routing gate and append-only parent reasoning upgrade attempt have been removed.
The parent model, effort and conversation configuration stay unchanged; advisory children retain
the existing restricted research toolsets and cannot delegate or act. A pinned advisor does not
inherit the parent fallback chain, and now explicitly disables delegation fallbacks as well.
Source change only; a rebuilt Hermes base and matching cloud policy are required before rollout.

## Scheduled continuation (source, 2026-10-18)

The existing jobs run endpoint accepts a prior execution id, request id and revised direction.
Admission remains on the same job, records its request identity durably and requires the latest
terminal execution with an exact session binding. A native future-completion marker or a dead
execution owner establishes that no detached worker is still acting. Duplicate admission never
rearms an older request.

The scheduler uses the existing CLI resume loader for compression-tip resolution, native history
and resume-size checks, then calls its ordinary conversation runner. The executions ledger binds
the resolved session before tools. Task output can be selected by execution id, excluding prior
attempt replies. Iollo publication uses distinct execution identities while preserving the same
session. No alternate scheduler, model or context serializer is introduced.
