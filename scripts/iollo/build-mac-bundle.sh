#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <iollo-tag> [arm64|x86_64] [output-directory]" >&2
  exit 2
fi
script_dir=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$script_dir" rev-parse --show-toplevel)
tag=$1
arch=${2:-$(uname -m)}
[[ $(uname -s) == Darwin && $(uname -m) == "$arch" ]] || {
  echo "Build on a native macOS $arch runner; cross-builds cannot pass the boot checks." >&2
  exit 1
}
python3 "$script_dir/release.py" check-tag "$tag"
command -v zstd >/dev/null
mkdir -p "${3:-$repo/dist/iollo}"
output=$(cd "${3:-$repo/dist/iollo}" && pwd)
name="hermes-runtime-$tag-macos-$arch"
[[ ! -e "$output/$name.tar.zst" ]] || { echo "Archive already exists" >&2; exit 1; }
work=$(mktemp -d "${TMPDIR:-/tmp}/iollo-build.XXXXXX")
trap 'rm -rf "$work"' EXIT
root="$work/$name"
mkdir -p "$root/hermes" "$root/bin"
url=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["url"])' "$script_dir/python-standalone.json" "$arch")
sha=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["sha256"])' "$script_dir/python-standalone.json" "$arch")
curl -fsSL --retry 3 "$url" -o "$work/python.tar.gz"
printf '%s  %s\n' "$sha" "$work/python.tar.gz" | shasum -a 256 -c -
tar -xzf "$work/python.tar.gz" -C "$root"
python="$root/python/bin/python3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
# Include only tracked checkout files, never local credentials, caches or build output.
# Retain the entire source layout: upstream resolves lazy imports and data via __file__.
git -C "$repo" ls-files -z | COPYFILE_DISABLE=1 tar -C "$repo" --null -T - -cf - | tar -C "$root/hermes" -xf -
# Export upstream's locked dependency closure, including hashes. The exporter is
# a build tool in a temporary venv, not an extra dependency of the runtime.
"$python" -m venv "$work/build-tools"
"$work/build-tools/bin/python3" -m pip install --disable-pip-version-check 'uv==0.11.6'
"$work/build-tools/bin/uv" export --project "$root/hermes" --frozen --no-dev \
  --extra messaging --extra mcp --extra web --no-emit-project \
  --format requirements-txt --output-file "$root/requirements.lock.txt" >/dev/null
"$python" -m pip install --disable-pip-version-check --no-compile --require-hashes \
  --report "$root/DEPENDENCIES-REPORT.json" -r "$root/requirements.lock.txt"
# Upstream's sealed-package build switch; only enabled for this pip invocation.
# No Hermes source edits and no Nix/managed-mode flag in the runtime environment.
HERMES_NIX_BUILD=1 "$python" -m pip install --disable-pip-version-check \
  --no-compile --no-deps --report "$root/INSTALL-REPORT.json" "$root/hermes[messaging,mcp,web]"
"$python" -m pip check
"$python" "$script_dir/bundle.py" prepare "$root"
"$python" -I -B - <<'PY'
import json
from pathlib import Path
import iollo_envelope
from iollo_envelope.producer import Producer
from gateway.platforms import api_server_runs

catalog = Path(iollo_envelope.__file__).with_name("catalog.schema.json")
assert json.loads(catalog.read_text())["$schema"]
assert api_server_runs._iollo_envelope_installed
producer = Producer("bundle_smoke_fake", surface="mac")
producer.event({"event": "tool.started", "tool": "terminal", "preview": "echo fake"})
partial = producer.revision()
final = producer.revision(text="Ready.", status="final")
assert partial["status"] == "partial" and final["status"] == "final"
assert partial["message_id"] == final["message_id"] and final["rev"] > partial["rev"]
PY
cp "$script_dir/hermes-launcher.sh" "$root/bin/hermes"
chmod +x "$root/bin/hermes"
printf '%s\n' "$tag" > "$root/VERSION"
cp "$script_dir/python-standalone.json" "$root/PYTHON-STANDALONE.json"
# Hide the original build path before exercising the installed entry points.
mkdir "$work/relocated directory"
mv "$root" "$work/relocated directory/"
root="$work/relocated directory/$name"
"$root/python/bin/python3" "$script_dir/smoke-bundle.py" "$root"
"$root/python/bin/python3" "$script_dir/bundle.py" manifest "$root"
COPYFILE_DISABLE=1 tar -C "$(dirname "$root")" -cf - "$name" | zstd -T0 -10 -o "$output/$name.tar.zst"
python3 "$script_dir/release.py" create "$tag" "$output"
"$script_dir/verify-bundle.sh" "$output/$name.tar.zst"
echo "Built $output/$name.tar.zst"
