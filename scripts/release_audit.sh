#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${LIAGENT_PYTHON:-}" ]]; then
  PYTHON="$LIAGENT_PYTHON"
elif [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON="$ROOT/venv/bin/python"
else
  PYTHON="python3"
fi

if [[ -n "${LIAGENT_PYTEST:-}" ]]; then
  PYTEST_CMD=("$LIAGENT_PYTEST")
elif "$PYTHON" -m pytest --version >/dev/null 2>&1; then
  PYTEST_CMD=("$PYTHON" "-m" "pytest")
elif command -v pytest >/dev/null 2>&1 && pytest --version >/dev/null 2>&1; then
  PYTEST_CMD=("pytest")
else
  PYTEST_CMD=()
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "ERROR: ripgrep (rg) is required for release audit." >&2
  exit 1
fi

fail=0

echo "[1/7] Syntax check"
PYTHONPYCACHEPREFIX="$ROOT/.pycache" "$PYTHON" -m compileall -q src

echo "[2/7] Fast unit tests"
if [[ "${#PYTEST_CMD[@]}" -eq 0 ]]; then
  echo "ERROR: pytest is not available for $PYTHON." >&2
  echo "Install it first: $PYTHON -m pip install pytest" >&2
  fail=1
else
  PYTHONPATH="$ROOT/src" "${PYTEST_CMD[@]}" -q \
    tests/test_events.py \
    tests/test_skill_router_simplified.py \
    tests/test_health.py \
    tests/test_brain_layers.py \
    tests/test_web_event_envelope.py \
    tests/test_tool_parsing.py
fi

echo "[3/7] Secret-pattern scan"
if rg -n \
  "(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,}|BEGIN PRIVATE KEY|PRIVATE KEY-----)" \
  . \
  --hidden \
  --glob '!.git/**' \
  --glob '!venv/**' \
  --glob '!.venv/**' \
  --glob '!_public_release/**' \
  --glob '!scripts/release_audit.sh' \
  --glob '!scripts/create_public_snapshot.sh'; then
  echo "ERROR: potential secret patterns found." >&2
  fail=1
fi

echo "[4/7] Public entry files"
required_files=(
  "README.md"
  "README(EN).md"
  "LICENSE"
  "CONTRIBUTING.md"
  "SECURITY.md"
  "SUPPORT.md"
  "CODE_OF_CONDUCT.md"
  ".env.example"
  "config.example.json"
  "docs/getting-started.md"
  "docs/current-limitations.md"
)
for rel in "${required_files[@]}"; do
  if [[ ! -e "$ROOT/$rel" ]]; then
    echo "ERROR: missing public entry file: $rel" >&2
    fail=1
  fi
done

localized_readme_count="$(find "$ROOT" -maxdepth 1 -type f -name 'README(*).md' | wc -l | tr -d ' ')"
if [[ "${localized_readme_count:-0}" -lt 2 ]]; then
  echo "ERROR: expected README(EN).md plus one additional localized README." >&2
  fail=1
fi

echo "[5/7] English-only scan"
english_only_scan_args=(
  .
  --hidden
  --glob '!.git/**'
  --glob '!venv/**'
  --glob '!.venv/**'
  --glob '!_public_release/**'
)
while IFS= read -r localized_readme; do
  rel="${localized_readme#$ROOT/}"
  english_only_scan_args+=(--glob "!$rel")
done < <(find "$ROOT" -maxdepth 1 -type f -name 'README(*).md' ! -name 'README(EN).md' | sort)

if rg --pcre2 -n "[\p{Han}]" "${english_only_scan_args[@]}"; then
  echo "ERROR: non-English Han content found outside the localized Chinese README." >&2
  fail=1
fi

echo "[6/7] Personal path scan (release candidates)"
if rg -n \
  "(/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|C:\\\\Users\\\\[A-Za-z0-9._-]+\\\\)" \
  . \
  --hidden \
  --glob '!.git/**' \
  --glob '!venv/**' \
  --glob '!.venv/**' \
  --glob '!_public_release/**' \
  --glob '!scripts/release_audit.sh' >/tmp/liagent_path_scan.out 2>/dev/null; then
  cat /tmp/liagent_path_scan.out
  echo "ERROR: personal absolute paths found." >&2
  fail=1
fi
rm -f /tmp/liagent_path_scan.out

echo "[7/7] Working tree status"
git status --short
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean. Commit/stage only release files before publishing." >&2
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "Release audit failed."
  exit 1
fi

echo "Release audit passed."
