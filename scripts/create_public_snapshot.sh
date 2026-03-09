#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/_public_release}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "ERROR: rsync is required." >&2
  exit 1
fi

echo "Preparing public snapshot at: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

INCLUDE_PATHS=(
  ".github"
  ".gitignore"
  ".env.example"
  "CHANGELOG.md"
  "CODE_OF_CONDUCT.md"
  "CONTRIBUTING.md"
  "LICENSE"
  "README.md"
  "README(EN).md"
  "SECURITY.md"
  "SUPPORT.md"
  "config.example.json"
  "models"
  "pyproject.toml"
  "scripts"
  "src"
  "tests"
)

PUBLIC_DOCS=(
  "docs/getting-started.md"
  "docs/current-limitations.md"
  "docs/architecture.md"
  "docs/roadmap.md"
  "docs/assets"
)

for rel in "${INCLUDE_PATHS[@]}"; do
  if [[ -e "$ROOT/$rel" ]]; then
    rsync -a "$ROOT/$rel" "$OUT_DIR/"
  fi
done

for rel in "${PUBLIC_DOCS[@]}"; do
  if [[ -e "$ROOT/$rel" ]]; then
    mkdir -p "$OUT_DIR/$(dirname "$rel")"
    rsync -a "$ROOT/$rel" "$OUT_DIR/$(dirname "$rel")/"
  fi
done

for rel in "$ROOT"/README\(*\).md; do
  if [[ -e "$rel" ]]; then
    rsync -a "$rel" "$OUT_DIR/"
  fi
done

# Remove local-only and generated artifacts from the snapshot.
rm -f "$OUT_DIR/.env" "$OUT_DIR/config.json" "$OUT_DIR/liagent.db"
rm -rf "$OUT_DIR/venv" "$OUT_DIR/.venv" "$OUT_DIR/.pytest_cache" "$OUT_DIR/.mypy_cache"
rm -rf "$OUT_DIR/src/.vscode" "$OUT_DIR/src/liagent.egg-info"
find "$OUT_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$OUT_DIR" -type f \( -name "*.pyc" -o -name ".DS_Store" \) -delete

# Keep helper scripts but remove nested snapshot output if it exists.
rm -rf "$OUT_DIR/_public_release"

if command -v rg >/dev/null 2>&1; then
  echo "Running quick snapshot checks..."
  (
    cd "$OUT_DIR"
    if rg -n \
      "(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[A-Za-z0-9-]{20,}|BEGIN PRIVATE KEY|PRIVATE KEY-----)" \
      . \
      --glob '!scripts/release_audit.sh' \
      --glob '!scripts/create_public_snapshot.sh'; then
      echo "ERROR: snapshot contains possible secret patterns." >&2
      exit 1
    fi
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
      "docs/architecture.md"
      "docs/roadmap.md"
      "docs/assets/web-ui.jpeg"
      "docs/assets/cli.png"
      "docs/assets/discord-app.jpeg"
    )
    for rel in "${required_files[@]}"; do
      if [[ ! -e "$rel" ]]; then
        echo "ERROR: snapshot is missing required file: $rel" >&2
        exit 1
      fi
    done
    localized_readme_count="$(find . -maxdepth 1 -type f -name 'README(*).md' | wc -l | tr -d ' ')"
    if [[ "${localized_readme_count:-0}" -lt 2 ]]; then
      echo "ERROR: snapshot is missing the localized README set." >&2
      exit 1
    fi
  )
fi

(
  cd "$OUT_DIR"
  rm -rf .git
  git init -b main >/dev/null
  git add .
  if git commit -m "chore: initial public release snapshot" >/dev/null 2>&1; then
    echo "Initial snapshot commit created."
  else
    echo "INFO: auto-commit skipped (git user.name/user.email may be missing)."
    echo "Run manually:"
    echo "  cd \"$OUT_DIR\""
    echo "  git commit -m \"chore: initial public release snapshot\""
  fi
)

echo "Snapshot ready."
echo "Next steps:"
echo "  cd \"$OUT_DIR\""
echo "  ./scripts/release_audit.sh"
echo "  git remote add origin <YOUR_PUBLIC_REPO_URL>"
echo "  git push -u origin main"
