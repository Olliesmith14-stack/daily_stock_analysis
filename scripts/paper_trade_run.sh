#!/usr/bin/env bash
# Runs the paper-trading ledger inside GitHub Actions.
#
# The ledger is stored encrypted (AES-256, key = PAPER_LEDGER_KEY secret) as the
# only file on the `paper-ledger` branch, so it stays private even though the
# repository is public. Skips quietly when PAPER_LEDGER_KEY is not set.
set -euo pipefail

if [ -z "${PAPER_LEDGER_KEY:-}" ]; then
  echo "PAPER_LEDGER_KEY not set, paper trading skipped."
  exit 0
fi

BRANCH="${PAPER_LEDGER_BRANCH:-paper-ledger}"
export PAPER_LEDGER_PATH="${PAPER_LEDGER_PATH:-data/paper_ledger.json}"
ENC="$(mktemp)"
mkdir -p "$(dirname "$PAPER_LEDGER_PATH")"

if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git fetch -q --depth=1 origin "$BRANCH"
  git show FETCH_HEAD:ledger.json.enc > "$ENC"
  if ! openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -in "$ENC" -out "$PAPER_LEDGER_PATH" -pass env:PAPER_LEDGER_KEY 2>/dev/null; then
    echo "Could not decrypt the existing ledger. Has PAPER_LEDGER_KEY changed? Stopping so the ledger is not reset."
    exit 1
  fi
  echo "Existing ledger loaded."
else
  echo "No ledger yet, starting a new one."
fi

python scripts/paper_trade.py

openssl enc -aes-256-cbc -pbkdf2 -iter 200000 -salt -in "$PAPER_LEDGER_PATH" -out "$ENC" -pass env:PAPER_LEDGER_KEY
rm -f "$PAPER_LEDGER_PATH"

WORK="$(mktemp -d)"
cp "$ENC" "$WORK/ledger.json.enc"
cd "$WORK"
git init -q
git checkout -q -b "$BRANCH"
git add ledger.json.enc
git -c user.name="paper-ledger-bot" -c user.email="paper-ledger-bot@users.noreply.github.com" \
  commit -q -m "Update paper ledger"
git push -q -f "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" "$BRANCH"
echo "Ledger saved."
