#!/usr/bin/env bash
# End-to-end reproduction driver (macOS/Linux).
# Full regeneration needs OPENAI_API_KEY and R with stylo + networkD3 on PATH.
# Without them, pass --no-openai / --no-stylo to run_all.py and rely on shipped artifacts.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"

if [ ! -d ".venv" ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -r analysis/requirements.txt
python -m spacy download en_core_web_lg

# Full pipeline (skips OpenAI/stylo automatically if key/Rscript are missing).
python analysis/src/run_all.py "$@"

# Paper-ready tables and figures.
python analysis/src/build_jql_package.py
python analysis/src/build_jql_extended_results.py

echo "Done. See analysis/paper_package_jql and analysis/paper_package_jql_extended."
