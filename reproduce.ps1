# End-to-end reproduction driver (Windows PowerShell).
# Full regeneration needs OPENAI_API_KEY and R with stylo + networkD3 (Rscript on PATH).
# Without them, pass --no-openai / --no-stylo and rely on the shipped artifacts.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".\.venv\Scripts\python.exe" -m pip install -r "analysis\requirements.txt"
& ".\.venv\Scripts\python.exe" -m spacy download en_core_web_lg

# Full pipeline (skips OpenAI/stylo automatically if key/Rscript are missing).
& ".\.venv\Scripts\python.exe" "analysis\src\run_all.py" @args

# Paper-ready tables and figures.
& ".\.venv\Scripts\python.exe" "analysis\src\build_jql_package.py"
& ".\.venv\Scripts\python.exe" "analysis\src\build_jql_extended_results.py"

Write-Output "Done. See analysis\paper_package_jql and analysis\paper_package_jql_extended."
