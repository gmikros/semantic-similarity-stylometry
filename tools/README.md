# tools/

Helper scripts that sit outside the core reproducible pipeline in `analysis/`.

## Corpus generation (runnable)

| Script | Purpose | Requirements |
|--------|---------|--------------|
| `generate_blocks_llm.py` | Generate fresh author-voice and shared topic blocks with an LLM, following `docs/prompt_templates.md`. | `OPENAI_API_KEY`; `openai` package (in `analysis/requirements.txt`). |

Workflow:

```bash
python tools/generate_blocks_llm.py --intra 0.5 --inter 0.5 --length 300 \
       --n-topics 20 --out blocks/scenario_new
python analysis/src/generate_scenario.py --intra 0.5 --inter 0.5 --length 300 \
       --blocks-dir blocks/scenario_new --out "Corpus/Scenario new"
```

This is path-clean and portable; output is non-deterministic unless the model/params make
it so.

## Manuscript authoring helpers (machine-specific, NOT runnable as-is)

These contain **hardcoded absolute paths** to local `.docx`/`.pptx` files and are retained
for provenance only. They will not run on another machine.

| Script | Purpose |
|--------|---------|
| `extract_sources_for_rewrite.py` | Extract text from the source presentation/manuscript to summarize prior analyses. |
| `rewrite_methods_results_in_paper_docx.py` | Write regenerated Methods/Results prose into a manuscript `.docx`. |

To reproduce the study's results, use the pipeline in `analysis/` (see the top-level
`README.md`). The reproducible Methods/Results document generator
(`analysis/src/build_methods_results_docx.py`) uses only relative paths and stays in
`analysis/src/`.
