# tools/ — authoring helpers (not part of the reproducible pipeline)

These scripts were used by the authors to build and edit the manuscript. They contain
**hardcoded, machine-specific absolute paths** (e.g. to local `.docx`/`.pptx` files) and
therefore **will not run as-is on another machine**. They are retained here for provenance
and transparency only.

They are intentionally separated from `analysis/src/`, which contains the reproducible
analysis pipeline.

| Script | Purpose |
|--------|---------|
| `extract_sources_for_rewrite.py` | Extracts text from the source presentation/manuscript to summarize prior analyses. |
| `rewrite_methods_results_in_paper_docx.py` | Writes regenerated Methods/Results prose into a manuscript `.docx`. |

To reproduce the study's results, use the pipeline in `analysis/` instead (see the
top-level `README.md`). The reproducible Methods/Results document generator
(`analysis/src/build_methods_results_docx.py`) uses only relative paths and remains in
`analysis/src/`.
