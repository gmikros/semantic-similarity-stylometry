# JQL Results Package

This folder contains a paper-ready results package targeted to the
**Journal of Quantitative Linguistics**.

## Contents

- `tables/table_scenario_overview.csv`
- `tables/table_method_performance.csv`
- `tables/table_significance_tests.csv`
- `figures/fig1_accuracy_heatmap.png`
- `figures/fig2_ari_heatmap.png`
- `figures/fig3_scenario3_within_across.png`
- `figures/fig4_scenario5_within_across.png`
- `draft/jql_section_by_section_draft.md`

## Rebuild

Use the project virtual environment:

```powershell
C:\Users\USER01\.venvs\qualico_mikros_cech\Scripts\python.exe src\build_jql_package.py
```

Or run focused scripts:

```powershell
C:\Users\USER01\.venvs\qualico_mikros_cech\Scripts\python.exe src\make_figures.py
C:\Users\USER01\.venvs\qualico_mikros_cech\Scripts\python.exe src\run_significance_tests.py
```

## Reporting Notes (JQL style)

- Emphasize effect sizes and interval estimates before binary significance claims.
- Report exact p-values where feasible.
- Keep scenario design and corpus controls explicit, especially topic confounding
  (Scenario 5) and the restored canonical Scenario 3 corpus.
