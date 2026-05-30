"""Run and export significance tests used in the JQL results package.

This script reuses the core package builder and writes:
  paper_package_jql/tables/table_significance_tests.csv
"""
from __future__ import annotations

from build_jql_package import build_tables_and_figures


def main() -> None:
    _, _, sig = build_tables_and_figures()
    print(sig.to_string(index=False))


if __name__ == "__main__":
    main()
