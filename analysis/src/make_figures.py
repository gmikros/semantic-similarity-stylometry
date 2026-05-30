"""Generate all figures for the JQL paper package.

Figures are written to:
  paper_package_jql/figures/
"""
from __future__ import annotations

from build_jql_package import build_tables_and_figures


def main() -> None:
    build_tables_and_figures()
    print("Figures regenerated under paper_package_jql/figures/")


if __name__ == "__main__":
    main()
