# AGENTS.md — Repository Rules & Operating Guidelines

## Core Rules

1. **Holdout Protection**: Holdout seasons are 2024-2025. Never read their data or EPA until explicitly instructed by the user.
2. **EPA & xpass Roles**: `epa` is an outcome only, never a model feature. `xpass` is reference only.
3. **Call Label**: The call label is `qb_dropback` (1 = pass), not `play_type`.
4. **Primary Model Restrictions**: The primary model excludes `shotgun` and `no_huddle`.
5. **Spread Convention**: `spread_line > 0` means the home team is favored.
6. **Data Splits**:
   - `early`: 2016–2021 (baseline / exploratory)
   - `tune`: 2022–2023 (validation / tuning)
   - `holdout`: 2024–2025 (confirmatory holdout)
7. **Random Seeds**: Use fixed random seeds imported from `src/config.py`.
8. **Data Hygiene**: Never commit raw data (`data/raw/`). Small result artifacts in `results/` are fine.
9. **Causal Claims**: Use association-only language. Never use "causes", "proves", or "optimal".
10. **Plan Integrity**: Do not change `PLAN.md` after it is committed without logging every change in `DEVIATIONS.md`.
11. **Ambiguity Protocol**: If a definition is unclear, stop and ask the user. Do not invent thresholds.
12. **Commit Hygiene**: One deliverable per commit, using the message format `P0-D<n>: <name>`.

---

## Repository Layout

```text
tendency-tax/
├── .gitignore          # Git ignore rules (raw data, cache, env, venv)
├── AGENTS.md           # Repository operating guidelines and agent rules
├── Makefile            # Project command runner (setup, test, etc.)
├── PLAN.md             # Research plan and pre-registered protocol
├── pyproject.toml      # uv-managed dependencies and ruff configuration
├── uv.lock             # Deterministic dependency lockfile
├── app/                # Streamlit web application
├── data/
│   ├── raw/            # Raw nflverse play-by-play files (git-ignored)
│   └── cache/          # Processed intermediate data caches (git-ignored)
├── results/            # Model outputs, small summary tables, and metrics
├── scripts/            # Standalone execution and pipeline scripts
├── src/                # Reusable library modules and constants
│   ├── __init__.py
│   └── config.py       # Constants, splits, features, and thresholds
└── tests/              # Pytest test suite
    ├── __init__.py
    ├── test_config.py  # Tests for configuration and splits
    └── test_smoke.py   # Baseline environment smoke test
```

---

## Make Commands

- `make setup`: Set up local virtual environment and install all dependencies with `uv sync`.
- `make data`: Data pipeline execution target (stub).
- `make test`: Run full test suite (`uv run pytest`) and linter (`uv run ruff check .`).
- `make dev-run`: Run development analysis pipeline (stub).
- `make confirm`: Run confirmatory evaluation on holdout set (stub).
- `make app`: Launch the interactive Streamlit application (stub).

---

## Definition of Done

A deliverable or phase is considered **Done** when:
1. **Tests Pass**: `uv run pytest` runs cleanly with zero failures or errors.
2. **Ruff Clean**: `uv run ruff check .` passes without lint or formatting violations.
3. **Explicit Reporting**: Detailed summary reporting exactly what was done, what files were created or modified, test results, and explicitly stating what was NOT done.
