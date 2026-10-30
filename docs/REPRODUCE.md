# Reproducing Tendency Tax

This document describes two reproducibility paths:
1. **App only** — clone and run the app against the committed results (no raw data needed).
2. **Full reproduction** — clone, download raw data, and re-derive all results from scratch.

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) 0.4+  (`pip install uv` or see uv docs)
- Python 3.11+ (uv manages the interpreter automatically)
- Internet access for `make download`

> **Note:** `pip install -r requirements.txt` also works if you prefer pip over uv.
> `uv.lock` pins every dependency to an exact version for bit-for-bit reproducibility.

---

## Path 1 — Clone → Running App (no raw data needed)

The repository ships committed results in `results/`. The app reads only those files.

```bash
git clone https://github.com/abdulsaboor190/tendency-tax.git
cd tendency-tax

make setup          # uv sync — installs all pinned dependencies

make test           # pytest + ruff
                    # ~12 tests skip (they need raw data) — this is expected
                    # All other tests pass

make app            # launches Streamlit at http://localhost:8501
```

All four pages (Test panel, Situation explorer, Team scout, About the method) render
from the committed `results/` files with no raw data present.

---

## Path 2 — Clone → Fully Reproduced Results

This path re-derives every result from the raw play-by-play data.

```bash
git clone https://github.com/abdulsaboor190/tendency-tax.git
cd tendency-tax

make setup          # install dependencies

make download       # fetch play_by_play_{2016..2025}.csv.gz from nflverse
                    # (~30 min on a slow connection; files are ~500 MB total)
                    # Raw data goes into data/raw/ (git-ignored, never committed)

make data           # build and validate the data cache in data/cache/
                    # Validates schema, column names, and split boundaries

make dev-run        # fit the primary logistic-regression model on 2016–2023,
                    # run the bootstrap (n=500) and placebo permutations
                    # Expected runtime: ~20–30 min depending on CPU
                    # Outputs: results/dev_primary.json, results/placebo_*.parquet,
                    #          results/models/phat_model_2016_2023.joblib

# Secondary pipeline (cells.parquet, teams.parquet) re-derives situation and team tables:
make secondary      # Expected runtime: ~5 min

make app            # launch the app — all pages render from re-derived results
```

### Determinism

Running `make dev-run` twice on a clean copy of `results/` produces bit-for-bit
identical outputs. Verified SHA-256:

```
dev_primary.json      1e773877a4357a192bab3299cecdffbcbf3e346055ea2aa844dc6165d1c46c13 (match)
placebo_early.parquet match
placebo_tune.parquet  match
```

All random seeds are imported from `src/config.py`. No external randomness sources.

---

## Holdout Note

The confirmatory holdout evaluation (seasons 2024–2025) is sealed per the
pre-registered protocol in `PLAN.md`. Do **not** set `TENDENCY_ALLOW_HOLDOUT`
outside the confirmatory run. Running `make confirm` requires both raw holdout data
and explicit acknowledgement of the pre-registration commitment.

---

## Data Attribution

Raw play-by-play data is downloaded from
[nflverse-data](https://github.com/nflverse/nflverse-data) and is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Raw files are never committed to this repository (see `.gitignore`).

---

## pip Alternative

```bash
pip install -r requirements.txt    # installs all pinned dependencies via pip
# then use `python -m streamlit run app/app.py` instead of `make app`
```

