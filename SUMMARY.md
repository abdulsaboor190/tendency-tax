# Tendency Tax — Project Summary & Deployment Guide

## 1. Executive Summary

**Tendency Tax** is a rigorous, pre-registered sports analytics research project and interactive web application investigating whether NFL offensive play calls (pass vs. run) systematically underperform in expected points added (EPA) when they are predictable from pre-snap game context.

### Research Question
Does an offense experience an expected points added penalty or benefit when its play call is predictable from situational context (down, distance, yardline, clock, score)?
- **Hypothesis**: Offenses face a "tendency tax" (penalty) when calling passes in pass-predictable situations or runs in run-predictable situations.
- **Pre-registered Specification**: $\Delta = \text{EPA}(\text{surprise}) - \text{EPA}(\text{predictable})$ evaluated on held-out regular season games.
- **Language & Philosophy**: Strict adherence to association-only language. No causal claims, no words such as "causes", "proves", or "optimal".

---

## 2. Core Architecture & Methodology

### Data Protocol & Splits
- **Source**: `nflverse` regular season play-by-play (2016–2025).
- **Early Split (2016–2021)**: Baseline exploratory data.
- **Tune Split (2022–2023)**: Model tuning and specification selection.
- **Holdout Split (2024–2025)**: Strictly protected confirmatory evaluation.

### Play-Call Modeling
- **Target Label**: `qb_dropback` (1 = intended pass call, 0 = designed run).
- **Features**: 10 pre-snap situational features only (`down`, `ydstogo`, `yardline_100`, `qtr`, `game_seconds_remaining`, `half_seconds_remaining`, `score_differential`, `spread_line`, `posteam_timeouts_remaining`, `defteam_timeouts_remaining`).
- **Restrictions**: Excludes `shotgun`, `no_huddle`, and post-snap features to prevent leakage. `epa` is outcome only.

### Decision Rules for Primary Hypothesis
All three pre-declared criteria must hold simultaneously for the hypothesis to be supported:
1. **$C_1$**: The 95% bootstrap confidence interval on the held-out sample excludes zero.
2. **$C_2$**: The sign in the holdout matches the sign in the early baseline split.
3. **$C_3$**: The held-out estimate lies outside the empirical 95% placebo permutation interval.

---

## 3. Interactive Web Application

The interactive application is built with Streamlit (`app/`) and consumes pre-computed results directly from `results/` without requiring raw play-by-play files:

1. **Test Panel (`app/views/test_panel.py`)**:
   - Confirmatory verdict banner with primary estimate, 95% bootstrap CI, and sample statistics.
   - Decile gap chart (pass minus run EPA gap across deciles of predicted pass probability).
   - Placebo permutation null distribution histogram.
   - Robustness checks and inline methodology documentation.
2. **Situation Explorer (`app/views/situation_explorer.py`)**:
   - 27-cell grid across down, distance, and field position.
   - Metric cards displaying sample breakdown, adjusted $p$-value ($q$), predictability score, and middle-half IQR.
   - Interactive what-if scenario calculator.
3. **Team Scout (`app/views/team_scout.py`)**:
   - Empirical Bayes shrinkage estimates comparing raw vs. shrunk tendency differences.
   - League-wide distribution strip with between-team variance ($\tau^2$) and low-sample safeguards.
4. **About the Method (`app/views/about_method.py`)**:
   - Explanation of pre-registration integrity, statistical controls, bootstrap procedures, and shrinkage mathematics.

---

## 4. Local Development & Setup

### Requirements
- Python 3.11+
- `uv` package manager (recommended) or `pip`

### Quick Start
```bash
# Clone the repository
git clone https://github.com/abdulsaboor190/tendency-tax.git
cd tendency-tax

# 1. Install dependencies
make setup
# Or with pip: pip install -r requirements.txt

# 2. Run test suite & linter
make test

# 3. Launch the local interactive app
make app
```
The Streamlit app will start at `http://localhost:8501`.

---

## 5. Deployment Options

Because the application is designed to render from the committed artifacts in `results/`, deployment requires **no raw data downloads** and minimal resources.

### Option A: Streamlit Community Cloud (Recommended & Simplest)

1. Push your repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. Click **"New app"**.
4. Configure the settings:
   - **Repository**: `abdulsaboor190/tendency-tax` (or your fork)
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
   - **Python version**: `3.11`
5. Click **Deploy!**
   - Streamlit Cloud automatically reads `.streamlit/config.toml` and installs dependencies from `requirements.txt` or `pyproject.toml`.

---

### Option B: Docker Container

Create a containerized build suitable for any container platform (AWS ECS, Google Cloud Run, Azure Container Apps, or DigitalOcean):

#### 1. `Dockerfile`
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency specifications
COPY pyproject.toml uv.lock ./

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy application code, configuration, and results
COPY app/ ./app/
COPY src/ ./src/
COPY results/ ./results/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["uv", "run", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. Build & Run
```bash
# Build the Docker image
docker build -t tendency-tax:latest .

# Run the container
docker run -p 8501:8501 tendency-tax:latest
```
Access at `http://localhost:8501`.

---

### Option C: Google Cloud Run / AWS ECS / Render

1. **Build and push** the Docker image to your container registry (e.g., Google Artifact Registry, Amazon ECR, Docker Hub).
2. **Deploy as Web Service**:
   - Port: `8501`
   - Health check path: `/_stcore/health`
   - Memory: 512 MB to 1 GB (memory footprint is lightweight since it reads pre-aggregated parquets)
   - CPU: 1 vCPU
3. Set environment variables (optional):
   - `STREAMLIT_SERVER_HEADLESS=true`
   - `STREAMLIT_SERVER_ENABLE_CORS=false`

---

## 6. Directory Structure Reference

```text
tendency-tax/
├── .github/workflows/ci.yml # Automated CI (ruff + pytest)
├── .streamlit/config.toml  # Streamlit theme & layout configuration
├── AGENTS.md               # Repository rules and operating constraints
├── Makefile                # Standardized developer commands
├── PLAN.md                 # Pre-registered analysis plan
├── SUMMARY.md              # Project overview and deployment guide (this file)
├── pyproject.toml          # uv configuration and pinned dependencies
├── requirements.txt        # pip-compatible requirements file
├── app/                    # Streamlit web application
│   ├── app.py              # Main multi-page entry point
│   ├── data.py             # Results loader and bundle caching
│   ├── ui/                 # Reusable components, theme, and charts
│   └── views/              # Page views (Test Panel, Explorer, Scout, Method)
├── data/                   # Raw play-by-play and intermediate caches (git-ignored)
├── docs/                   # Extended documentation and QA audits
├── results/                # Committed evaluation parquets and JSON artifacts
├── scripts/                # Data download and pipeline runners
├── src/                    # Feature engineering, modeling, and estimation library
└── tests/                  # Pytest test suite
```
