"""Predictive model and phat construction module.

AGENTS.md and PLAN.md are binding:
- Features are strictly config.FEATURES (the 10 situational features).
- Never use epa, xpass, success, wp, shotgun, or no_huddle as features.
- Never read or score holdout seasons (2024-2025) here.
- Hyperparameters chosen using log loss on TUNE split only.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from src import config
from src.load import get_split

# Global cache for early OOF models and tune model
_MODEL_CACHE: dict[str, Any] = {}


def fit_model(
    train_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> HistGradientBoostingClassifier:
    """Fit HistGradientBoostingClassifier predicting call_pass from config.FEATURES.

    - Features are strictly config.FEATURES (the 10 situation features).
    - Assert none of FORBIDDEN_FEATURES are in config.FEATURES.
    - early_stopping is set to False.
    - random_state is fixed from config.RANDOM_SEED.
    - Target is call_pass.
    """
    for forbidden in config.FORBIDDEN_FEATURES:
        if forbidden in config.FEATURES:
            raise ValueError(
                f"Forbidden feature '{forbidden}' detected in config.FEATURES"
            )

    model_params = dict(config.MODEL_PARAMS)
    if params:
        model_params.update(params)

    # Strictly enforce project rules
    model_params["early_stopping"] = False
    model_params["random_state"] = config.RANDOM_SEED

    X = train_df[config.FEATURES]
    y = train_df["call_pass"].astype(int)

    clf = HistGradientBoostingClassifier(**model_params)
    clf.fit(X, y)
    return clf


def tune_hyperparameters(
    early_df: pd.DataFrame | None = None,
    tune_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate candidate hyperparameters on early split and score log loss on tune.

    Prints the search table and returns (best_params, search_results_df).
    Does NOT inspect EPA or holdout data.
    """
    if early_df is None:
        early_df = get_split("early")
    if tune_df is None:
        tune_df = get_split("tune")

    X_tune = tune_df[config.FEATURES]
    y_tune = tune_df["call_pass"].astype(int)

    # Handful of candidate parameter combinations (shallow trees, small lr, L2 reg)
    candidate_list: list[dict[str, Any]] = [
        {"max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "l2_regularization": 1.0},
        {"max_depth": 5, "learning_rate": 0.05, "l2_regularization": 1.0},
        {"max_depth": 3, "learning_rate": 0.10, "l2_regularization": 1.0},
        {"max_depth": 4, "learning_rate": 0.10, "l2_regularization": 1.0},
        {"max_depth": 5, "learning_rate": 0.10, "l2_regularization": 1.0},
        {"max_depth": 4, "learning_rate": 0.05, "l2_regularization": 5.0},
        {"max_depth": 4, "learning_rate": 0.10, "l2_regularization": 5.0},
    ]
    for c in candidate_list:
        c["max_iter"] = 100

    records: list[dict[str, Any]] = []
    print("\nHyperparameter Search Table (Evaluated on TUNE split):")
    header = (
        f"{'max_depth':<10} | {'learning_rate':<14} | "
        f"{'l2_reg':<8} | {'max_iter':<8} | {'tune_log_loss':<14}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    best_params: dict[str, Any] = {}
    best_loss = float("inf")

    for p in candidate_list:
        clf = fit_model(early_df, params=p)
        preds = clf.predict_proba(X_tune)[:, 1]
        loss = float(log_loss(y_tune, preds))

        record = {
            "max_depth": p["max_depth"],
            "learning_rate": p["learning_rate"],
            "l2_regularization": p["l2_regularization"],
            "max_iter": p["max_iter"],
            "tune_log_loss": round(loss, 6),
        }
        records.append(record)

        if loss < best_loss:
            best_loss = loss
            best_params = p

        row_str = (
            f"{p['max_depth']:<10} | {p['learning_rate']:<14.2f} | "
            f"{p['l2_regularization']:<8.1f} | {p['max_iter']:<8} | {loss:<14.6f}"
        )
        print(row_str)

    print(separator)
    print(f"Chosen Hyperparameters (Lowest Tune Log Loss = {best_loss:.6f}):")
    print(f"  {best_params}\n")

    return best_params, pd.DataFrame(records)


def fit_early_oof_models(
    early_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """Fit leave-one-season-out models across early seasons (2016-2021).

    For each early season s:
      trains on all other early seasons in early_df,
      records training indices and the fitted model.
    """
    early_seasons = sorted(early_df["season"].unique())
    oof_models: dict[int, dict[str, Any]] = {}

    for s in early_seasons:
        train_idx = early_df.index[early_df["season"] != s]
        train_data = early_df.loc[train_idx]
        model = fit_model(train_data, params=params)
        oof_models[s] = {
            "model": model,
            "train_indices": train_idx,
            "held_out_season": s,
        }

    return oof_models


def add_phat(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    early_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build phat probability with exact pre-registered scheme:

    - early seasons (2016-2021): leave-one-season-out out-of-fold predictions
      (train on the other five early seasons, predict held-out season).
    - tune seasons (2022-2023): one model trained on all of 2016-2021.
    - holdout seasons (2024-2025): NOT scored here (set to NaN).
    """
    out = df.copy()
    out["phat"] = np.nan

    seasons_in_df = set(out["season"].unique())
    early_seasons_target = set(config.SPLITS["early"])
    tune_seasons_target = set(config.SPLITS["tune"])

    # Determine the early pool for training
    if early_df is not None:
        early_pool = early_df
    elif early_seasons_target.issubset(seasons_in_df):
        early_pool = out[out["season"].isin(early_seasons_target)]
    else:
        # Load canonical early split
        early_pool = get_split("early")

    use_cache = (params is None and early_df is None)

    # 1. Early seasons: leave-one-season-out out-of-fold
    early_present = sorted(seasons_in_df.intersection(early_seasons_target))
    if early_present:
        if use_cache and "oof_models" in _MODEL_CACHE:
            oof_models = _MODEL_CACHE["oof_models"]
        else:
            oof_models = fit_early_oof_models(early_pool, params=params)
            if use_cache:
                _MODEL_CACHE["oof_models"] = oof_models

        out.attrs["oof_models"] = oof_models

        for s in early_present:
            if s in oof_models:
                mask = out["season"] == s
                X_s = out.loc[mask, config.FEATURES]
                out.loc[mask, "phat"] = oof_models[s]["model"].predict_proba(X_s)[:, 1]

    # 2. Tune seasons: one model trained on all early seasons (2016-2021)
    tune_present = sorted(seasons_in_df.intersection(tune_seasons_target))
    if tune_present:
        if use_cache and "tune_model" in _MODEL_CACHE:
            tune_model = _MODEL_CACHE["tune_model"]
        else:
            tune_model = fit_model(early_pool, params=params)
            if use_cache:
                _MODEL_CACHE["tune_model"] = tune_model

        out.attrs["tune_model"] = tune_model

        tune_mask = out["season"].isin(tune_present)
        X_tune = out.loc[tune_mask, config.FEATURES]
        out.loc[tune_mask, "phat"] = tune_model.predict_proba(X_tune)[:, 1]

    # 3. Holdout seasons: NOT scored here (remain NaN)
    # No action required; out['phat'] already initialized to np.nan

    return out


def baseline_phat(
    df: pd.DataFrame,
    early_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute cell-level pass rate baseline for early and tune seasons.

    - early seasons (2016-2021): leave-one-season-out out-of-fold pass rates
      learned on the other 5 early seasons.
    - tune seasons (2022-2023): learned on all 2016-2021 early seasons.
    - Fallback: if a cell is missing from training data, fall back to the
      pass rate of its down x distance bucket (dd_bucket), then overall rate.
    - holdout seasons (2024-2025): NOT scored (remain NaN).

    Adds both 'phat_baseline' and 'baseline_phat' columns and returns df copy.
    """
    out = df.copy()
    out["phat_baseline"] = np.nan
    out["baseline_phat"] = np.nan

    seasons_in_df = set(out["season"].unique())
    early_seasons_target = set(config.SPLITS["early"])
    tune_seasons_target = set(config.SPLITS["tune"])

    if early_df is not None:
        early_pool = early_df
    elif early_seasons_target.issubset(seasons_in_df):
        early_pool = out[out["season"].isin(early_seasons_target)]
    else:
        early_pool = get_split("early")

    # 1. Early seasons: leave-one-season-out
    early_present = sorted(seasons_in_df.intersection(early_seasons_target))
    for s in early_present:
        mask = out["season"] == s
        train_data = early_pool[early_pool["season"] != s]
        cell_rates = train_data.groupby("cell_id")["call_pass"].mean()
        dd_rates = train_data.groupby("dd_bucket")["call_pass"].mean()
        overall_rate = float(train_data["call_pass"].mean())

        preds = out.loc[mask, "cell_id"].map(cell_rates)
        preds = (
            preds.fillna(out.loc[mask, "dd_bucket"].map(dd_rates))
            .fillna(overall_rate)
            .astype(float)
        )
        out.loc[mask, "phat_baseline"] = preds
        out.loc[mask, "baseline_phat"] = preds

    # 2. Tune seasons: learned on all early seasons (2016-2021)
    tune_present = sorted(seasons_in_df.intersection(tune_seasons_target))
    if tune_present:
        tune_mask = out["season"].isin(tune_present)
        cell_rates = early_pool.groupby("cell_id")["call_pass"].mean()
        dd_rates = early_pool.groupby("dd_bucket")["call_pass"].mean()
        overall_rate = float(early_pool["call_pass"].mean())

        preds = out.loc[tune_mask, "cell_id"].map(cell_rates)
        preds = (
            preds.fillna(out.loc[tune_mask, "dd_bucket"].map(dd_rates))
            .fillna(overall_rate)
            .astype(float)
        )
        out.loc[tune_mask, "phat_baseline"] = preds
        out.loc[tune_mask, "baseline_phat"] = preds

    # 3. Holdout seasons: NOT scored here (remain NaN)

    return out


class ScoreResult(NamedTuple):
    """Container for score() output."""

    log_loss: float
    brier_score: float
    calibration: pd.DataFrame


def score(df: pd.DataFrame, col: str) -> ScoreResult:
    """Compute log loss, Brier score, and 10-bin calibration table for predictor col.

    - Target is call_pass.
    - Probabilities clipped to [1e-6, 1-1e-6] before log loss.
    - Calibration table has exactly 10 equal-frequency bins.
    """
    y_true = df["call_pass"].astype(int)
    p = df[col].astype(float)

    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    ll = float(log_loss(y_true, p_clipped))
    bs = float(brier_score_loss(y_true, p))

    # 10 equal-frequency bins; rank method='first' breaks ties deterministically
    bin_series = pd.qcut(p.rank(method="first"), q=10, labels=range(1, 11))
    calib_df = pd.DataFrame(
        {
            "bin": bin_series,
            "pred": p,
            "obs": y_true,
        }
    )
    calib_table = (
        calib_df.groupby("bin", observed=False)
        .agg(
            mean_pred=("pred", "mean"),
            observed_pass_rate=("obs", "mean"),
            n=("obs", "count"),
        )
        .reset_index()
    )

    return ScoreResult(log_loss=ll, brier_score=bs, calibration=calib_table)


def evaluate_and_save_scores(
    raw_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Score model, baseline, and xpass on early and tune splits.

    - Verifies GATE G1: model must beat baseline on tune split log loss.
    - Saves results/model_scores.json.
    - Saves results/calibration_early.parquet and results/calibration_tune.parquet.
    """
    res_dir = Path(results_dir) if results_dir is not None else config.RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    scores_path = res_dir / "model_scores.json"
    calib_early_path = res_dir / "calibration_early.parquet"
    calib_tune_path = res_dir / "calibration_tune.parquet"

    print("\n--- Scoring Predictors on EARLY Split (2016-2021) ---")
    early_df = get_split("early", raw_dir=raw_dir)
    early_df = add_phat(early_df)
    early_df = baseline_phat(early_df)

    early_model = score(early_df, "phat")
    early_base = score(early_df, "phat_baseline")
    early_xpass = score(early_df, "xpass")

    print("\n--- Scoring Predictors on TUNE Split (2022-2023) ---")
    tune_df = get_split("tune", raw_dir=raw_dir)
    tune_df = add_phat(tune_df, early_df=early_df)
    tune_df = baseline_phat(tune_df, early_df=early_df)

    tune_model = score(tune_df, "phat")
    tune_base = score(tune_df, "phat_baseline")
    tune_xpass = score(tune_df, "xpass")

    # GATE G1: model must beat baseline on log loss on TUNE split
    print("\nGATE G1 Evaluation (TUNE Split Log Loss):")
    print(f"  Model Log Loss:    {tune_model.log_loss:.6f}")
    print(f"  Baseline Log Loss: {tune_base.log_loss:.6f}")
    if tune_model.log_loss >= tune_base.log_loss:
        diff = tune_model.log_loss - tune_base.log_loss
        raise RuntimeError(
            f"GATE G1 FAILED: Model log loss ({tune_model.log_loss:.6f}) did not beat "
            f"cell baseline log loss ({tune_base.log_loss:.6f}) by {diff:.6f}."
        )
    diff = tune_base.log_loss - tune_model.log_loss
    print(f"  [GATE G1 PASSED] Model beats baseline by {diff:.6f} log loss points!\n")

    # Build model_scores.json
    scores_records = [
        {
            "split": "early",
            "predictor": "model",
            "log_loss": round(early_model.log_loss, 6),
            "brier_score": round(early_model.brier_score, 6),
        },
        {
            "split": "early",
            "predictor": "baseline",
            "log_loss": round(early_base.log_loss, 6),
            "brier_score": round(early_base.brier_score, 6),
        },
        {
            "split": "early",
            "predictor": "xpass",
            "log_loss": round(early_xpass.log_loss, 6),
            "brier_score": round(early_xpass.brier_score, 6),
        },
        {
            "split": "tune",
            "predictor": "model",
            "log_loss": round(tune_model.log_loss, 6),
            "brier_score": round(tune_model.brier_score, 6),
        },
        {
            "split": "tune",
            "predictor": "baseline",
            "log_loss": round(tune_base.log_loss, 6),
            "brier_score": round(tune_base.brier_score, 6),
        },
        {
            "split": "tune",
            "predictor": "xpass",
            "log_loss": round(tune_xpass.log_loss, 6),
            "brier_score": round(tune_xpass.brier_score, 6),
        },
    ]

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(scores_records, f, indent=2)
    print(f"[SAVED] Model scores -> {scores_path}")

    # Build calibration dataframes
    def _combine_calib(
        split_name: str,
        m_cal: pd.DataFrame,
        b_cal: pd.DataFrame,
        x_cal: pd.DataFrame,
    ) -> pd.DataFrame:
        m_df = m_cal.copy()
        m_df["predictor"] = "model"
        b_df = b_cal.copy()
        b_df["predictor"] = "baseline"
        x_df = x_cal.copy()
        x_df["predictor"] = "xpass"
        combined = pd.concat([m_df, b_df, x_df], ignore_index=True)
        combined["split"] = split_name
        cols = ["split", "predictor", "bin", "mean_pred", "observed_pass_rate", "n"]
        return combined[cols]

    cal_early_df = _combine_calib(
        "early",
        early_model.calibration,
        early_base.calibration,
        early_xpass.calibration,
    )
    cal_early_df.to_parquet(calib_early_path, index=False, engine="pyarrow")
    print(f"[SAVED] Early calibration -> {calib_early_path}")

    cal_tune_df = _combine_calib(
        "tune",
        tune_model.calibration,
        tune_base.calibration,
        tune_xpass.calibration,
    )
    cal_tune_df.to_parquet(calib_tune_path, index=False, engine="pyarrow")
    print(f"[SAVED] Tune calibration -> {calib_tune_path}")

    # Print summary score table
    print("\n" + "=" * 62)
    print(f"{'Split':<8} | {'Predictor':<10} | {'Log Loss':<10} | {'Brier Score':<12}")
    print("-" * 62)
    for r in scores_records:
        print(
            f"{r['split']:<8} | {r['predictor']:<10} | "
            f"{r['log_loss']:<10.6f} | {r['brier_score']:<12.6f}"
        )
    print("=" * 62 + "\n")

    return {
        "scores": scores_records,
        "early_model": early_model,
        "early_base": early_base,
        "early_xpass": early_xpass,
        "tune_model": tune_model,
        "tune_base": tune_base,
        "tune_xpass": tune_xpass,
    }


def train_and_save_frozen_model(
    raw_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Train a FROZEN final model on 2016-2023 (early + tune) and save with joblib.

    Saves:
    - Model: results/models/phat_model_2016_2023.joblib
    - Metadata: results/models/phat_model_2016_2023.json

    This model is used in Phase 4 only. Do not run it on holdout.
    """
    models_dir = (
        Path(results_dir) / "models" if results_dir is not None else config.MODELS_DIR
    )
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "phat_model_2016_2023.joblib"
    meta_path = models_dir / "phat_model_2016_2023.json"

    early_df = get_split("early", raw_dir=raw_dir)
    tune_df = get_split("tune", raw_dir=raw_dir)
    train_df = pd.concat([early_df, tune_df], ignore_index=True)

    training_seasons = sorted(train_df["season"].unique().tolist())
    expected_seasons = config.SPLITS["early"] + config.SPLITS["tune"]
    if training_seasons != sorted(expected_seasons):
        raise ValueError(
            f"Expected training seasons {expected_seasons}, but got {training_seasons}"
        )

    print(
        f"\nTraining FROZEN final model on seasons {training_seasons} "
        f"({len(train_df):,d} plays)..."
    )
    clf = fit_model(train_df, params=config.MODEL_PARAMS)

    # Save model binary
    joblib.dump(clf, model_path)
    print(f"  [SAVED] Frozen model -> {model_path}")

    # Save metadata
    meta = {
        "sklearn_version": sklearn.__version__,
        "features": config.FEATURES,
        "params": config.MODEL_PARAMS,
        "training_seasons": training_seasons,
        "n_plays": len(train_df),
        "target": config.LABEL,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [SAVED] Frozen model metadata -> {meta_path}")

    return model_path, meta_path


if __name__ == "__main__":
    tune_hyperparameters()
    evaluate_and_save_scores()
    train_and_save_frozen_model()
