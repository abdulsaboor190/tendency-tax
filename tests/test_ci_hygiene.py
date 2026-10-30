"""Tests for repo-wide causal hygiene, manifest consistency, and app isolation (D10.1).

Covers:
  1. Repo-wide scan for banned words ('causes', 'proves', 'optimal', 'causal effect')
     in app/, results/ text files, and docs/ (markdown, text, json, python).
  2. Verification that results/MANIFEST.json lists every file in results/ (excluding
     MANIFEST.json, models directory, and .gitkeep), and every listed file validates
     against src/schema.py.
  3. App isolation: app/ never imports src.load, never references data/raw, and
     never sets TENDENCY_ALLOW_HOLDOUT.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.schema import (
    PARQUET_SCHEMAS,
    DevPrimaryResult,
    ModelDiagnosticsResult,
    ModelScoreItem,
    PowerResult,
    PrimaryResult,
    validate_parquet_df,
)


def test_repo_wide_no_banned_causal_words():
    """Scan app/, results/ text files, and docs for banned causal words.

    Banned terms: 'causes', 'proves', 'optimal', 'causal effect'.
    Fails with the exact file path and line number if found.
    """
    banned_patterns = [
        re.compile(r"\bcauses\b", re.IGNORECASE),
        re.compile(r"\bproves\b", re.IGNORECASE),
        re.compile(r"\boptimal\b", re.IGNORECASE),
        re.compile(r"\bcausal effect\b", re.IGNORECASE),
    ]

    target_dirs = [Path("app"), Path("results"), Path("docs")]
    text_extensions = {".py", ".md", ".json", ".txt", ".log", ".yaml", ".yml"}

    violations: list[str] = []

    for target in target_dirs:
        if not target.exists():
            continue
        for file_path in target.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in text_extensions:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                for pat in banned_patterns:
                    match = pat.search(line)
                    if match:
                        violations.append(
                            f"{file_path}:{line_no} [{match.group(0)}]: {line.strip()}"
                        )

    assert not violations, (
        f"Found {len(violations)} prohibited causal claim word violation(s):\n"
        + "\n".join(violations)
    )


def test_manifest_lists_every_result_file_and_validates():
    """Verify results/MANIFEST.json lists every file in results/ and validates.

    - Every file in results/ (except MANIFEST.json, .gitkeep, and directories)
      must be present in MANIFEST.json.
    - Every listed file must be validated against its corresponding schema in
      src/schema.py.
    """
    results_dir = Path("results")
    if not results_dir.exists():
        pytest.skip("results/ directory not present")

    manifest_path = results_dir / "MANIFEST.json"
    assert manifest_path.exists(), "results/MANIFEST.json is missing!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_files = manifest.get("files", {})

    # 1. Check all files in results/ are listed in manifest
    ignored_names = {"MANIFEST.json", ".gitkeep"}
    actual_files = set()
    for item in results_dir.iterdir():
        if item.is_file() and item.name not in ignored_names:
            actual_files.add(item.name)

    unlisted_files = actual_files - set(manifest_files.keys())
    assert not unlisted_files, (
        f"Files in results/ missing from MANIFEST.json: {unlisted_files}"
    )

    # 2. Validate every listed file against src/schema.py
    for filename in manifest_files.keys():
        file_path = results_dir / filename
        assert file_path.exists(), f"File listed in manifest does not exist: {filename}"

        if filename.endswith(".parquet"):
            # Determine schema_name from filename
            schema_name = filename.replace(".parquet", "")
            # Handle split/variant naming conventions:
            if "decile_gap" in schema_name:
                schema_key = "decile_gap"
            elif "placebo" in schema_name:
                schema_key = "placebo"
            elif "calibration" in schema_name:
                schema_key = "calibration"
            elif schema_name in PARQUET_SCHEMAS:
                schema_key = schema_name
            else:
                schema_key = schema_name

            assert schema_key in PARQUET_SCHEMAS, (
                f"No parquet schema defined for '{filename}' in src/schema.py"
            )
            import pandas as pd

            df = pd.read_parquet(file_path)
            validate_parquet_df(df, schema_name=schema_key, filename=filename)

        elif filename == "primary.json":
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            PrimaryResult.model_validate(data)

        elif filename == "dev_primary.json":
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            DevPrimaryResult.model_validate(data)

        elif filename == "power.json":
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            PowerResult.model_validate(data)

        elif filename == "model_scores.json":
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            assert isinstance(data, list)
            for item in data:
                ModelScoreItem.model_validate(item)

        elif filename == "model_diagnostics.json":
            with open(file_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            ModelDiagnosticsResult.model_validate(data)

        elif filename in ("RUN_LOG.md", "HOLDOUT_ACCESS.log"):
            # Plain text/log files: ensure readable and non-empty
            content = file_path.read_text(encoding="utf-8")
            assert len(content) > 0


def test_app_isolation_strictly_enforced():
    """Verify app/ never imports src.load, touches data/raw, or sets HOLDOUT."""
    app_dir = Path("app")
    assert app_dir.exists(), "app/ directory not found"

    forbidden_patterns = [
        re.compile(r"\bimport\s+src\.load\b"),
        re.compile(r"\bfrom\s+src\.load\s+import\b"),
        re.compile(r"data/raw"),
        re.compile(r"data\\raw"),
        re.compile(r"TENDENCY_ALLOW_HOLDOUT"),
    ]

    py_files = list(app_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python files found in app/"

    violations: list[str] = []
    for f in py_files:
        content = f.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), start=1):
            for pat in forbidden_patterns:
                if pat.search(line):
                    violations.append(
                        f"{f}:{line_no} matches {pat.pattern}: {line.strip()}"
                    )

    assert not violations, "App isolation violations:\n" + "\n".join(violations)
