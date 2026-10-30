"""Downloader for raw nflverse play-by-play files (Deliverable D10.3).

Downloads play_by_play_{year}.csv.gz (2016–2025) directly from the official
nflverse data releases (CC BY 4.0):
https://github.com/nflverse/nflverse-data/releases/download/pbp/

Data Hygiene:
- Raw play-by-play data is NEVER committed to git (strictly enforced by .gitignore).
- This downloader does NOT inspect, parse, or read any rows of the downloaded data.
- Existing files in data/raw/ are preserved and skipped to prevent duplicate downloads.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from src import config

NFLVERSE_PBP_URL_TEMPLATE: str = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/"
    "play_by_play_{year}.csv.gz"
)


def download_season(
    year: int,
    output_dir: Path,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Path:
    """Download a single season's play-by-play file if not already present.

    Parameters
    ----------
    year : int
        NFL season year (e.g. 2016 to 2025).
    output_dir : Path
        Target directory (e.g. data/raw/).
    max_retries : int
        Maximum retry attempts on network error.
    backoff_seconds : float
        Exponential backoff base delay in seconds.

    Returns
    -------
    Path
        Path to the downloaded or existing file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_gz = output_dir / f"play_by_play_{year}.csv.gz"
    target_csv = output_dir / f"play_by_play_{year}.csv"

    if target_gz.exists() and target_gz.stat().st_size > 0:
        print(f"[SKIP] Season {year}: {target_gz.name} already exists.")
        return target_gz

    if target_csv.exists() and target_csv.stat().st_size > 0:
        print(f"[SKIP] Season {year}: {target_csv.name} already exists.")
        return target_csv

    url = NFLVERSE_PBP_URL_TEMPLATE.format(year=year)
    temp_target = output_dir / f"play_by_play_{year}.csv.gz.tmp"

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            print(
                f"[DOWNLOADING] Season {year} from {url} "
                f"(attempt {attempt}/{max_retries})..."
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TendencyTax-Reproducibility/1.0"},
            )
            with (
                urllib.request.urlopen(req, timeout=60) as resp,
                open(temp_target, "wb") as out_f,
            ):
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

            # Atomic rename once complete
            temp_target.replace(target_gz)
            size_mb = target_gz.stat().st_size / 1e6
            print(
                f"[DONE] Season {year}: saved to {target_gz.name} "
                f"({size_mb:.1f} MB)."
            )
            return target_gz

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
            print(f"[WARN] Failed attempt {attempt} for season {year}: {e}")
            if temp_target.exists():
                temp_target.unlink()
            if attempt < max_retries:
                sleep_time = backoff_seconds * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    raise RuntimeError(
        f"Failed to download season {year} from {url} after "
        f"{max_retries} attempts: {last_err}"
    )


def download_all_seasons(
    raw_dir: Path | None = None,
    seasons: tuple[int, ...] = tuple(range(2016, 2026)),
) -> list[Path]:
    """Download all requested seasons into the raw data directory."""
    target_dir = raw_dir if raw_dir is not None else config.RAW_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading nflverse play-by-play files for seasons "
        f"{seasons[0]}–{seasons[-1]} into '{target_dir}'..."
    )
    paths: list[Path] = []
    for yr in seasons:
        p = download_season(yr, target_dir)
        paths.append(p)

    print("All requested seasons verified in raw data directory.")
    return paths


def main() -> None:
    """CLI entrypoint for scripts/download_data.py."""
    parser = argparse.ArgumentParser(
        description="Download raw play-by-play files from nflverse (CC BY 4.0)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Target folder for raw data (default: data/raw/)",
    )
    args = parser.parse_args()

    try:
        download_all_seasons(raw_dir=args.raw_dir)
    except Exception as exc:
        print(f"Error during download: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
