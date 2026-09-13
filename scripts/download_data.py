#!/usr/bin/env python3
"""Download the case dataset from the Google Drive folder into data/raw/.

The 6 .npy files (~380 MB total) are intentionally not committed to this
repository (see .gitignore) so the repo stays small and the file selection
for graders happens explicitly. Run this once before anything else:

    python scripts/download_data.py

Requires `gdown` (already in requirements.txt). If the shared folder ever
moves, update DRIVE_FOLDER_URL below or pass --url.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1lo_hfgbo5hIuHCq4Jhg2tiUTXWXt017K"
)
EXPECTED_FILES = [
    "Classes.npy",
    "Dados_1.npy",
    "Dados_2.npy",
    "Dados_3.npy",
    "Dados_4.npy",
    "Dados_5.npy",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DRIVE_FOLDER_URL, help="Google Drive folder URL")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="destination directory (default: data/raw)",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        print("gdown is required: pip install -r requirements.txt", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download_folder(url=args.url, output=str(args.out), quiet=False)
    if not downloaded:
        print("gdown reported no files downloaded; check the folder URL/permissions.",
              file=sys.stderr)
        return 1

    # gdown.download_folder recreates the Drive folder name as a subdirectory;
    # flatten it into --out so downstream paths (configs/default.yaml) stay simple.
    for path_str in downloaded:
        path = Path(path_str)
        if path.parent != args.out and path.exists():
            shutil.move(str(path), str(args.out / path.name))

    missing = [f for f in EXPECTED_FILES if not (args.out / f).exists()]
    if missing:
        print(f"Warning: expected files still missing after download: {missing}", file=sys.stderr)
        return 1

    print(f"All {len(EXPECTED_FILES)} files present in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
