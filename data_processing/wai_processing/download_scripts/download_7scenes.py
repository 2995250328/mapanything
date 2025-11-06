# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
"""Download helper for the 7Scenes dataset and pseudo ground-truth poses."""

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from wai_processing.utils.download import extract_zip_archives, parallel_download

SCENE_NAMES: tuple[str, ...] = (
    "chess",
    "fire",
    "heads",
    "office",
    "pumpkin",
    "redkitchen",
    "stairs",
)

BASE_URL = (
    "http://download.microsoft.com/download/2/8/5/"
    "28564B23-0828-408F-8631-23B1EFF1DAC8"
)

VISLOC_REPO_URL = "https://github.com/tsattler/visloc_pseudo_gt_limitations.git"


def build_download_map(selected_scenes: Iterable[str] | None = None) -> dict[str, str]:
    """Return a mapping from archive names to their download URLs."""

    scenes = list(selected_scenes) if selected_scenes else list(SCENE_NAMES)
    return {f"{scene}.zip": f"{BASE_URL}/{scene}.zip" for scene in scenes}


def _collect_all_zips(target_dir: Path) -> Iterable[Path]:
    for path in target_dir.rglob("*.zip"):
        yield path


def _clone_visloc_repo(dest: Path, update: bool) -> None:
    """Clone or update the visloc pseudo-GT repository."""
    if dest.exists():
        if not update:
            print(f"visloc pseudo-GT repo already exists at {dest}, skipping clone.")
            return
        print(f"Updating visloc pseudo-GT repo in {dest}...")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True)
        return

    print(f"Cloning visloc pseudo-GT repo into {dest} ...")
    subprocess.run(["git", "clone", VISLOC_REPO_URL, str(dest)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download the original 7Scenes RGB-D sequences and the pseudo ground-"
            "truth repository used for PGT poses."
        )
    )
    parser.add_argument(
        "--target_dir",
        required=True,
        help="Directory where the raw scene archives should be downloaded.",
    )
    parser.add_argument(
        "--extract_dir",
        default=None,
        help="Directory where the RGB-D sequences should be unpacked.",
    )
    parser.add_argument(
        "--pgt_dir",
        default=None,
        help="Optional path where the visloc pseudo-GT repository should be cloned.",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of parallel workers for downloading and extraction.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["download", "extract", "pgt", "cleanup", "all"],
        help="Pipeline stages to run.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=SCENE_NAMES,
        help=(
            "Optional subset of scene names to download/extract. Defaults to all "
            "7Scenes sequences when omitted."
        ),
    )
    parser.add_argument(
        "--update_pgt",
        action="store_true",
        help="Update an existing pseudo-GT clone by running 'git pull'.",
    )
    parser.add_argument(
        "--delete_archives",
        action="store_true",
        help="Remove downloaded zip files after extraction completes.",
    )

    args = parser.parse_args()
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    extract_dir = Path(args.extract_dir) if args.extract_dir else target_dir / "extracted"
    stages = set(args.stages)
    run_all = "all" in stages

    if args.scenes:
        selected_scenes = tuple(dict.fromkeys(args.scenes))
        print(f"Restricting download/extraction to scenes: {', '.join(selected_scenes)}")
    else:
        selected_scenes = None

    if run_all or "download" in stages:
        print("Downloading 7Scenes archives ...")
        parallel_download(
            target_dir,
            build_download_map(selected_scenes),
            n_workers=args.n_workers,
        )

    if run_all or "extract" in stages:
        print("Extracting scene archives ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        for archive_name in build_download_map(selected_scenes):
            archive_path = target_dir / archive_name
            scene_extract_dir = extract_dir / archive_path.stem
            scene_extract_dir.mkdir(parents=True, exist_ok=True)
            extract_zip_archives(
                archive_path.parent,
                scene_extract_dir,
                n_workers=args.n_workers,
            )
            for nested_zip in list(scene_extract_dir.glob("*.zip")):
                nested_target = nested_zip.with_suffix("")
                extract_zip_archives(
                    nested_zip.parent,
                    nested_target,
                    n_workers=1,
                )
                nested_zip.unlink()

    if run_all or "cleanup" in stages:
        if args.delete_archives:
            print("Removing downloaded archives ...")
            for zip_path in _collect_all_zips(target_dir):
                try:
                    zip_path.unlink()
                except OSError as exc:
                    print(f"Warning: failed to delete {zip_path}: {exc}")
        else:
            print("Cleanup stage requested without --delete_archives; skipping archive removal.")

    if (run_all or "pgt" in stages) and args.pgt_dir:
        _clone_visloc_repo(Path(args.pgt_dir), update=args.update_pgt)


if __name__ == "__main__":
    main()
