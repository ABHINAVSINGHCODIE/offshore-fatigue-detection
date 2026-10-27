"""
scripts/download_datasets.py
════════════════════════════
Cross-platform Python downloader for all Offshore Fatigue Detection datasets.

Usage:
    python scripts/download_datasets.py            # download all missing datasets
    python scripts/download_datasets.py --only uta # only extract UTA/YawDD
    python scripts/download_datasets.py --only mrl # only download MRL Eye
    python scripts/download_datasets.py --only coco mpii  # multiple targets
    python scripts/download_datasets.py --verify   # verify existing datasets

Datasets handled:
    1. UTA-RLDD + YawDD    — extracted from DATASET_ZIP (user-provided)
    2. MRL Eye Dataset      — via Kaggle API (kaggle.json required)
    3. MPII Human Pose      — direct URL download (~12.9 GB images + annotations)
    4. COCO Pose 2017       — direct URL download (~780 MB images + 240 MB ann)
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# ── Ensure project root is on sys.path ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("downloader")

# ── Download Sources ──────────────────────────────────────────────────────────
# All URLs are environment-overridable so the user can point to mirrors.
COCO_IMAGES_URL = os.environ.get(
    "OFFSHORE_COCO_IMG_URL",
    "http://images.cocodataset.org/zips/val2017.zip"
)
COCO_ANN_URL = os.environ.get(
    "OFFSHORE_COCO_ANN_URL",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
MPII_IMAGES_URL = os.environ.get(
    "OFFSHORE_MPII_IMG_URL",
    "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
)
MPII_ANN_URL = os.environ.get(
    "OFFSHORE_MPII_ANN_URL",
    "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip"
)
KAGGLE_MRL_SLUG = os.environ.get(
    "OFFSHORE_MRL_KAGGLE_SLUG",
    "akashshingha850/mrl-eye-dataset"
)
MRL_DIRECT_URL = os.environ.get(
    "OFFSHORE_MRL_DIRECT_URL",
    "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"
)

# ── Minimum file counts for validation ───────────────────────────────────────
MIN_FILE_COUNTS = {
    "uta_yawdd": int(os.environ.get("OFFSHORE_MIN_UTA_FILES",  "400")),
    "mrl_eye":   int(os.environ.get("OFFSHORE_MIN_MRL_FILES",  "10000")),
    "mpii_pose": int(os.environ.get("OFFSHORE_MIN_MPII_FILES", "100")),
    "coco_pose": int(os.environ.get("OFFSHORE_MIN_COCO_FILES", "100")),
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_files(directory: Path) -> int:
    """Recursively count files in a directory."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*") if _.is_file())


def _dir_is_ready(directory: Path, min_files: int) -> bool:
    """Return True if directory exists and has at least min_files files."""
    count = _count_files(directory)
    return count >= min_files


def _hr_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes //= 1024
    return f"{n_bytes:.1f} TB"


def _progress_hook(downloaded: int, total: int, bar_width: int = 40) -> None:
    if total <= 0:
        sys.stdout.buffer.write(
            f"\r  Downloaded {_hr_size(downloaded)}".encode("ascii", errors="replace")
        )
        sys.stdout.buffer.flush()
        return
    pct = downloaded / total
    filled = int(bar_width * pct)
    bar = "#" * filled + "-" * (bar_width - filled)
    line = f"\r  [{bar}] {pct*100:.1f}%  {_hr_size(downloaded)} / {_hr_size(total)}"
    sys.stdout.buffer.write(line.encode("ascii", errors="replace"))
    sys.stdout.buffer.flush()


def download_file_resumable(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """
    Download a file to dest, resuming if partially downloaded.
    Uses Range headers when the server supports them.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = Request(url, headers=headers)
    try:
        resp = urlopen(req, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Cannot connect to {url}: {e}") from e

    total = int(resp.headers.get("Content-Length", 0)) + existing
    log.info(f"  Target: {dest}")
    log.info(f"  Size  : {_hr_size(total) if total else 'unknown'}")
    if existing and resp.status == 206:
        log.info(f"  Resuming from {_hr_size(existing)} …")
    elif existing:
        log.info(f"  Server does not support resume — restarting download.")
        existing = 0

    mode = "ab" if existing else "wb"
    downloaded = existing
    t0 = time.time()
    with open(dest, mode) as fh:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            downloaded += len(buf)
            _progress_hook(downloaded, total)
    print()  # newline after progress bar
    elapsed = time.time() - t0
    log.info(f"  Done  : {_hr_size(downloaded)} in {elapsed:.0f}s")
    return dest


def extract_zip(zip_path: Path, dest_dir: Path, remove_after: bool = True) -> None:
    """Extract a .zip archive to dest_dir."""
    log.info(f"  Extracting {zip_path.name} → {dest_dir} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    log.info(f"  Extraction complete.")
    if remove_after:
        zip_path.unlink()
        log.info(f"  Removed intermediate ZIP: {zip_path.name}")


def extract_tar(tar_path: Path, dest_dir: Path, remove_after: bool = True) -> None:
    """Extract a .tar.gz archive to dest_dir."""
    import tarfile
    log.info(f"  Extracting {tar_path.name} → {dest_dir} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as t:
        t.extractall(dest_dir)
    log.info(f"  Extraction complete.")
    if remove_after:
        tar_path.unlink()
        log.info(f"  Removed intermediate TAR: {tar_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: UTA-RLDD + YawDD (from local ZIP)
# ─────────────────────────────────────────────────────────────────────────────

def setup_uta_yawdd() -> bool:
    """Extract Dataset Combined.zip to data/raw/uta_yawdd/ if not already done."""
    log.info("═" * 60)
    log.info("  Dataset 1: UTA-RLDD + YawDD")
    log.info("═" * 60)

    target = config.UTA_DIR
    min_files = MIN_FILE_COUNTS["uta_yawdd"]

    if _dir_is_ready(target, min_files):
        count = _count_files(target)
        log.info(f"  ✓ Already ready ({count} files in {target}). Skipping.")
        return True

    zip_path = config.DATASET_ZIP
    if not zip_path.exists():
        log.error(f"  ✗ Dataset ZIP not found: {zip_path}")
        log.error("    Set env var OFFSHORE_DATASET_ZIP to point to 'Dataset Combined.zip'")
        return False

    log.info(f"  Extracting from ZIP: {zip_path}")
    target.mkdir(parents=True, exist_ok=True)

    all_frames_alert    = 0
    all_frames_fatigued = 0
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        csvs  = [n for n in names if n.endswith(".csv") and "desktop.ini" not in n.lower()]
        log.info(f"  Found {len(csvs)} CSVs inside ZIP …")
        for csv_name in csvs:
            parts = csv_name.replace("\\", "/").split("/")
            # Determine alert/fatigued from folder name in zip
            label_folder = None
            for p in parts:
                pl = p.lower()
                if "alert" in pl:
                    label_folder = "Alert"
                elif "fatigue" in pl or "yawn" in pl:
                    label_folder = "Fatigued"
            if label_folder is None:
                label_folder = "Uncategorised"

            out_dir = target / label_folder
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / Path(csv_name).name

            with z.open(csv_name) as src, open(out_file, "wb") as dst:
                dst.write(src.read())

            if label_folder == "Alert":
                all_frames_alert += 1
            elif label_folder == "Fatigued":
                all_frames_fatigued += 1

    total = all_frames_alert + all_frames_fatigued
    log.info(f"  ✓ Extracted {total} CSVs  (Alert: {all_frames_alert}, Fatigued: {all_frames_fatigued})")
    log.info(f"  Saved to: {target}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: MRL Eye Dataset (Kaggle)
# ─────────────────────────────────────────────────────────────────────────────

def setup_mrl_eye() -> bool:
    """Download MRL Eye Dataset via Kaggle API or direct official university URL."""
    log.info("═" * 60)
    log.info("  Dataset 2: MRL Eye Dataset")
    log.info("═" * 60)

    target   = config.MRL_DIR
    min_files = MIN_FILE_COUNTS["mrl_eye"]

    if _dir_is_ready(target, min_files):
        count = _count_files(target)
        log.info(f"  ✓ Already ready ({count} files in {target}). Skipping.")
        return True

    target.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    used_kaggle = False

    if kaggle_json.exists():
        try:
            import kaggle  # noqa — triggers kaggle.json validation
            log.info(f"  Downloading '{KAGGLE_MRL_SLUG}' via Kaggle API …")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "kaggle", "datasets", "download",
                 "-d", KAGGLE_MRL_SLUG, "--unzip", "-p", str(target)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                used_kaggle = True
            else:
                log.warning(f"  Kaggle download failed ({result.stderr}), falling back to direct URL …")
        except Exception as e:
            log.warning(f"  Kaggle error ({e}), falling back to direct URL …")

    if not used_kaggle:
        log.info(f"  Downloading directly from official source: {MRL_DIRECT_URL} …")
        zip_path = target / "mrlEyes_2018_01.zip"
        try:
            download_file_resumable(MRL_DIRECT_URL, zip_path)
            extract_zip(zip_path, target, remove_after=True)
        except Exception as e:
            log.error(f"  ✗ Direct MRL download/extract failed: {e}")
            return False

    count = _count_files(target)
    log.info(f"  ✓ MRL Eye downloaded and extracted ({count} files).")
    log.info(f"  Saved to: {target}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 3: MPII Human Pose
# ─────────────────────────────────────────────────────────────────────────────

def setup_mpii_pose() -> bool:
    """Download and extract MPII Human Pose images + annotations."""
    log.info("═" * 60)
    log.info("  Dataset 3: MPII Human Pose")
    log.info("═" * 60)

    images_dir = config.MPII_DIR / "images"
    ann_dir    = config.MPII_DIR / "annotations"
    min_files  = MIN_FILE_COUNTS["mpii_pose"]

    all_ready = _dir_is_ready(images_dir, min_files) and _dir_is_ready(ann_dir, 1)
    if all_ready:
        img_count = _count_files(images_dir)
        log.info(f"  ✓ Already ready ({img_count} images). Skipping.")
        return True

    ok = True

    # Images (~12.9 GB)
    if not _dir_is_ready(images_dir, min_files):
        tar_path = config.MPII_DIR / "mpii_human_pose_v1.tar.gz"
        log.info(f"  ⚠ MPII images download is ~12.9 GB. This may take a while.")
        try:
            download_file_resumable(MPII_IMAGES_URL, tar_path)
            extract_tar(tar_path, images_dir, remove_after=True)
        except Exception as e:
            log.error(f"  ✗ MPII images download/extract failed: {e}")
            ok = False

    # Annotations
    if not _dir_is_ready(ann_dir, 1):
        ann_zip = config.MPII_DIR / "mpii_human_pose_v1_u12_2.zip"
        try:
            download_file_resumable(MPII_ANN_URL, ann_zip)
            extract_zip(ann_zip, ann_dir, remove_after=True)
        except Exception as e:
            log.error(f"  ✗ MPII annotations download/extract failed: {e}")
            ok = False

    if ok:
        log.info(f"  ✓ MPII Pose ready at: {config.MPII_DIR}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 4: COCO Pose 2017 (val set)
# ─────────────────────────────────────────────────────────────────────────────

def setup_coco_pose() -> bool:
    """Download and extract COCO Pose 2017 validation images + annotations."""
    log.info("═" * 60)
    log.info("  Dataset 4: COCO Pose 2017 (val set)")
    log.info("═" * 60)

    images_dir = config.COCO_DIR / "images"
    ann_dir    = config.COCO_DIR / "annotations"
    min_files  = MIN_FILE_COUNTS["coco_pose"]

    all_ready = _dir_is_ready(images_dir, min_files) and _dir_is_ready(ann_dir, 1)
    if all_ready:
        img_count = _count_files(images_dir)
        log.info(f"  ✓ Already ready ({img_count} images). Skipping.")
        return True

    ok = True

    # Images (~780 MB)
    if not _dir_is_ready(images_dir, min_files):
        img_zip = config.COCO_DIR / "val2017.zip"
        try:
            download_file_resumable(COCO_IMAGES_URL, img_zip)
            extract_zip(img_zip, images_dir, remove_after=True)
        except Exception as e:
            log.error(f"  ✗ COCO images download/extract failed: {e}")
            ok = False

    # Annotations (~240 MB)
    if not _dir_is_ready(ann_dir, 1):
        ann_zip = config.COCO_DIR / "annotations_trainval2017.zip"
        try:
            download_file_resumable(COCO_ANN_URL, ann_zip)
            extract_zip(ann_zip, ann_dir, remove_after=True)
        except Exception as e:
            log.error(f"  ✗ COCO annotations download/extract failed: {e}")
            ok = False

    if ok:
        log.info(f"  ✓ COCO Pose ready at: {config.COCO_DIR}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Summary & Verification
# ─────────────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    log.info("═" * 60)
    log.info("  DATASET VERIFICATION SUMMARY")
    log.info("═" * 60)

    entries = [
        ("UTA-RLDD + YawDD",  config.UTA_DIR,              MIN_FILE_COUNTS["uta_yawdd"]),
        ("MRL Eye",           config.MRL_DIR,              MIN_FILE_COUNTS["mrl_eye"]),
        ("MPII Images",       config.MPII_DIR / "images",  MIN_FILE_COUNTS["mpii_pose"]),
        ("MPII Annotations",  config.MPII_DIR / "annotations", 1),
        ("COCO Images",       config.COCO_DIR / "images",  MIN_FILE_COUNTS["coco_pose"]),
        ("COCO Annotations",  config.COCO_DIR / "annotations", 1),
    ]

    all_ok = True
    for name, path, minimum in entries:
        count  = _count_files(path)
        ready  = count >= minimum
        status = "✅ READY" if ready else f"❌ MISSING (found {count}, need ≥{minimum})"
        if not ready:
            all_ok = False
        log.info(f"  {name:<22}  {status}")
        if path.exists():
            log.info(f"    Path: {path}")

    log.info("")
    if all_ok:
        log.info("  All datasets ready.  Run: python train_models.py")
    else:
        log.info("  Some datasets are missing. Re-run with the appropriate --only flag.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    targets = args.only if args.only else ["uta", "mrl", "mpii", "coco"]

    results: dict[str, bool] = {}
    if "uta" in targets:
        results["uta"]  = setup_uta_yawdd()
    if "mrl" in targets:
        results["mrl"]  = setup_mrl_eye()
    if "mpii" in targets:
        results["mpii"] = setup_mpii_pose()
    if "coco" in targets:
        results["coco"] = setup_coco_pose()

    print_summary()

    failed = [k for k, v in results.items() if not v]
    if failed:
        log.warning(f"  Failed datasets: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract all Offshore Fatigue Detection datasets."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["uta", "mrl", "mpii", "coco"],
        help="Limit to specific datasets (e.g. --only uta coco)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify dataset readiness without downloading.",
    )
    args = parser.parse_args()

    if args.verify:
        print_summary()
    else:
        main(args)
