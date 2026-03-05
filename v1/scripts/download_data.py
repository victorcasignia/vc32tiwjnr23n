"""
Download popular super-resolution benchmark datasets.

Supported datasets:
  - DIV2K      (800 train + 100 val, ×2/×3/×4 LR)
  - Flickr2K   (2650 images)
  - Set5       (5 images)
  - Set14      (14 images)
  - BSD100     (100 images)
  - Urban100   (100 images)
  - Manga109   (109 images)

Usage:
    python -m scripts.download_data --datasets div2k set5 set14 bsd100 urban100
    python -m scripts.download_data --datasets all --data-dir ./data
"""

import argparse
import logging
import os
import sys
import time as _time
import zipfile
import tarfile
import shutil
from pathlib import Path
from urllib.request import urlopen, Request

from tqdm import tqdm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

DATASET_URLS = {
    # DIV2K — official download links
    "div2k_train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "div2k_val_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
    "div2k_train_lr_x2": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
    "div2k_train_lr_x3": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X3.zip",
    "div2k_train_lr_x4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
    "div2k_val_lr_x2": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
    "div2k_val_lr_x3": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X3.zip",
    "div2k_val_lr_x4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",

    # Flickr2K
    "flickr2k": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",

    # Benchmark test sets (commonly hosted on GitHub / Google Drive)
    "benchmark": "https://cv.snu.ac.kr/research/EDSR/benchmark.tar",
}


def download_file(url: str, dest: str, max_retries: int = 5) -> str:
    """Download a file with tqdm progress bar and retry logic."""
    if os.path.exists(dest):
        log.info("Already downloaded: %s", dest)
        return dest

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fname = os.path.basename(dest)
    headers = {"User-Agent": "Mozilla/5.0 (DCNO-SR dataset downloader)"}

    for attempt in range(1, max_retries + 1):
        try:
            log.info("Downloading %s (attempt %d/%d)", url, attempt, max_retries)
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as resp:
                total_header = resp.headers.get("Content-Length")
                total = int(total_header) if total_header and total_header.isdigit() else None

                with (
                    open(dest, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"  {fname}",
                        dynamic_ncols=True,
                        miniters=1,
                        smoothing=0.1,
                        leave=True,
                        file=sys.stdout,
                        bar_format=(
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                            "[{elapsed}<{remaining}, {rate_fmt}]"
                        ) if total else None,
                    ) as pbar,
                ):
                    while True:
                        chunk = resp.read(1 << 16)  # 64 KB
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            return dest
        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt, e)
            if os.path.exists(dest):
                os.remove(dest)
            if attempt < max_retries:
                wait = min(2 ** attempt, 30)
                log.info("Retrying in %ds...", wait)
                _time.sleep(wait)
            else:
                log.error("All %d attempts failed for %s", max_retries, url)
                raise


def extract_archive(archive_path: str, extract_to: str):
    """Extract zip or tar archive with tqdm progress."""
    log.info("Extracting %s", archive_path)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            for m in tqdm(members, desc="  Extracting", unit="file", leave=False):
                zf.extract(m, extract_to)
    elif archive_path.endswith(".tar") or archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            for m in tqdm(members, desc="  Extracting", unit="file", leave=False):
                tf.extract(m, extract_to)
    else:
        log.warning("Unknown archive format: %s", archive_path)


def download_div2k(data_dir: str, scales: list = [2, 3, 4]):
    """Download DIV2K dataset (HR + bicubic LR at given scales)."""
    log.info("=== DIV2K Dataset ===")
    div2k_dir = os.path.join(data_dir, "DIV2K")
    cache_dir = os.path.join(data_dir, "_cache")

    parts = [f"div2k_{s}_hr" for s in ("train", "val")]
    for scale in scales:
        parts += [f"div2k_{s}_lr_x{scale}" for s in ("train", "val")]

    for key in tqdm(parts, desc="DIV2K parts", unit="part"):
        ext = ".zip"
        archive = download_file(
            DATASET_URLS[key],
            os.path.join(cache_dir, f"{key}{ext}"),
        )
        extract_archive(archive, div2k_dir)

    log.info("DIV2K ready at %s", div2k_dir)


def download_flickr2k(data_dir: str):
    """Download Flickr2K dataset."""
    log.info("=== Flickr2K Dataset ===")
    cache_dir = os.path.join(data_dir, "_cache")
    archive = download_file(
        DATASET_URLS["flickr2k"],
        os.path.join(cache_dir, "Flickr2K.tar"),
    )
    extract_archive(archive, data_dir)
    log.info("Flickr2K ready at %s", os.path.join(data_dir, "Flickr2K"))


def download_benchmark(data_dir: str):
    """Download benchmark test sets (Set5, Set14, BSD100, Urban100)."""
    log.info("=== Benchmark Datasets (Set5, Set14, BSD100, Urban100) ===")
    cache_dir = os.path.join(data_dir, "_cache")
    archive = download_file(
        DATASET_URLS["benchmark"],
        os.path.join(cache_dir, "benchmark.tar"),
    )
    extract_archive(archive, data_dir)
    log.info("Benchmark sets ready at %s", os.path.join(data_dir, "benchmark"))


DATASET_DOWNLOADERS = {
    "div2k": download_div2k,
    "flickr2k": download_flickr2k,
    "set5": download_benchmark,
    "set14": download_benchmark,
    "bsd100": download_benchmark,
    "urban100": download_benchmark,
    "manga109": download_benchmark,
}


def main():
    parser = argparse.ArgumentParser(description="Download SR datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["div2k"],
        choices=list(DATASET_DOWNLOADERS.keys()) + ["all", "df2k"],
        help="Datasets to download",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store datasets",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="LR scales to download (for DIV2K)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    log.info("Data directory: %s", data_dir)

    datasets = args.datasets
    if "all" in datasets:
        datasets = list(DATASET_DOWNLOADERS.keys())
    if "df2k" in datasets:
        datasets = ["div2k", "flickr2k"] + [d for d in datasets if d not in ["df2k", "div2k", "flickr2k"]]

    # Avoid downloading benchmark multiple times
    benchmark_downloaded = False
    for name in datasets:
        if name in ["set5", "set14", "bsd100", "urban100", "manga109"]:
            if not benchmark_downloaded:
                download_benchmark(data_dir)
                benchmark_downloaded = True
        elif name == "div2k":
            download_div2k(data_dir, args.scales)
        elif name == "flickr2k":
            download_flickr2k(data_dir)

    log.info("All downloads complete!")


if __name__ == "__main__":
    main()
