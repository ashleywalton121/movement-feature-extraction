"""
01_probe_videos.py — Probe AVI metadata for all patients

Phase 1.1 of the M1 video-EEG pipeline. Reads config.yaml (from 00_build_config.py)
and probes AVI files on N:\\DbData for metadata: resolution, fps, duration, file size,
codec. Uses OpenCV (cv2.VideoCapture) to read AVI headers without decoding frames.

Modes:
  - sample (default): Probe first + last AVI per folder, estimate rest from file size
  - --full:           Probe every AVI file (hours — warns before starting)
  - --sample-rate N:  Also probe every Nth file for more coverage

Usage:
    python scripts/01_probe_videos.py                        # sample mode
    python scripts/01_probe_videos.py --full                 # probe every file
    python scripts/01_probe_videos.py --sample-rate 10       # every 10th file too
    python scripts/01_probe_videos.py --parallel 8           # 8 concurrent folders
    python scripts/01_probe_videos.py --patient EM1283       # single patient
    python scripts/01_probe_videos.py --dry-run              # show plan, don't probe
    python scripts/01_probe_videos.py --resume               # skip folders already in CSV
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_YAML = OUTPUT_DIR / "config.yaml"
OUTPUT_CSV = OUTPUT_DIR / "video_metadata.csv"

DEFAULT_PARALLEL = 4


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> tuple[str, list[dict]]:
    """Read config.yaml and return (data_root, folder_tasks).

    folder_tasks is a flat list of dicts, one per has_video=True folder:
        {patient_id, folder_name, uuid, avi_count}
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_root = config["data_root"]
    folder_tasks = []

    for patient_id, patient_info in config["patients"].items():
        if patient_info.get("status") not in ("confirmed", "filtered"):
            continue
        for folder in patient_info.get("matched_folders", []):
            if not folder.get("has_video", False):
                continue
            folder_tasks.append({
                "patient_id": patient_id,
                "folder_name": folder["folder_name"],
                "uuid": folder.get("uuid", ""),
                "avi_count": folder.get("avi_count", 0),
            })

    return data_root, folder_tasks


# ---------------------------------------------------------------------------
# AVI scanning
# ---------------------------------------------------------------------------
def extract_avi_index(filename: str) -> int:
    """Extract the numeric index from AVI filenames like 'Study1_Video_0042.avi'.

    Returns -1 if no index found.
    """
    match = re.search(r"_(\d{4})\.avi$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: try any trailing digits before .avi
    match = re.search(r"(\d+)\.avi$", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1


def scan_folder_avis(folder_path: Path) -> list[dict]:
    """Use os.scandir to list all AVI files in a folder with size and mtime.

    Returns list of dicts sorted by avi_index:
        {file_name, avi_index, file_size_bytes, file_modified_iso}
    """
    avis = []
    try:
        with os.scandir(folder_path) as entries:
            for entry in entries:
                if not entry.is_file(follow_symlinks=False):
                    continue
                if not entry.name.lower().endswith(".avi"):
                    continue
                stat = entry.stat(follow_symlinks=False)
                avis.append({
                    "file_name": entry.name,
                    "avi_index": extract_avi_index(entry.name),
                    "file_size_bytes": stat.st_size,
                    "file_modified_iso": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
    except OSError as e:
        print(f"  WARNING: Cannot scan {folder_path}: {e}", file=sys.stderr)
        return []

    avis.sort(key=lambda x: x["avi_index"])
    return avis


# ---------------------------------------------------------------------------
# OpenCV probing
# ---------------------------------------------------------------------------
def probe_avi_metadata(avi_path: Path) -> dict:
    """Open an AVI with cv2.VideoCapture and read header properties.

    Returns dict with: width, height, fps, frame_count, duration_sec, codec_fourcc.
    On failure, returns dict with error key and NaN values.
    """
    result = {
        "width": float("nan"),
        "height": float("nan"),
        "fps": float("nan"),
        "frame_count": float("nan"),
        "duration_sec": float("nan"),
        "codec_fourcc": "",
        "error": "",
    }

    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            result["error"] = "cv2 could not open file"
            return result

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        cap.release()

        # Decode fourcc integer to 4-char string
        codec_fourcc = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).strip("\x00")

        duration_sec = frame_count / fps if fps > 0 else float("nan")

        result.update({
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "codec_fourcc": codec_fourcc,
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------------
def select_probe_indices(n_files: int, mode: str, sample_rate: int) -> set[int]:
    """Determine which file indices (0-based positions) to probe.

    Returns a set of indices into the sorted AVI list.
    """
    if n_files == 0:
        return set()
    if mode == "full":
        return set(range(n_files))

    # Sample mode: always probe first and last
    indices = {0, n_files - 1}

    # Add every Nth file if sample_rate specified
    if sample_rate and sample_rate > 0:
        indices.update(range(0, n_files, sample_rate))

    return indices


def process_folder(
    data_root: str,
    task: dict,
    mode: str,
    sample_rate: int,
) -> list[dict]:
    """Process a single folder: scan AVIs, probe selected ones, estimate rest.

    Returns list of row dicts (one per AVI file).
    """
    patient_id = task["patient_id"]
    folder_name = task["folder_name"]
    uuid = task["uuid"]
    folder_path = Path(data_root) / folder_name

    # Scan all AVIs in the folder
    avis = scan_folder_avis(folder_path)
    if not avis:
        return []

    # Determine which files to probe
    probe_indices = select_probe_indices(len(avis), mode, sample_rate)

    # Probe selected files
    probed_metadata = {}
    for idx in sorted(probe_indices):
        avi_path = folder_path / avis[idx]["file_name"]
        probed_metadata[idx] = probe_avi_metadata(avi_path)

    # Compute bytes_per_second from successfully probed files
    total_bytes = 0
    total_duration = 0.0
    for idx, meta in probed_metadata.items():
        dur = meta.get("duration_sec", float("nan"))
        size = avis[idx]["file_size_bytes"]
        if dur and dur > 0 and not (dur != dur):  # not NaN
            total_bytes += size
            total_duration += dur

    folder_bytes_per_sec = total_bytes / total_duration if total_duration > 0 else float("nan")

    # Detect settings_changed: compare first vs last probed file
    settings_changed = False
    if len(probed_metadata) >= 2:
        first_meta = probed_metadata.get(0, {})
        last_idx = max(probed_metadata.keys())
        last_meta = probed_metadata.get(last_idx, {})
        for key in ("width", "height", "fps", "codec_fourcc"):
            v1 = first_meta.get(key)
            v2 = last_meta.get(key)
            if v1 and v2 and v1 != v2:
                # Skip NaN comparisons for numeric fields
                if isinstance(v1, float) and v1 != v1:
                    continue
                if isinstance(v2, float) and v2 != v2:
                    continue
                settings_changed = True
                break

    # Build output rows
    rows = []
    for idx, avi in enumerate(avis):
        is_probed = idx in probed_metadata
        meta = probed_metadata.get(idx, {})

        if is_probed:
            estimated_duration = meta.get("duration_sec", float("nan"))
        elif folder_bytes_per_sec and not (folder_bytes_per_sec != folder_bytes_per_sec):
            estimated_duration = avi["file_size_bytes"] / folder_bytes_per_sec
        else:
            estimated_duration = float("nan")

        rows.append({
            "patient_id": patient_id,
            "folder_name": folder_name,
            "uuid": uuid,
            "avi_index": avi["avi_index"],
            "file_name": avi["file_name"],
            "width": meta.get("width", float("nan")),
            "height": meta.get("height", float("nan")),
            "fps": meta.get("fps", float("nan")),
            "frame_count": meta.get("frame_count", float("nan")),
            "duration_sec": meta.get("duration_sec", float("nan")),
            "codec_fourcc": meta.get("codec_fourcc", ""),
            "file_size_bytes": avi["file_size_bytes"],
            "file_modified_iso": avi["file_modified_iso"],
            "is_probed": is_probed,
            "estimated_duration_sec": estimated_duration,
            "folder_bytes_per_sec": folder_bytes_per_sec,
            "settings_changed": settings_changed,
            "error": meta.get("error", ""),
        })

    return rows


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_completed_folders(csv_path: Path) -> set[tuple[str, str]]:
    """Load (patient_id, folder_name) pairs already in the CSV."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["patient_id", "folder_name"])
        return set(zip(df["patient_id"], df["folder_name"]))
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Probe AVI metadata for all patients in config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_YAML),
        help=f"Path to config.yaml (default: {CONFIG_YAML})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_CSV),
        help=f"Output CSV path (default: {OUTPUT_CSV})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Probe every AVI file (slow — hours for full dataset)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        metavar="N",
        help="Also probe every Nth file (in addition to first/last)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        metavar="N",
        help=f"Number of concurrent folder workers (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        metavar="EM_ID",
        help="Process a single patient (for debugging)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without probing any files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folders already present in the output CSV",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)
    mode = "full" if args.full else "sample"

    # 1. Load config
    print(f"Loading config from {config_path} ...")
    data_root, folder_tasks = load_config(config_path)
    print(f"  Data root: {data_root}")
    print(f"  Found {len(folder_tasks)} folders with video")

    # Filter to single patient if requested
    if args.patient:
        folder_tasks = [t for t in folder_tasks if t["patient_id"] == args.patient]
        if not folder_tasks:
            print(f"ERROR: No video folders found for patient {args.patient}")
            sys.exit(1)
        print(f"  Filtered to patient {args.patient}: {len(folder_tasks)} folders")

    # Resume: skip already-completed folders
    if args.resume:
        completed = load_completed_folders(output_path)
        before = len(folder_tasks)
        folder_tasks = [
            t for t in folder_tasks
            if (t["patient_id"], t["folder_name"]) not in completed
        ]
        skipped = before - len(folder_tasks)
        print(f"  Resume: skipping {skipped} already-completed folders, {len(folder_tasks)} remaining")

    if not folder_tasks:
        print("Nothing to do.")
        return

    # Compute totals
    total_avi_files = sum(t["avi_count"] for t in folder_tasks)
    unique_patients = len(set(t["patient_id"] for t in folder_tasks))

    if mode == "sample":
        # First + last per folder, plus sample_rate
        estimated_probes = 0
        for t in folder_tasks:
            n = t["avi_count"]
            if n == 0:
                continue
            indices = select_probe_indices(n, mode, args.sample_rate)
            estimated_probes += len(indices)
        probe_desc = f"sample (first+last per folder)"
        if args.sample_rate:
            probe_desc += f" + every {args.sample_rate}th"
    else:
        estimated_probes = total_avi_files
        probe_desc = "full (every file)"

    # Summary
    print(f"\n{'='*60}")
    print(f"Probe plan:")
    print(f"  Patients:      {unique_patients}")
    print(f"  Folders:       {len(folder_tasks)}")
    print(f"  Total AVIs:    {total_avi_files:,}")
    print(f"  Mode:          {probe_desc}")
    print(f"  Probes:        ~{estimated_probes:,}")
    print(f"  Parallel:      {args.parallel} workers")
    print(f"  Output:        {output_path}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] Per-patient breakdown:")
        patient_summary = {}
        for t in folder_tasks:
            pid = t["patient_id"]
            if pid not in patient_summary:
                patient_summary[pid] = {"folders": 0, "avis": 0}
            patient_summary[pid]["folders"] += 1
            patient_summary[pid]["avis"] += t["avi_count"]
        for pid in sorted(patient_summary):
            s = patient_summary[pid]
            print(f"  {pid}: {s['folders']} folders, {s['avis']:,} AVIs")
        print("\nDry run complete. Remove --dry-run to probe.")
        return

    # Full mode warning
    if mode == "full" and total_avi_files > 5000:
        print(f"\nWARNING: Full mode will probe {total_avi_files:,} files.")
        print("This may take several hours. Press Ctrl+C to abort.")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    # 2. Process folders in parallel
    print(f"\nProbing AVIs ...")
    t0 = time.time()
    all_rows = []
    completed_count = 0
    error_folders = []

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_task = {
            executor.submit(
                process_folder, data_root, task, mode, args.sample_rate
            ): task
            for task in folder_tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed_count += 1
            try:
                rows = future.result()
                all_rows.extend(rows)
            except Exception as e:
                error_folders.append((task["patient_id"], task["folder_name"], str(e)))
                print(
                    f"  ERROR [{task['patient_id']}/{task['folder_name']}]: {e}",
                    file=sys.stderr,
                )

            # Progress update every 10 folders or at the end
            if completed_count % 10 == 0 or completed_count == len(folder_tasks):
                elapsed = time.time() - t0
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(
                    f"  {completed_count}/{len(folder_tasks)} folders "
                    f"({elapsed:.1f}s, {rate:.1f} folders/s, "
                    f"{len(all_rows):,} rows)"
                )

    elapsed = time.time() - t0
    print(f"\nProbing complete in {elapsed:.1f}s")

    if not all_rows:
        print("No data collected.")
        return

    # 3. Build DataFrame and sort
    df = pd.DataFrame(all_rows)
    df.sort_values(
        by=["patient_id", "folder_name", "avi_index"],
        inplace=True,
        ignore_index=True,
    )

    # 4. Write CSV (append if resuming)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and output_path.exists():
        # Append without header
        df.to_csv(output_path, mode="a", header=False, index=False)
        print(f"\nAppended {len(df):,} rows to {output_path}")
        # Re-read to get totals for summary
        df_total = pd.read_csv(output_path)
    else:
        df.to_csv(output_path, index=False)
        print(f"\nWrote {len(df):,} rows to {output_path}")
        df_total = df

    # 5. Print summary stats
    probed = df[df["is_probed"]]
    estimated = df[~df["is_probed"]]
    n_errors = df["error"].notna() & (df["error"] != "")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total rows:          {len(df_total):,}")
    print(f"  This run:            {len(df):,} rows")
    print(f"  Probed:              {len(probed):,}")
    print(f"  Estimated:           {len(estimated):,}")
    print(f"  Errors:              {n_errors.sum()}")
    print(f"  Folders processed:   {completed_count}")
    if error_folders:
        print(f"  Folder errors:       {len(error_folders)}")

    # Resolution breakdown from probed files
    if len(probed) > 0 and "width" in probed.columns:
        res_counts = (
            probed.dropna(subset=["width", "height"])
            .groupby(["width", "height"])
            .size()
            .sort_values(ascending=False)
        )
        if len(res_counts) > 0:
            print(f"\n  Resolutions found (from probed files):")
            for (w, h), count in res_counts.items():
                print(f"    {int(w)}x{int(h)}: {count} files")

    # FPS breakdown
    if len(probed) > 0 and "fps" in probed.columns:
        fps_counts = (
            probed.dropna(subset=["fps"])
            .groupby("fps")
            .size()
            .sort_values(ascending=False)
        )
        if len(fps_counts) > 0:
            print(f"\n  FPS values found:")
            for fps_val, count in fps_counts.items():
                print(f"    {fps_val}: {count} files")

    # Settings changed flags
    changed = df_total[df_total["settings_changed"] == True]
    if len(changed) > 0:
        changed_folders = changed[["patient_id", "folder_name"]].drop_duplicates()
        print(f"\n  Settings changed in {len(changed_folders)} folder(s):")
        for _, row in changed_folders.iterrows():
            print(f"    {row['patient_id']}/{row['folder_name']}")

    # Per-patient totals
    print(f"\n  Per-patient AVI counts:")
    patient_counts = df_total.groupby("patient_id").size().sort_index()
    for pid, count in patient_counts.items():
        est_dur = df_total[df_total["patient_id"] == pid]["estimated_duration_sec"].sum()
        hours = est_dur / 3600 if est_dur and not (est_dur != est_dur) else 0
        print(f"    {pid}: {count:>6,} AVIs  (~{hours:.1f}h)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
