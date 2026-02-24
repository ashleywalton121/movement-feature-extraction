"""
02_extract_thumbnails.py — Extract thumbnail frames for visual triage

Phase 1.2 of the M1 video-EEG pipeline. Reads video_metadata.csv (from
01_probe_videos.py) and extracts thumbnail frames from representative AVIs
for each folder. Output feeds 03_generate_contact_sheet.py.

For each folder, extracts 3 frames from the first AVI (start, middle, end)
to assess: camera framing, lighting, patient visibility, obstructions.

Usage:
    python scripts/02_extract_thumbnails.py                          # default (720p+)
    python scripts/02_extract_thumbnails.py --min-resolution 720     # explicit min height
    python scripts/02_extract_thumbnails.py --min-resolution 0       # all resolutions
    python scripts/02_extract_thumbnails.py --patient EM1334         # single patient
    python scripts/02_extract_thumbnails.py --parallel 8             # 8 workers
    python scripts/02_extract_thumbnails.py --dry-run                # show plan only
    python scripts/02_extract_thumbnails.py --resume                 # skip existing thumbnails
    python scripts/02_extract_thumbnails.py --last-avi               # also grab frames from last AVI
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
METADATA_CSV = OUTPUT_DIR / "video_metadata.csv"
THUMBS_DIR = OUTPUT_DIR / "thumbs"

DEFAULT_PARALLEL = 4
DEFAULT_MIN_RESOLUTION = 720  # minimum height in pixels
THUMB_QUALITY = 85  # JPEG quality (0-100)

# Frame positions to extract (fraction of total frames)
FRAME_POSITIONS = {
    "start": 0.02,   # 2% in (skip potential blank frames)
    "mid": 0.50,     # middle
    "end": 0.98,     # 2% from end
}


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------
def extract_frames_from_avi(
    avi_path: Path,
    output_prefix: str,
    output_dir: Path,
    positions: dict[str, float],
) -> list[dict]:
    """Extract frames at specified positions from an AVI file.

    Returns list of result dicts with: position, output_path, success, error.
    """
    results = []

    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return [{"position": p, "output_path": "", "success": False,
                     "error": "cv2 could not open file"} for p in positions]

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return [{"position": p, "output_path": "", "success": False,
                     "error": f"frame_count={frame_count}"} for p in positions]

        for pos_name, fraction in positions.items():
            frame_idx = max(0, min(int(frame_count * fraction), frame_count - 1))
            out_path = output_dir / f"{output_prefix}_{pos_name}.jpg"

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                cv2.imwrite(
                    str(out_path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY],
                )
                results.append({
                    "position": pos_name,
                    "output_path": str(out_path),
                    "success": True,
                    "error": "",
                })
            else:
                results.append({
                    "position": pos_name,
                    "output_path": "",
                    "success": False,
                    "error": f"failed to read frame {frame_idx}",
                })

        cap.release()

    except Exception as e:
        results = [{"position": p, "output_path": "", "success": False,
                    "error": str(e)} for p in positions]

    return results


# ---------------------------------------------------------------------------
# Folder processing
# ---------------------------------------------------------------------------
def process_folder(
    data_root: str,
    folder_name: str,
    patient_id: str,
    uuid: str,
    first_avi: str,
    last_avi: str | None,
    output_dir: Path,
    include_last: bool,
) -> list[dict]:
    """Extract thumbnails for a single folder.

    Returns list of result dicts with folder metadata + extraction results.
    """
    folder_path = Path(data_root) / folder_name
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # First AVI — always extract
    prefix_first = f"{patient_id}_{uuid}_first"
    first_path = folder_path / first_avi
    results = extract_frames_from_avi(
        first_path, prefix_first, patient_dir, FRAME_POSITIONS
    )
    for r in results:
        r.update({
            "patient_id": patient_id,
            "folder_name": folder_name,
            "uuid": uuid,
            "source_avi": first_avi,
            "avi_role": "first",
        })
    all_results.extend(results)

    # Last AVI — optional
    if include_last and last_avi and last_avi != first_avi:
        prefix_last = f"{patient_id}_{uuid}_last"
        last_path = folder_path / last_avi
        results = extract_frames_from_avi(
            last_path, prefix_last, patient_dir, FRAME_POSITIONS
        )
        for r in results:
            r.update({
                "patient_id": patient_id,
                "folder_name": folder_name,
                "uuid": uuid,
                "source_avi": last_avi,
                "avi_role": "last",
            })
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Build folder work list from metadata CSV
# ---------------------------------------------------------------------------
def build_folder_tasks(
    df: pd.DataFrame,
    min_resolution: int,
    patient_filter: str | None,
) -> list[dict]:
    """Build list of folders to process from video_metadata.csv.

    Filters by resolution and optionally by patient. For each folder,
    identifies the first and last AVI files.
    """
    # Filter by resolution using probed rows
    probed = df[df["is_probed"] == True].copy()
    if min_resolution > 0:
        # Get folders where probed height >= min_resolution
        qualified = probed[probed["height"] >= min_resolution][
            ["patient_id", "folder_name"]
        ].drop_duplicates()
        # Also exclude settings_changed folders where resolution drops below threshold
        # (keep only if ALL probed frames meet the threshold)
        folders_below = probed[probed["height"] < min_resolution][
            ["patient_id", "folder_name"]
        ].drop_duplicates()
        qualified = qualified.merge(
            folders_below, on=["patient_id", "folder_name"],
            how="left", indicator=True
        )
        qualified = qualified[qualified["_merge"] == "left_only"].drop(columns="_merge")
    else:
        qualified = probed[["patient_id", "folder_name"]].drop_duplicates()

    if patient_filter:
        qualified = qualified[qualified["patient_id"] == patient_filter]

    if qualified.empty:
        return []

    # For each qualified folder, find first and last AVI by avi_index
    tasks = []
    for _, row in qualified.iterrows():
        pid = row["patient_id"]
        fname = row["folder_name"]
        folder_df = df[(df["patient_id"] == pid) & (df["folder_name"] == fname)]
        folder_df = folder_df.sort_values("avi_index")

        uuid = folder_df["uuid"].iloc[0]
        first_avi = folder_df.iloc[0]["file_name"]
        last_avi = folder_df.iloc[-1]["file_name"]

        tasks.append({
            "patient_id": pid,
            "folder_name": fname,
            "uuid": uuid,
            "first_avi": first_avi,
            "last_avi": last_avi,
            "avi_count": len(folder_df),
        })

    return tasks


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def find_completed_folders(thumbs_dir: Path) -> set[str]:
    """Find UUIDs that already have thumbnails."""
    completed = set()
    if not thumbs_dir.exists():
        return completed
    for patient_dir in thumbs_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        for f in patient_dir.iterdir():
            if f.suffix == ".jpg" and "_first_start" in f.name:
                # Extract UUID from filename: {patient_id}_{uuid}_first_start.jpg
                parts = f.stem.split("_first_start")[0]
                # UUID is everything after the first underscore (patient_id_uuid)
                idx = parts.find("_")
                if idx > 0:
                    uuid = parts[idx + 1:]
                    completed.add(uuid)
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract thumbnail frames for visual triage of video folders"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(METADATA_CSV),
        help=f"Path to video_metadata.csv (default: {METADATA_CSV})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(THUMBS_DIR),
        help=f"Output directory for thumbnails (default: {THUMBS_DIR})",
    )
    parser.add_argument(
        "--min-resolution",
        type=int,
        default=DEFAULT_MIN_RESOLUTION,
        metavar="PX",
        help=f"Minimum video height in pixels (default: {DEFAULT_MIN_RESOLUTION})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        metavar="N",
        help=f"Number of concurrent workers (default: {DEFAULT_PARALLEL})",
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
        help="Show plan without extracting thumbnails",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip folders that already have thumbnails",
    )
    parser.add_argument(
        "--last-avi",
        action="store_true",
        help="Also extract frames from the last AVI in each folder",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root from config.yaml",
    )
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)

    # 1. Load metadata
    print(f"Loading metadata from {metadata_path} ...")
    df = pd.read_csv(metadata_path, low_memory=False)
    print(f"  {len(df):,} rows, {df['patient_id'].nunique()} patients, "
          f"{df[['patient_id', 'folder_name']].drop_duplicates().shape[0]} folders")

    # Get data_root from config.yaml if not overridden
    if args.data_root:
        data_root = args.data_root
    else:
        import yaml
        config_path = OUTPUT_DIR / "config.yaml"
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        data_root = config["data_root"]
    print(f"  Data root: {data_root}")

    # 2. Build work list
    tasks = build_folder_tasks(df, args.min_resolution, args.patient)
    if not tasks:
        print(f"No folders match filters (min_resolution={args.min_resolution}px"
              + (f", patient={args.patient}" if args.patient else "") + ")")
        return

    # Resume: skip already-done folders
    if args.resume:
        completed_uuids = find_completed_folders(output_dir)
        before = len(tasks)
        tasks = [t for t in tasks if t["uuid"] not in completed_uuids]
        skipped = before - len(tasks)
        print(f"  Resume: skipping {skipped} completed folders, {len(tasks)} remaining")

    if not tasks:
        print("Nothing to do.")
        return

    unique_patients = len(set(t["patient_id"] for t in tasks))
    total_avis = sum(t["avi_count"] for t in tasks)
    frames_per_folder = len(FRAME_POSITIONS) * (2 if args.last_avi else 1)
    total_frames = len(tasks) * frames_per_folder

    # 3. Summary
    print(f"\n{'='*60}")
    print(f"Thumbnail extraction plan:")
    print(f"  Min resolution:  {args.min_resolution}px height")
    print(f"  Patients:        {unique_patients}")
    print(f"  Folders:         {len(tasks)}")
    print(f"  Total AVIs:      {total_avis:,} (in selected folders)")
    print(f"  Frames/folder:   {frames_per_folder} ({', '.join(FRAME_POSITIONS.keys())})"
          + (" x2 (first+last AVI)" if args.last_avi else " (first AVI)"))
    print(f"  Total frames:    {total_frames:,}")
    print(f"  Parallel:        {args.parallel} workers")
    print(f"  Output:          {output_dir}")
    print(f"{'='*60}")

    if args.dry_run:
        print(f"\n[DRY RUN] Per-patient breakdown:")
        patient_summary = {}
        for t in tasks:
            pid = t["patient_id"]
            if pid not in patient_summary:
                patient_summary[pid] = {"folders": 0, "avis": 0}
            patient_summary[pid]["folders"] += 1
            patient_summary[pid]["avis"] += t["avi_count"]
        for pid in sorted(patient_summary):
            s = patient_summary[pid]
            n_frames = s["folders"] * frames_per_folder
            print(f"  {pid}: {s['folders']} folders, {n_frames} frames")
        print(f"\nDry run complete. Remove --dry-run to extract.")
        return

    # 4. Extract thumbnails
    print(f"\nExtracting thumbnails ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_results = []
    completed_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        future_to_task = {
            executor.submit(
                process_folder,
                data_root,
                task["folder_name"],
                task["patient_id"],
                task["uuid"],
                task["first_avi"],
                task["last_avi"],
                output_dir,
                args.last_avi,
            ): task
            for task in tasks
        }

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed_count += 1
            try:
                results = future.result()
                all_results.extend(results)
                folder_errors = sum(1 for r in results if not r["success"])
                if folder_errors:
                    error_count += folder_errors
            except Exception as e:
                error_count += frames_per_folder
                print(
                    f"  ERROR [{task['patient_id']}/{task['uuid']}]: {e}",
                    file=sys.stderr,
                )

            if completed_count % 20 == 0 or completed_count == len(tasks):
                elapsed = time.time() - t0
                rate = completed_count / elapsed if elapsed > 0 else 0
                print(
                    f"  {completed_count}/{len(tasks)} folders "
                    f"({elapsed:.1f}s, {rate:.1f} folders/s)"
                )

    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s")

    # 5. Write thumbnail index CSV
    if all_results:
        index_df = pd.DataFrame(all_results)
        index_path = output_dir / "thumbnail_index.csv"
        index_df.to_csv(index_path, index=False)
        print(f"Wrote thumbnail index: {index_path}")

    # 6. Summary
    successes = sum(1 for r in all_results if r["success"])
    failures = sum(1 for r in all_results if not r["success"])

    # Compute output size
    total_size = 0
    for r in all_results:
        if r["success"] and r["output_path"]:
            try:
                total_size += os.path.getsize(r["output_path"])
            except OSError:
                pass

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Folders processed:  {completed_count}")
    print(f"  Frames extracted:   {successes}")
    print(f"  Frames failed:      {failures}")
    print(f"  Total size:         {total_size / (1024*1024):.1f} MB")
    print(f"  Output directory:   {output_dir}")

    if failures > 0:
        print(f"\n  Failed extractions:")
        failed = [r for r in all_results if not r["success"]]
        # Show up to 10 failures
        for r in failed[:10]:
            print(f"    {r['patient_id']}/{r['uuid']} "
                  f"({r['position']}): {r['error']}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
