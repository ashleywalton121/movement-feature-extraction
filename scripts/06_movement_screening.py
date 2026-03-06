"""
06_movement_screening.py — Cross-patient movement screening

Quick movement survey across multiple patients using frame differencing on
a sparse sample of AVIs per usable folder. Produces per-AVI summary stats
and a cross-patient comparison table to identify clips with similar
movement levels.

Processes one patient at a time and saves results incrementally.

Usage:
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287
    python scripts/06_movement_screening.py --patients EM1279 --sample-size 5
    python scripts/06_movement_screening.py --patients EM1279 --dry-run
    python scripts/06_movement_screening.py --patients EM1279 --timeout 60
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287 --tag-brightness
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287 --scan-all
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287 --scan-all --workers 8
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287 --build-copy-list
    python scripts/06_movement_screening.py --patients EM1279,EM1269,EM1201,EM1287 --thumbnails
"""

import argparse
import base64
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
METADATA_CSV = OUTPUT_DIR / "video_metadata.csv"
CONFIG_YAML = OUTPUT_DIR / "config.yaml"
DEFAULT_TRIAGE_CSV = Path(r"C:\Users\ashle\Downloads\video_triage.csv")
SCREENING_DIR = OUTPUT_DIR / "movement_screening"

DEFAULT_SAMPLE_SIZE = 10   # AVIs per folder
DEFAULT_TIMEOUT = 90       # seconds per AVI before skipping
MOTION_THRESHOLD = 5.0     # mean_diff value to consider "moving"


# ---------------------------------------------------------------------------
# Data loading (reuses patterns from 04/05)
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_triage(triage_path: Path, patient_ids: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(triage_path)
    usable = df[df["usable"].str.lower() == "yes"].copy()
    if patient_ids:
        usable = usable[usable["patient_id"].isin(patient_ids)]
    return usable


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, low_memory=False)


def build_folder_tasks(
    triage: pd.DataFrame,
    metadata: pd.DataFrame,
) -> list[dict]:
    """Merge triage (usable) + metadata (AVI details) into a work list."""
    tasks = []
    for _, trow in triage.iterrows():
        pid = trow["patient_id"]
        uuid = trow["uuid"]
        folder_name = trow["folder_name"]

        folder_df = metadata[
            (metadata["patient_id"] == pid) & (metadata["uuid"] == uuid)
        ].sort_values("avi_index")

        if folder_df.empty:
            print(f"  WARNING: no metadata for {pid}/{uuid}, skipping",
                  file=sys.stderr)
            continue

        avi_list = []
        cumulative = 0.0
        for _, mrow in folder_df.iterrows():
            dur = mrow["estimated_duration_sec"]
            if pd.isna(dur) or dur <= 0:
                dur = 120.0
            avi_list.append({
                "file_name": mrow["file_name"],
                "avi_index": int(mrow["avi_index"]),
                "duration": dur,
                "start_time": cumulative,
            })
            cumulative += dur

        probed = folder_df[folder_df["is_probed"] == True]
        if not probed.empty:
            p = probed.iloc[0]
            fps = float(p["fps"])
        else:
            fps = 30.0

        tasks.append({
            "patient_id": pid,
            "uuid": uuid,
            "folder_name": folder_name,
            "avi_list": avi_list,
            "total_duration": cumulative,
            "avi_count": len(folder_df),
            "fps": fps,
        })

    return tasks


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_avis(avi_list: list[dict], n: int) -> list[dict]:
    """Pick n evenly-spaced AVIs from the list."""
    total = len(avi_list)
    if total <= n:
        return avi_list
    indices = np.linspace(0, total - 1, n, dtype=int)
    return [avi_list[i] for i in indices]


# ---------------------------------------------------------------------------
# Frame differencing (adapted from extract_frame_diff.py)
# ---------------------------------------------------------------------------
def compute_motion_stats(video_path: Path) -> dict | None:
    """Run frame differencing on one AVI, return summary stats."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    first_frame_brightness = float(np.mean(prev_gray))
    hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    first_frame_saturation = float(np.mean(hsv[:, :, 1]))
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0).astype(np.float32)

    mean_diffs = []
    area_fracs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)

        diff = np.abs(gray - prev_gray)
        mean_diffs.append(np.mean(diff))
        area_fracs.append(np.mean(diff > 10))

        prev_gray = gray

    cap.release()

    if not mean_diffs:
        return None

    arr = np.array(mean_diffs)
    area_arr = np.array(area_fracs)

    return {
        "frames_processed": frame_idx,
        "duration_sec": frame_idx / fps,
        "fps": fps,
        "mean_brightness": first_frame_brightness,
        "mean_saturation": first_frame_saturation,
        "is_grayscale": first_frame_saturation < GRAYSCALE_SATURATION_THRESHOLD,
        "mean_motion": float(np.mean(arr)),
        "median_motion": float(np.median(arr)),
        "peak_motion_95": float(np.percentile(arr, 95)),
        "std_motion": float(np.std(arr)),
        "mean_area_frac": float(np.mean(area_arr)),
        "pct_time_above_threshold": float(np.mean(arr > MOTION_THRESHOLD) * 100),
    }


def compute_motion_stats_with_timeout(video_path: Path, timeout: int) -> dict | None:
    """Run compute_motion_stats with a timeout. Returns None on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compute_motion_stats, video_path)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            return None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Thumbnail extraction
# ---------------------------------------------------------------------------
THUMB_WIDTH = 480
THUMB_QUALITY = 85


def extract_thumbnail(avi_path: Path, output_path: Path) -> bool:
    """Open AVI, seek to 50%, grab one frame, resize to 480px wide, save JPEG."""
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False

    mid = max(0, frame_count // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return False

    h, w = frame.shape[:2]
    if w > 0:
        new_w = THUMB_WIDTH
        new_h = int(h * new_w / w)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    return True


def extract_thumbnail_with_timeout(
    avi_path: Path, output_path: Path, timeout: int,
) -> bool:
    """Run extract_thumbnail with a timeout. Returns False on timeout/error."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(extract_thumbnail, avi_path, output_path)
        try:
            return future.result(timeout=timeout)
        except (FuturesTimeout, Exception):
            return False


# ---------------------------------------------------------------------------
# Brightness measurement (lightweight — one frame only)
# ---------------------------------------------------------------------------
GRAYSCALE_SATURATION_THRESHOLD = 15  # mean saturation below this = grayscale/IR


def measure_frame_properties(avi_path: Path) -> dict | None:
    """Open AVI, read first frame, return brightness + saturation metrics."""
    cap = cv2.VideoCapture(str(avi_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mean_sat = float(np.mean(hsv[:, :, 1]))
    return {
        "mean_brightness": float(np.mean(gray)),
        "mean_saturation": mean_sat,
        "is_grayscale": mean_sat < GRAYSCALE_SATURATION_THRESHOLD,
    }


def measure_frame_properties_with_timeout(avi_path: Path, timeout: int) -> dict | None:
    """Run measure_frame_properties with a timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(measure_frame_properties, avi_path)
        try:
            return future.result(timeout=timeout)
        except (FuturesTimeout, Exception):
            return None


# ---------------------------------------------------------------------------
# Process one folder
# ---------------------------------------------------------------------------
def process_folder(
    task: dict, data_root: str, sample_size: int, timeout: int,
) -> list[dict]:
    """Sample AVIs from one folder and compute motion stats for each."""
    pid = task["patient_id"]
    folder_name = task["folder_name"]
    uuid_short = task["uuid"][:10]
    folder_path = Path(data_root) / folder_name

    if not folder_path.exists():
        print(f"    WARNING: folder not found: {folder_path}", file=sys.stderr)
        return []

    sampled = sample_avis(task["avi_list"], sample_size)
    results = []

    for i, avi in enumerate(sampled, 1):
        avi_path = folder_path / avi["file_name"]
        label = f"    [{i}/{len(sampled)}] idx {avi['avi_index']:>4}"

        print(f"{label}  processing...", end=" ", flush=True)
        t0 = time.time()
        stats = compute_motion_stats_with_timeout(avi_path, timeout)
        elapsed = time.time() - t0

        row = {
            "patient_id": pid,
            "uuid": task["uuid"],
            "folder_name": folder_name,
            "avi_file": avi["file_name"],
            "avi_index": avi["avi_index"],
            "recording_offset_sec": avi["start_time"],
            "total_avis_in_folder": task["avi_count"],
            "total_duration_sec": task["total_duration"],
        }

        if stats:
            row.update(stats)
            row["status"] = "ok"
            print(f"{stats['frames_processed']} frames, "
                  f"mean={stats['mean_motion']:.2f}, "
                  f"{elapsed:.1f}s")
        elif elapsed >= timeout - 1:
            row["status"] = "timeout"
            print(f"TIMEOUT ({timeout}s)")
        else:
            row["status"] = "failed"
            print(f"FAILED ({elapsed:.1f}s)")

        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Activity band + brightness classification
# ---------------------------------------------------------------------------
COL_ORDER = [
    "patient_id", "uuid", "folder_name", "avi_file", "avi_index",
    "recording_offset_sec", "total_avis_in_folder", "total_duration_sec",
    "status", "frames_processed", "duration_sec", "fps",
    "mean_brightness", "brightness_bin", "mean_saturation", "is_grayscale",
    "mean_motion", "median_motion", "peak_motion_95", "std_motion",
    "mean_area_frac", "pct_time_above_threshold",
]


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COL_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in cols]
    return df[cols + remaining]


def classify_activity(mean_motion: float) -> str:
    """Classify a mean_motion value into an activity band."""
    if mean_motion < 1.0:
        return "quiet"
    elif mean_motion < 3.0:
        return "low"
    elif mean_motion < 6.0:
        return "moderate"
    else:
        return "high"


def classify_brightness(mean_brightness: float) -> str:
    """Classify mean frame brightness (0-255) into a lighting bin."""
    if mean_brightness < 30:
        return "dark"
    elif mean_brightness < 80:
        return "dim"
    elif mean_brightness < 160:
        return "normal"
    else:
        return "bright"


# ---------------------------------------------------------------------------
# Cross-patient comparison
# ---------------------------------------------------------------------------
def print_cross_patient_summary(df: pd.DataFrame):
    """Print a console summary comparing movement across patients."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo successful clips to compare.")
        return

    ok["activity_band"] = ok["mean_motion"].apply(classify_activity)

    # --- Per-folder summary ---
    print("\n" + "=" * 80)
    print("PER-FOLDER SUMMARY")
    print("=" * 80)

    folder_agg = ok.groupby(["patient_id", "uuid"]).agg(
        folder_name=("folder_name", "first"),
        clips_sampled=("avi_file", "count"),
        total_avis=("total_avis_in_folder", "first"),
        folder_duration_hr=("total_duration_sec", lambda x: x.iloc[0] / 3600),
        mean_motion=("mean_motion", "mean"),
        median_motion=("median_motion", "mean"),
        peak_95=("peak_motion_95", "mean"),
        mean_area_frac=("mean_area_frac", "mean"),
        pct_above_thresh=("pct_time_above_threshold", "mean"),
    ).reset_index()

    folder_agg["activity_band"] = folder_agg["mean_motion"].apply(classify_activity)
    folder_agg = folder_agg.sort_values("mean_motion", ascending=False)

    print(f"\n{'Patient':<10} {'Folder UUID (short)':<12} {'AVIs':>5} "
          f"{'Dur(hr)':>8} {'Mean':>7} {'Med':>7} {'P95':>7} "
          f"{'Area%':>7} {'%>Thr':>7} {'Band':<10}")
    print("-" * 95)

    for _, r in folder_agg.iterrows():
        print(f"{r['patient_id']:<10} {r['uuid'][:10]:<12} "
              f"{r['total_avis']:>5.0f} {r['folder_duration_hr']:>8.1f} "
              f"{r['mean_motion']:>7.2f} {r['median_motion']:>7.2f} "
              f"{r['peak_95']:>7.2f} {r['mean_area_frac']:>7.3f} "
              f"{r['pct_above_thresh']:>7.1f} {r['activity_band']:<10}")

    # --- Per-patient summary ---
    print("\n" + "=" * 80)
    print("PER-PATIENT SUMMARY")
    print("=" * 80)

    patient_agg = ok.groupby("patient_id").agg(
        folders=("uuid", "nunique"),
        clips=("avi_file", "count"),
        mean_motion=("mean_motion", "mean"),
        median_motion=("median_motion", "mean"),
        peak_95=("peak_motion_95", "mean"),
        pct_above_thresh=("pct_time_above_threshold", "mean"),
    ).reset_index()

    patient_agg["activity_band"] = patient_agg["mean_motion"].apply(classify_activity)
    patient_agg = patient_agg.sort_values("mean_motion", ascending=False)

    print(f"\n{'Patient':<10} {'Folders':>8} {'Clips':>6} "
          f"{'Mean':>7} {'Med':>7} {'P95':>7} {'%>Thr':>7} {'Band':<10}")
    print("-" * 70)

    for _, r in patient_agg.iterrows():
        print(f"{r['patient_id']:<10} {r['folders']:>8} {r['clips']:>6} "
              f"{r['mean_motion']:>7.2f} {r['median_motion']:>7.2f} "
              f"{r['peak_95']:>7.2f} {r['pct_above_thresh']:>7.1f} "
              f"{r['activity_band']:<10}")

    # --- Cross-patient comparable clips ---
    print("\n" + "=" * 80)
    print("CROSS-PATIENT COMPARABLE CLIPS")
    print("=" * 80)

    for band in ["quiet", "low", "moderate", "high"]:
        band_clips = ok[ok["activity_band"] == band]
        patients_in_band = band_clips["patient_id"].nunique()
        if band_clips.empty:
            continue

        print(f"\n  {band.upper()} activity ({len(band_clips)} clips, "
              f"{patients_in_band} patients):")

        for _, clip in band_clips.sort_values("mean_motion").iterrows():
            uuid_short = clip["uuid"][:8]
            print(f"    {clip['patient_id']:<10} {uuid_short}  "
                  f"avi_idx={clip['avi_index']:>4}  "
                  f"offset={clip['recording_offset_sec']/3600:.1f}hr  "
                  f"mean={clip['mean_motion']:.2f}  "
                  f"area={clip['mean_area_frac']:.3f}")

    # --- Best cross-patient pairs ---
    print("\n" + "=" * 80)
    print("CLOSEST CROSS-PATIENT PAIRS (by mean_motion)")
    print("=" * 80)

    patients = ok["patient_id"].unique()
    pairs = []
    for i, p1 in enumerate(patients):
        for p2 in patients[i + 1:]:
            clips1 = ok[ok["patient_id"] == p1]
            clips2 = ok[ok["patient_id"] == p2]
            for _, c1 in clips1.iterrows():
                for _, c2 in clips2.iterrows():
                    diff = abs(c1["mean_motion"] - c2["mean_motion"])
                    pairs.append({
                        "patient_1": p1,
                        "clip_1": f"{c1['uuid'][:8]}/idx{c1['avi_index']}",
                        "motion_1": c1["mean_motion"],
                        "patient_2": p2,
                        "clip_2": f"{c2['uuid'][:8]}/idx{c2['avi_index']}",
                        "motion_2": c2["mean_motion"],
                        "diff": diff,
                    })

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values("diff").head(15)
        print(f"\n{'Patient1':<10} {'Clip1':<20} {'Motion1':>8} "
              f"{'Patient2':<10} {'Clip2':<20} {'Motion2':>8} {'Diff':>6}")
        print("-" * 90)
        for _, p in pairs_df.iterrows():
            print(f"{p['patient_1']:<10} {p['clip_1']:<20} {p['motion_1']:>8.2f} "
                  f"{p['patient_2']:<10} {p['clip_2']:<20} {p['motion_2']:>8.2f} "
                  f"{p['diff']:>6.2f}")

    print()


# ---------------------------------------------------------------------------
# Review HTML generation
# ---------------------------------------------------------------------------
BAND_COLORS = {
    "quiet": "#e0e0e0",
    "low": "#c8e6c9",
    "moderate": "#fff9c4",
    "high": "#ffcdd2",
}


def _encode_image_base64(path: Path) -> str:
    """Read a JPEG file and return base64-encoded data URI."""
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    except (OSError, IOError):
        return ""


BRIGHTNESS_COLORS = {
    "dark": "#424242",
    "dim": "#8d6e63",
    "normal": "#66bb6a",
    "bright": "#fdd835",
}
BRIGHTNESS_TEXT = {
    "dark": "#fff",
    "dim": "#fff",
    "normal": "#fff",
    "bright": "#333",
}


def generate_review_html(df: pd.DataFrame, thumbs_dir: Path, output_html: Path):
    """Build a self-contained HTML with base64-embedded thumbnails for review."""
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        output_html.write_text(
            "<html><body><h1>No successful clips to review.</h1></body></html>",
            encoding="utf-8",
        )
        return

    ok["activity_band"] = ok["mean_motion"].apply(classify_activity)
    ok["offset_hr"] = ok["recording_offset_sec"] / 3600
    ok["uuid_short"] = ok["uuid"].str[:10]

    has_brightness = "brightness_bin" in ok.columns and ok["brightness_bin"].notna().any()
    if has_brightness:
        ok["brightness_bin"] = ok["brightness_bin"].fillna("unknown")
    else:
        ok["brightness_bin"] = "unknown"
        ok["mean_brightness"] = float("nan")

    has_grayscale = "is_grayscale" in ok.columns and ok["is_grayscale"].notna().any()
    if has_grayscale:
        ok["is_grayscale"] = ok["is_grayscale"].fillna(False).astype(bool)
    else:
        ok["is_grayscale"] = False
    if "mean_saturation" not in ok.columns:
        ok["mean_saturation"] = float("nan")

    # Build thumbnail lookup: filename stem -> base64
    thumb_lookup = {}
    if thumbs_dir.exists():
        for jpg in thumbs_dir.glob("*.jpg"):
            thumb_lookup[jpg.stem] = _encode_image_base64(jpg)

    # Group by patient, then by uuid, sorted by offset
    patients = sorted(ok["patient_id"].unique())
    total_clips = len(ok)

    patient_sections = []
    for pid in patients:
        pat_df = ok[ok["patient_id"] == pid].sort_values("recording_offset_sec")
        folders = pat_df.groupby("uuid", sort=False)

        folder_blocks = []
        for uuid, fgroup in folders:
            uuid_short = uuid[:10]
            cards = []
            for _, row in fgroup.iterrows():
                band = row["activity_band"]
                color = BAND_COLORS.get(band, "#fff")
                b_bin = row["brightness_bin"]
                b_color = BRIGHTNESS_COLORS.get(b_bin, "#999")
                b_text = BRIGHTNESS_TEXT.get(b_bin, "#fff")
                is_gs = bool(row.get("is_grayscale", False))
                gs_str = "grayscale" if is_gs else "color"

                brightness_val = row.get("mean_brightness", float("nan"))
                sat_val = row.get("mean_saturation", float("nan"))
                brightness_str = f"{brightness_val:.0f}" if brightness_val == brightness_val else "?"
                sat_str = f"{sat_val:.0f}" if sat_val == sat_val else "?"

                # Thumbnail lookup
                thumb_key = f"{row['patient_id']}_{uuid_short}_{int(row['avi_index']):04d}"
                b64 = thumb_lookup.get(thumb_key, "")

                if b64:
                    img_tag = f'<img src="{b64}" style="width:100%;border-radius:4px;">'
                else:
                    img_tag = ('<div style="width:100%;height:120px;background:#f0f0f0;'
                               'display:flex;align-items:center;justify-content:center;'
                               'color:#aaa;border-radius:4px;font-size:12px;">no thumb</div>')

                brightness_badge = (
                    f'<span style="display:inline-block;padding:1px 6px;border-radius:3px;'
                    f'background:{b_color};color:{b_text};font-size:10px;font-weight:600;">'
                    f'{b_bin} ({brightness_str})</span>'
                )

                if is_gs:
                    gs_badge = ('<span style="display:inline-block;padding:1px 6px;'
                                'border-radius:3px;background:#263238;color:#b0bec5;'
                                'font-size:10px;font-weight:600;margin-left:3px;">'
                                f'IR/night (sat={sat_str})</span>')
                else:
                    gs_badge = (f'<span style="display:inline-block;padding:1px 6px;'
                                f'border-radius:3px;background:#e8f5e9;color:#2e7d32;'
                                f'font-size:10px;font-weight:600;margin-left:3px;">'
                                f'color (sat={sat_str})</span>')

                # Dim the card border if grayscale
                border_style = "border:2px solid #546e7a;" if is_gs else "border:1px solid #ccc;"

                cards.append(f"""
          <div class="clip-card" data-brightness="{b_bin}" data-activity="{band}"
               data-grayscale="{gs_str}" data-patient="{row['patient_id']}"
               style="background:{color};{border_style}border-radius:6px;
                      padding:8px;width:240px;flex-shrink:0;">
            {img_tag}
            <div style="margin-top:6px;font-size:11px;line-height:1.5;">
              <b>{row['patient_id']}</b> &middot; {uuid_short}<br>
              AVI {int(row['avi_index']):04d} &middot; {row['offset_hr']:.1f}h<br>
              mean={row['mean_motion']:.2f} &middot;
              <span style="font-weight:600;">{band}</span><br>
              {brightness_badge} {gs_badge}
            </div>
          </div>""")

            folder_blocks.append(f"""
        <div style="margin-bottom:16px;">
          <div style="font-size:12px;color:#666;margin-bottom:6px;font-family:monospace;">
            {uuid_short}... ({len(fgroup)} clips)
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:10px;">
            {''.join(cards)}
          </div>
        </div>""")

        pat_clips = len(pat_df)
        patient_sections.append(f"""
      <div class="patient-section" style="margin-bottom:32px;" id="pat-{pid}">
        <h2 style="font-size:18px;border-bottom:2px solid #1a1a2e;padding-bottom:4px;
                   margin-bottom:12px;">{pid}
          <span style="font-size:13px;color:#777;font-weight:normal;">
            &mdash; {pat_clips} clips, {pat_df['uuid'].nunique()} folders
          </span>
        </h2>
        {''.join(folder_blocks)}
      </div>""")

    # Legend
    activity_legend = " ".join(
        f'<span style="display:inline-block;padding:3px 10px;border-radius:3px;'
        f'background:{color};font-size:12px;margin-right:4px;">{band}</span>'
        for band, color in BAND_COLORS.items()
    )

    # Brightness filter buttons
    brightness_bins = [b for b in ["dark", "dim", "normal", "bright"]
                       if b in ok["brightness_bin"].values]
    brightness_buttons = " ".join(
        f'<button class="filter-btn brightness-btn" data-filter="{b}"'
        f' style="padding:4px 12px;border:2px solid {BRIGHTNESS_COLORS[b]};'
        f'border-radius:4px;background:{BRIGHTNESS_COLORS[b]};'
        f'color:{BRIGHTNESS_TEXT[b]};font-size:12px;font-weight:600;'
        f'cursor:pointer;opacity:1;" onclick="toggleFilter(\'brightness\',\'{b}\')">'
        f'{b}</button>'
        for b in brightness_bins
    )

    # Activity filter buttons
    activity_bins = [b for b in ["quiet", "low", "moderate", "high"]
                     if b in ok["activity_band"].values]
    activity_buttons = " ".join(
        f'<button class="filter-btn activity-btn" data-filter="{b}"'
        f' style="padding:4px 12px;border:2px solid {BAND_COLORS[b]};'
        f'border-radius:4px;background:{BAND_COLORS[b]};'
        f'color:#333;font-size:12px;font-weight:600;'
        f'cursor:pointer;opacity:1;" onclick="toggleFilter(\'activity\',\'{b}\')">'
        f'{b}</button>'
        for b in activity_bins
    )

    # Grayscale filter buttons
    gs_count = int(ok["is_grayscale"].sum())
    color_count = len(ok) - gs_count
    grayscale_buttons = (
        f'<button class="filter-btn grayscale-btn" data-filter="color"'
        f' style="padding:4px 12px;border:2px solid #4caf50;border-radius:4px;'
        f'background:#e8f5e9;color:#2e7d32;font-size:12px;font-weight:600;'
        f'cursor:pointer;" onclick="toggleFilter(\'grayscale\',\'color\')">'
        f'color ({color_count})</button> '
        f'<button class="filter-btn grayscale-btn" data-filter="grayscale"'
        f' style="padding:4px 12px;border:2px solid #546e7a;border-radius:4px;'
        f'background:#263238;color:#b0bec5;font-size:12px;font-weight:600;'
        f'cursor:pointer;" onclick="toggleFilter(\'grayscale\',\'grayscale\')">'
        f'IR/night ({gs_count})</button>'
    )

    # Header summary
    if has_brightness:
        bin_counts = ok["brightness_bin"].value_counts()
        dist_str = " &middot; ".join(
            f"{b}: {bin_counts.get(b, 0)}"
            for b in ["dark", "dim", "normal", "bright"]
            if bin_counts.get(b, 0) > 0
        )
        brightness_summary = (
            f"<div style='font-size:12px;color:#999;margin-top:2px;'>"
            f"Brightness: {dist_str} &middot; "
            f"color: {color_count}, grayscale/IR: {gs_count}</div>"
        )
    else:
        brightness_summary = "<div style='font-size:12px;color:#c62828;margin-top:2px;'>No brightness data. Run --tag-brightness first.</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Movement Screening Review</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.4;
    padding: 16px 24px;
  }}
  .filter-bar {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 10px 16px; margin-bottom: 16px;
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }}
  .filter-group {{ display: flex; align-items: center; gap: 6px; }}
  .filter-label {{ font-size: 12px; color: #666; font-weight: 600; }}
  .filter-btn.inactive {{ opacity: 0.3 !important; }}
  .clip-card.hidden {{ display: none !important; }}
  .match-count {{
    font-size: 13px; color: #555; margin-left: auto;
    font-weight: 600;
  }}
</style>
</head>
<body>
  <div style="margin-bottom:12px;">
    <h1 style="font-size:22px;">Movement Screening Review</h1>
    <div style="font-size:13px;color:#777;margin-top:4px;">
      {len(patients)} patients &middot; {total_clips} clips &middot;
      Activity: {activity_legend}
    </div>
    {brightness_summary}
    <div style="font-size:12px;color:#999;margin-top:4px;">
      Jump to:
      {' '.join(f'<a href="#pat-{p}">{p}</a>' for p in patients)}
    </div>
  </div>

  <div class="filter-bar">
    <div class="filter-group">
      <span class="filter-label">Mode:</span>
      {grayscale_buttons}
    </div>
    <div class="filter-group">
      <span class="filter-label">Brightness:</span>
      {brightness_buttons}
    </div>
    <div class="filter-group">
      <span class="filter-label">Activity:</span>
      {activity_buttons}
    </div>
    <button style="padding:4px 12px;border:1px solid #ccc;border-radius:4px;
                   background:#fff;color:#666;font-size:12px;cursor:pointer;"
            onclick="resetFilters()">Reset</button>
    <span class="match-count" id="match-count">{total_clips} / {total_clips} visible</span>
  </div>

  {''.join(patient_sections)}

<script>
const activeFilters = {{ brightness: new Set(), activity: new Set(), grayscale: new Set() }};
const TOTAL = {total_clips};

function toggleFilter(type, value) {{
  const set = activeFilters[type];
  if (set.has(value)) {{
    set.delete(value);
  }} else {{
    set.add(value);
  }}
  applyFilters();
}}

function resetFilters() {{
  activeFilters.brightness.clear();
  activeFilters.activity.clear();
  activeFilters.grayscale.clear();
  applyFilters();
}}

function applyFilters() {{
  // Update button styles
  ['brightness', 'activity', 'grayscale'].forEach(type => {{
    document.querySelectorAll('.' + type + '-btn').forEach(btn => {{
      const val = btn.dataset.filter;
      btn.classList.toggle('inactive',
        activeFilters[type].size > 0 && !activeFilters[type].has(val));
    }});
  }});

  // Filter cards
  let visible = 0;
  document.querySelectorAll('.clip-card').forEach(card => {{
    const b = card.dataset.brightness;
    const a = card.dataset.activity;
    const g = card.dataset.grayscale;
    const bMatch = activeFilters.brightness.size === 0 || activeFilters.brightness.has(b);
    const aMatch = activeFilters.activity.size === 0 || activeFilters.activity.has(a);
    const gMatch = activeFilters.grayscale.size === 0 || activeFilters.grayscale.has(g);
    const show = bMatch && aMatch && gMatch;
    card.classList.toggle('hidden', !show);
    if (show) visible++;
  }});

  document.getElementById('match-count').textContent = visible + ' / ' + TOTAL + ' visible';
}}
</script>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    size_mb = output_html.stat().st_size / (1024 * 1024)
    print(f"  Wrote {output_html} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Full brightness scan mode
# ---------------------------------------------------------------------------
SCAN_COL_ORDER = [
    "patient_id", "uuid", "folder_name", "avi_file", "avi_index",
    "recording_offset_sec", "mean_brightness", "brightness_bin",
    "mean_saturation", "is_grayscale", "status",
]


def _scan_one_avi(args_tuple):
    """Worker function for parallel brightness scanning."""
    avi_path, timeout = args_tuple
    props = measure_frame_properties_with_timeout(avi_path, timeout)
    if props is None:
        return None
    return props


def _run_scan_all(
    patient_ids: list[str],
    triage_path: Path,
    metadata_path: Path,
    config_path: Path,
    data_root_override: str | None,
    timeout: int,
    workers: int,
):
    """Scan ALL AVIs from usable folders for brightness/grayscale classification."""
    from concurrent.futures import as_completed

    config = load_config(config_path)
    data_root = data_root_override or config.get("data_root", "")
    triage = load_triage(triage_path, patient_ids)
    metadata = load_metadata(metadata_path)

    print(f"Loading data...")
    print(f"  Patients: {patient_ids}")
    print(f"  Data root: {data_root}")
    print(f"  Usable folders: {len(triage)}")
    print(f"  Workers: {workers}")
    print(f"  Timeout: {timeout}s per AVI")

    # Build full AVI list from triage + metadata
    tasks = build_folder_tasks(triage, metadata)
    SCREENING_DIR.mkdir(parents=True, exist_ok=True)

    # Group by patient
    tasks_by_patient = {}
    for t in tasks:
        tasks_by_patient.setdefault(t["patient_id"], []).append(t)

    grand_t0 = time.time()

    for pid_idx, pid in enumerate(patient_ids, 1):
        scan_csv = SCREENING_DIR / f"brightness_scan_{pid}.csv"

        # Load existing scan for resume
        existing_keys = set()
        if scan_csv.exists():
            existing_df = pd.read_csv(scan_csv)
            existing_keys = set(
                existing_df["patient_id"] + "|" + existing_df["uuid"]
                + "|" + existing_df["avi_index"].astype(str)
            )
            existing_rows = existing_df.to_dict("records")
        else:
            existing_rows = []

        patient_tasks = tasks_by_patient.get(pid, [])
        if not patient_tasks:
            print(f"\n[{pid_idx}/{len(patient_ids)}] {pid} — no usable folders")
            continue

        # Build work items for this patient (all AVIs, not sampled)
        work_items = []
        for task in patient_tasks:
            folder_path = Path(data_root) / task["folder_name"]
            for avi in task["avi_list"]:
                key = f"{pid}|{task['uuid']}|{avi['avi_index']}"
                if key in existing_keys:
                    continue
                work_items.append({
                    "patient_id": pid,
                    "uuid": task["uuid"],
                    "folder_name": task["folder_name"],
                    "avi_file": avi["file_name"],
                    "avi_index": avi["avi_index"],
                    "recording_offset_sec": avi["start_time"],
                    "avi_path": folder_path / avi["file_name"],
                })

        total_for_patient = sum(len(t["avi_list"]) for t in patient_tasks)
        already_done = total_for_patient - len(work_items)

        print(f"\n{'='*70}")
        print(f"[{pid_idx}/{len(patient_ids)}] {pid} — "
              f"{total_for_patient} AVIs total, {already_done} already scanned, "
              f"{len(work_items)} to scan")
        print("=" * 70)

        if not work_items:
            continue

        t0 = time.time()
        new_rows = []
        done = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(measure_frame_properties, item["avi_path"]): item
                for item in work_items
            }

            for future in as_completed(future_to_item):
                item = future_to_item[future]
                done += 1

                try:
                    props = future.result(timeout=timeout)
                except Exception:
                    props = None

                row = {
                    "patient_id": item["patient_id"],
                    "uuid": item["uuid"],
                    "folder_name": item["folder_name"],
                    "avi_file": item["avi_file"],
                    "avi_index": item["avi_index"],
                    "recording_offset_sec": item["recording_offset_sec"],
                }

                if props is not None:
                    row["mean_brightness"] = props["mean_brightness"]
                    row["brightness_bin"] = classify_brightness(props["mean_brightness"])
                    row["mean_saturation"] = props["mean_saturation"]
                    row["is_grayscale"] = props["is_grayscale"]
                    row["status"] = "ok"
                else:
                    row["status"] = "failed"
                    failed += 1

                new_rows.append(row)

                if done % 100 == 0 or done == len(work_items):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    gs_so_far = sum(1 for r in new_rows if r.get("is_grayscale"))
                    print(f"  {done}/{len(work_items)}  "
                          f"({elapsed:.0f}s, {rate:.1f}/s, "
                          f"{failed} failed, {gs_so_far} grayscale)")

                    # Incremental save every 100
                    all_rows = existing_rows + new_rows
                    scan_df = pd.DataFrame(all_rows)
                    cols = [c for c in SCAN_COL_ORDER if c in scan_df.columns]
                    remaining = [c for c in scan_df.columns if c not in cols]
                    scan_df = scan_df[cols + remaining]
                    scan_df.sort_values(["uuid", "avi_index"], inplace=True)
                    scan_df.to_csv(scan_csv, index=False)

        elapsed = time.time() - t0
        ok_count = sum(1 for r in new_rows if r.get("status") == "ok")
        gs_count = sum(1 for r in new_rows if r.get("is_grayscale"))
        print(f"\n  {pid} done: {ok_count} ok, {failed} failed, "
              f"{gs_count} grayscale, {elapsed:.1f}s")
        print(f"  Saved: {scan_csv}")

    # Merge all per-patient scan CSVs
    print(f"\n{'='*70}")
    print("MERGING SCAN RESULTS")
    print("=" * 70)

    per_patient_csvs = sorted(SCREENING_DIR.glob("brightness_scan_EM*.csv"))
    if not per_patient_csvs:
        print("No scan CSVs found.")
        return

    parts = []
    for csv_path in per_patient_csvs:
        pdf = pd.read_csv(csv_path)
        pid = csv_path.stem.replace("brightness_scan_", "")
        ok_n = (pdf["status"] == "ok").sum()
        gs_n = pdf["is_grayscale"].sum() if "is_grayscale" in pdf.columns else 0
        print(f"  {pid}: {len(pdf)} AVIs ({ok_n} ok, {int(gs_n)} grayscale)")
        parts.append(pdf)

    combined = pd.concat(parts, ignore_index=True)
    combined.sort_values(["patient_id", "uuid", "avi_index"], inplace=True)
    out_csv = SCREENING_DIR / "brightness_scan.csv"
    combined.to_csv(out_csv, index=False)

    total_ok = (combined["status"] == "ok").sum()
    total_gs = combined["is_grayscale"].sum() if "is_grayscale" in combined.columns else 0
    total_color = total_ok - total_gs
    grand_elapsed = time.time() - grand_t0
    print(f"\n  Combined: {out_csv}")
    print(f"  Total: {len(combined)} AVIs, {total_ok} ok")
    print(f"  Color: {total_color}, Grayscale/IR: {int(total_gs)}")
    print(f"\nALL DONE in {grand_elapsed:.1f}s")

    # Per-patient + per-folder breakdown
    ok = combined[combined["status"] == "ok"]
    if "is_grayscale" in ok.columns:
        print(f"\n{'='*70}")
        print("PER-FOLDER BREAKDOWN")
        print("=" * 70)
        for pid in patient_ids:
            pat = ok[ok["patient_id"] == pid]
            if pat.empty:
                continue
            print(f"\n  {pid}:")
            for uuid, grp in pat.groupby("uuid"):
                gs = grp["is_grayscale"].sum()
                col = len(grp) - gs
                pct_gs = gs / len(grp) * 100
                print(f"    {uuid[:10]}...  {len(grp):>4} AVIs  "
                      f"color: {col:>4}  gray: {int(gs):>4}  ({pct_gs:.0f}% night)")
        print()


# ---------------------------------------------------------------------------
# Brightness tagging mode
# ---------------------------------------------------------------------------
def _run_tag_brightness(
    patient_ids: list[str],
    config_path: Path,
    data_root_override: str | None,
    timeout: int,
):
    """Retroactively measure brightness for existing screening CSVs."""
    config = load_config(config_path)
    data_root = data_root_override or config.get("data_root", "")

    for pid in patient_ids:
        csv_path = SCREENING_DIR / f"screening_{pid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {csv_path.name} not found, skipping", file=sys.stderr)
            continue

        df = pd.read_csv(csv_path)

        # Initialize columns if missing
        for col, default in [("mean_brightness", np.nan), ("brightness_bin", ""),
                             ("mean_saturation", np.nan), ("is_grayscale", None)]:
            if col not in df.columns:
                df[col] = default

        ok_mask = df["status"] == "ok"
        # Need tagging if brightness or saturation is missing
        needs_tag = ok_mask & (df["mean_brightness"].isna() | df["mean_saturation"].isna())
        n_todo = needs_tag.sum()

        if n_todo == 0:
            print(f"  {pid}: already tagged ({len(df)} rows), skipping")
            continue

        print(f"\n  {pid}: {n_todo} clips to measure")

        t0 = time.time()
        measured = 0
        failed = 0

        for idx in df.index[needs_tag]:
            row = df.loc[idx]
            avi_path = Path(data_root) / row["folder_name"] / row["avi_file"]
            measured += 1
            print(f"    [{measured}/{n_todo}] avi_idx {int(row['avi_index']):>4}...",
                  end=" ", flush=True)

            props = measure_frame_properties_with_timeout(avi_path, timeout)
            if props is not None:
                df.at[idx, "mean_brightness"] = props["mean_brightness"]
                df.at[idx, "brightness_bin"] = classify_brightness(props["mean_brightness"])
                df.at[idx, "mean_saturation"] = props["mean_saturation"]
                df.at[idx, "is_grayscale"] = props["is_grayscale"]
                gs_label = "GRAY" if props["is_grayscale"] else "color"
                print(f"bright={props['mean_brightness']:.0f} "
                      f"sat={props['mean_saturation']:.0f} "
                      f"({classify_brightness(props['mean_brightness'])}, {gs_label})")
            else:
                failed += 1
                print("FAILED")

        elapsed = time.time() - t0
        print(f"  {pid}: {measured - failed} measured, {failed} failed, {elapsed:.1f}s")

        # Save updated CSV
        df = _reorder_columns(df)
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Print distribution summary
    print(f"\n{'='*60}")
    print("BRIGHTNESS & GRAYSCALE DISTRIBUTION")
    print("=" * 60)
    all_parts = []
    for pid in patient_ids:
        csv_path = SCREENING_DIR / f"screening_{pid}.csv"
        if csv_path.exists():
            all_parts.append(pd.read_csv(csv_path))
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        ok = combined[combined["status"] == "ok"]
        if "brightness_bin" in ok.columns:
            for pid in patient_ids:
                pat = ok[ok["patient_id"] == pid]
                if pat.empty:
                    continue
                b_counts = pat["brightness_bin"].value_counts()
                gs_count = pat["is_grayscale"].sum() if "is_grayscale" in pat.columns else 0
                color_count = len(pat) - gs_count
                parts_str = ", ".join(f"{k}: {v}" for k, v in sorted(b_counts.items()))
                print(f"  {pid}: {parts_str}  |  color: {color_count}, grayscale: {int(gs_count)}")
            print()
            total_b = ok["brightness_bin"].value_counts()
            total_gs = ok["is_grayscale"].sum() if "is_grayscale" in ok.columns else 0
            total_color = len(ok) - total_gs
            print(f"  TOTAL: {', '.join(f'{k}: {v}' for k, v in sorted(total_b.items()))}")
            print(f"  TOTAL: color: {total_color}, grayscale: {int(total_gs)}")
        print()


# ---------------------------------------------------------------------------
# Thumbnail mode
# ---------------------------------------------------------------------------
def _run_thumbnails(
    patient_ids: list[str],
    config_path: Path,
    data_root_override: str | None,
    timeout: int,
):
    """Load existing screening CSVs, extract one thumbnail per ok clip, build HTML."""
    config = load_config(config_path)
    data_root = data_root_override or config.get("data_root", "")

    # Load per-patient CSVs
    parts = []
    for pid in patient_ids:
        csv_path = SCREENING_DIR / f"screening_{pid}.csv"
        if csv_path.exists():
            pdf = pd.read_csv(csv_path)
            parts.append(pdf)
            print(f"  Loaded {csv_path.name}: {len(pdf)} rows")
        else:
            print(f"  WARNING: {csv_path.name} not found, skipping", file=sys.stderr)

    if not parts:
        print("No screening CSVs found. Run screening first.", file=sys.stderr)
        return

    df = pd.concat(parts, ignore_index=True)
    ok = df[df["status"] == "ok"]
    print(f"\n  Total clips: {len(df)}, ok: {len(ok)}")

    # Extract thumbnails
    thumbs_dir = SCREENING_DIR / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    extracted = 0
    skipped = 0
    failed = 0

    for i, (_, row) in enumerate(ok.iterrows(), 1):
        pid = row["patient_id"]
        uuid_short = row["uuid"][:10]
        avi_index = int(row["avi_index"])
        thumb_name = f"{pid}_{uuid_short}_{avi_index:04d}.jpg"
        thumb_path = thumbs_dir / thumb_name

        # Skip existing thumbnails
        if thumb_path.exists():
            skipped += 1
            continue

        avi_path = Path(data_root) / row["folder_name"] / row["avi_file"]
        print(f"  [{i}/{len(ok)}] {thumb_name}...", end=" ", flush=True)

        success = extract_thumbnail_with_timeout(avi_path, thumb_path, timeout)
        if success:
            extracted += 1
            print("ok")
        else:
            failed += 1
            print("FAILED")

    elapsed = time.time() - t0
    print(f"\n  Thumbnails: {extracted} extracted, {skipped} skipped (existing), "
          f"{failed} failed, {elapsed:.1f}s")

    # Generate review HTML
    print("\nGenerating review HTML...")
    output_html = SCREENING_DIR / "screening_review.html"
    generate_review_html(df, thumbs_dir, output_html)
    print("Done. Open screening_review.html in a browser to review.")


# ---------------------------------------------------------------------------
# Copy list builder
# ---------------------------------------------------------------------------
DEFAULT_ACTIVE_THRESHOLD = 0.8
DEFAULT_BUFFER_HR = 1.5
DEFAULT_DENSE_NTH = 3
DEFAULT_SPARSE_NTH = 15

COPY_LIST_DIR = SCREENING_DIR / "copy_list"


def _build_active_windows(
    motion_df: pd.DataFrame, threshold: float, buffer_hr: float,
) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """From sampled motion data, identify time windows with activity."""
    windows = {}
    for (pid, uuid), grp in motion_df.groupby(["patient_id", "uuid"]):
        active = grp[grp["mean_motion"] >= threshold]
        raw = [
            (r["recording_offset_sec"] / 3600 - buffer_hr,
             r["recording_offset_sec"] / 3600 + buffer_hr)
            for _, r in active.iterrows()
        ]
        if raw:
            raw.sort()
            merged = [raw[0]]
            for s, e in raw[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            windows[(pid, uuid)] = merged
        else:
            windows[(pid, uuid)] = []
    return windows


def _filter_copy_list(
    scan_df: pd.DataFrame,
    motion_df: pd.DataFrame,
    threshold: float,
    buffer_hr: float,
    dense_nth: int,
    sparse_nth: int,
) -> pd.DataFrame:
    """Build a filtered copy list from the brightness scan + motion data."""
    color = scan_df[scan_df["is_grayscale"] == False].copy()
    color["offset_hr"] = color["recording_offset_sec"] / 3600

    # Only use color motion samples for window building
    color_motion = motion_df.copy()
    if "is_grayscale" in color_motion.columns:
        color_motion = color_motion[color_motion["is_grayscale"] == False]
    color_motion["offset_hr"] = color_motion["recording_offset_sec"] / 3600

    active_windows = _build_active_windows(color_motion, threshold, buffer_hr)

    def in_active(row):
        ws = active_windows.get((row["patient_id"], row["uuid"]), [])
        return any(s <= row["offset_hr"] <= e for s, e in ws)

    color["in_active_window"] = color.apply(in_active, axis=1)

    parts = []
    for (pid, uuid), grp in color.groupby(["patient_id", "uuid"]):
        grp = grp.sort_values("avi_index")
        act = grp[grp["in_active_window"]]
        quiet = grp[~grp["in_active_window"]]
        if len(act) > 0:
            parts.append(act.iloc[::dense_nth])
        if len(quiet) > 0:
            parts.append(quiet.iloc[::sparse_nth])

    selected = pd.concat(parts).sort_values(
        ["patient_id", "uuid", "avi_index"]
    )
    return selected, active_windows


def _run_build_copy_list(
    patient_ids: list[str],
    config_path: Path,
    metadata_path: Path,
    data_root_override: str | None,
    timeout: int,
    workers: int,
):
    """Build filtered copy list, extract thumbnails, generate review HTML."""
    from concurrent.futures import as_completed

    config = load_config(config_path)
    data_root = data_root_override or config.get("data_root", "")
    meta = load_metadata(metadata_path)

    # Load brightness scan
    scan_csv = SCREENING_DIR / "brightness_scan.csv"
    if not scan_csv.exists():
        print("ERROR: brightness_scan.csv not found. Run --scan-all first.",
              file=sys.stderr)
        return
    scan = pd.read_csv(scan_csv)
    scan = scan[scan["patient_id"].isin(patient_ids)]

    # Load motion screening
    screening_csv = SCREENING_DIR / "screening_summary.csv"
    if not screening_csv.exists():
        print("ERROR: screening_summary.csv not found. Run screening first.",
              file=sys.stderr)
        return
    screening = pd.read_csv(screening_csv)
    screening = screening[
        (screening["status"] == "ok")
        & screening["patient_id"].isin(patient_ids)
    ]

    # Add grayscale info to screening
    screening = screening.merge(
        scan[["patient_id", "uuid", "avi_index", "is_grayscale"]],
        on=["patient_id", "uuid", "avi_index"], how="left",
    )

    # Build filtered copy list
    print("Building filtered copy list...")
    print(f"  Active threshold: {DEFAULT_ACTIVE_THRESHOLD}")
    print(f"  Buffer: +/- {DEFAULT_BUFFER_HR} hr")
    print(f"  Dense sampling: every {DEFAULT_DENSE_NTH}rd in active windows")
    print(f"  Sparse sampling: every {DEFAULT_SPARSE_NTH}th in quiet periods")

    selected, active_windows = _filter_copy_list(
        scan, screening,
        DEFAULT_ACTIVE_THRESHOLD, DEFAULT_BUFFER_HR,
        DEFAULT_DENSE_NTH, DEFAULT_SPARSE_NTH,
    )

    # Add file sizes and source paths
    selected = selected.merge(
        meta[["patient_id", "uuid", "avi_index", "file_size_bytes"]],
        on=["patient_id", "uuid", "avi_index"], how="left",
    )
    selected["source_path"] = selected.apply(
        lambda r: str(Path(data_root) / r["folder_name"] / r["avi_file"]),
        axis=1,
    )

    # Print active windows
    print(f"\nActive windows:")
    for (pid, uuid), wins in sorted(active_windows.items()):
        if wins:
            w_str = ", ".join(f"{s:.1f}-{e:.1f}hr" for s, e in wins)
        else:
            w_str = "(quiet — sparse only)"
        print(f"  {pid} {uuid[:10]}: {w_str}")

    # Summary
    total_gb = selected["file_size_bytes"].sum() / 1e9
    n_act = selected["in_active_window"].sum()
    print(f"\nCopy list: {len(selected)} AVIs, {total_gb:.1f} GB")
    print(f"  Active: {n_act}  |  Quiet baseline: {len(selected) - n_act}")
    for pid in sorted(selected["patient_id"].unique()):
        pat = selected[selected["patient_id"] == pid]
        gb = pat["file_size_bytes"].sum() / 1e9
        act = pat["in_active_window"].sum()
        print(f"  {pid}: {len(pat)} AVIs ({act} active, "
              f"{len(pat)-act} quiet) {gb:.1f} GB")

    # Save copy list CSV
    COPY_LIST_DIR.mkdir(parents=True, exist_ok=True)
    copy_csv = COPY_LIST_DIR / "copy_list.csv"
    selected.to_csv(copy_csv, index=False)
    print(f"\n  Saved: {copy_csv}")

    # Extract thumbnails
    thumbs_dir = COPY_LIST_DIR / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting thumbnails ({workers} workers)...")
    t0 = time.time()
    extracted = 0
    skipped = 0
    failed = 0

    # Build work items
    work = []
    for _, row in selected.iterrows():
        uuid_short = row["uuid"][:10]
        thumb_name = f"{row['patient_id']}_{uuid_short}_{int(row['avi_index']):04d}.jpg"
        thumb_path = thumbs_dir / thumb_name
        if thumb_path.exists():
            skipped += 1
            continue
        work.append({
            "avi_path": Path(row["source_path"]),
            "thumb_path": thumb_path,
        })

    if work:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(extract_thumbnail, item["avi_path"], item["thumb_path"]): item
                for item in work
            }
            done = 0
            for future in as_completed(future_to_item):
                done += 1
                try:
                    success = future.result(timeout=timeout)
                except Exception:
                    success = False

                if success:
                    extracted += 1
                else:
                    failed += 1

                if done % 50 == 0 or done == len(work):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  {done}/{len(work)} ({elapsed:.0f}s, "
                          f"{rate:.1f}/s, {failed} failed)")

    elapsed = time.time() - t0
    print(f"\n  Thumbnails: {extracted} new, {skipped} existing, "
          f"{failed} failed, {elapsed:.1f}s")

    # Generate review HTML
    print("\nGenerating review HTML...")
    # Add activity_band for the HTML (approximate from screening data where available)
    # For AVIs without direct motion data, label based on window membership
    if "mean_motion" not in selected.columns:
        selected["mean_motion"] = float("nan")

    # Merge motion data where we have it
    motion_cols = screening[["patient_id", "uuid", "avi_index", "mean_motion"]].copy()
    selected = selected.merge(motion_cols, on=["patient_id", "uuid", "avi_index"],
                              how="left", suffixes=("", "_screen"))
    if "mean_motion_screen" in selected.columns:
        selected["mean_motion"] = selected["mean_motion_screen"].combine_first(
            selected["mean_motion"]
        )
        selected.drop(columns=["mean_motion_screen"], inplace=True)

    # For AVIs without motion data, use window membership as proxy
    no_motion = selected["mean_motion"].isna()
    selected.loc[no_motion & selected["in_active_window"], "mean_motion"] = 1.0
    selected.loc[no_motion & ~selected["in_active_window"], "mean_motion"] = 0.5

    # Set status=ok for HTML generation
    selected["status"] = "ok"

    output_html = COPY_LIST_DIR / "copy_list_review.html"
    generate_review_html(selected, thumbs_dir, output_html)
    print(f"\nDone. Open copy_list_review.html in a browser to review before copying.")


# ---------------------------------------------------------------------------
# Matched contiguous-window review
# ---------------------------------------------------------------------------
MATCH_WINDOW_HR = 2.0
MATCH_STEP_HR = 1.0
MATCH_MIN_CLIPS = 3  # minimum clips per window to be considered


def _compute_matched_windows(
    df: pd.DataFrame,
    n_sets: int = 10,
    window_hr: float = MATCH_WINDOW_HR,
    step_hr: float = MATCH_STEP_HR,
    min_clips: int = MATCH_MIN_CLIPS,
) -> list[dict]:
    """Find top N matched contiguous time windows across patients.

    For each patient+folder, builds sliding 2-hour windows and computes
    mean brightness/saturation. Then finds cross-patient combinations that
    minimize the max pairwise Euclidean distance in normalized space.

    Returns list of dicts with: set_idx, max_dist, avg_bright, avg_sat, windows.
    Each window contains: patient_id, uuid, uuid_short, start_hr, end_hr,
    avi_min, avi_max, n_clips, mean_bright, mean_sat, clip_rows (list of row dicts).
    """
    from itertools import product

    patients = sorted(df["patient_id"].unique())
    if len(patients) < 2:
        print("  Need at least 2 patients for matching.")
        return []

    # Ensure offset_hr
    if "offset_hr" not in df.columns and "recording_offset_sec" in df.columns:
        df = df.copy()
        df["offset_hr"] = df["recording_offset_sec"] / 3600

    # Build per-patient sliding windows
    all_windows = []
    for pid in patients:
        pat = df[df["patient_id"] == pid]
        for uuid, g in pat.groupby("uuid"):
            g = g.sort_values("offset_hr")
            max_hr = g["offset_hr"].max()
            start = 0.0
            while start + window_hr <= max_hr + 0.5:
                w = g[(g["offset_hr"] >= start) & (g["offset_hr"] < start + window_hr)]
                if len(w) >= min_clips:
                    all_windows.append({
                        "patient_id": pid,
                        "uuid": uuid,
                        "uuid_short": uuid[:10],
                        "start_hr": start,
                        "end_hr": start + window_hr,
                        "n_clips": len(w),
                        "mean_bright": w["mean_brightness"].mean(),
                        "std_bright": w["mean_brightness"].std(),
                        "mean_sat": w["mean_saturation"].mean(),
                        "std_sat": w["mean_saturation"].std(),
                        "avi_min": int(w["avi_index"].min()),
                        "avi_max": int(w["avi_index"].max()),
                        "clip_rows": w.to_dict("records"),
                    })
                start += step_hr

    if not all_windows:
        return []

    wdf = pd.DataFrame([{k: v for k, v in w.items() if k != "clip_rows"}
                         for w in all_windows])

    # Normalize brightness and saturation to [0, 1]
    bright_min = wdf["mean_bright"].min()
    bright_range = max(wdf["mean_bright"].max() - bright_min, 1)
    sat_min = wdf["mean_sat"].min()
    sat_range = max(wdf["mean_sat"].max() - sat_min, 1)

    for w in all_windows:
        w["norm_b"] = (w["mean_bright"] - bright_min) / bright_range
        w["norm_s"] = (w["mean_sat"] - sat_min) / sat_range

    # Group by patient, keep top 15 by n_clips to limit combinations
    per_patient = {}
    for pid in patients:
        pw = [w for w in all_windows if w["patient_id"] == pid]
        if len(pw) > 15:
            pw = sorted(pw, key=lambda x: -x["n_clips"])[:15]
        per_patient[pid] = pw

    # Find best combinations
    best = []
    for combo in product(*[per_patient[p] for p in patients]):
        points = [(e["norm_b"], e["norm_s"]) for e in combo]
        max_dist = 0
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                d = ((points[a][0] - points[b][0])**2
                     + (points[a][1] - points[b][1])**2) ** 0.5
                max_dist = max(max_dist, d)

        best.append({
            "max_dist": max_dist,
            "avg_bright": np.mean([e["mean_bright"] for e in combo]),
            "avg_sat": np.mean([e["mean_sat"] for e in combo]),
            "total_clips": sum(e["n_clips"] for e in combo),
            "windows": list(combo),
        })

    best.sort(key=lambda x: x["max_dist"])

    # De-duplicate: skip sets that reuse the same window for 3+ patients
    seen = set()
    result = []
    for s in best:
        # Key = tuple of (pid, uuid, start_hr) for each window
        key = tuple(
            (w["patient_id"], w["uuid"], w["start_hr"]) for w in s["windows"]
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
        if len(result) >= n_sets:
            break

    # Add set indices
    for i, s in enumerate(result):
        s["set_idx"] = i + 1

    return result


def generate_matched_review_html(
    matched_sets: list[dict],
    df: pd.DataFrame,
    thumbs_dir: Path,
    output_html: Path,
):
    """Generate self-contained HTML showing matched contiguous windows.

    Each set shows one column per patient, with all clips in that patient's
    2-hour window displayed as thumbnail strips.
    """
    # Build thumbnail lookup
    thumb_lookup = {}
    if thumbs_dir.exists():
        for jpg in thumbs_dir.glob("*.jpg"):
            thumb_lookup[jpg.stem] = _encode_image_base64(jpg)

    patients = sorted(df["patient_id"].unique())
    n_patients = len(patients)

    # Build set sections
    set_sections = []
    for ms in matched_sets:
        # One column per patient window
        columns_html = []
        for win in ms["windows"]:
            pid = win["patient_id"]
            uuid_short = win["uuid_short"]

            # Build thumbnail strip for all clips in this window
            clip_cards = []
            clips = sorted(win["clip_rows"], key=lambda r: r.get("avi_index", 0))
            for clip in clips:
                avi_idx = int(clip.get("avi_index", 0))
                thumb_key = "%s_%s_%04d" % (pid, uuid_short, avi_idx)
                b64 = thumb_lookup.get(thumb_key, "")

                if b64:
                    img_tag = ('<img src="%s" style="width:100%%;'
                               'border-radius:3px;">' % b64)
                else:
                    img_tag = ('<div style="width:100%%;height:60px;background:#f0f0f0;'
                               'display:flex;align-items:center;justify-content:center;'
                               'color:#bbb;border-radius:3px;font-size:9px;">—</div>')

                c_bright = clip.get("mean_brightness", 0)
                c_sat = clip.get("mean_saturation", 0)
                c_hr = clip.get("offset_hr",
                                clip.get("recording_offset_sec", 0) / 3600)

                clip_cards.append(
                    '<div style="margin-bottom:4px;">'
                    '%s'
                    '<div style="font-size:9px;color:#888;margin-top:2px;">'
                    'AVI %04d &middot; %.1fh &middot; b=%d s=%d'
                    '</div>'
                    '</div>' % (img_tag, avi_idx, c_hr, c_bright, c_sat)
                )

            b_color = BRIGHTNESS_COLORS.get(
                win.get("brightness_bin", "normal"), "#66bb6a")

            columns_html.append(
                '<div style="flex:1;min-width:220px;max-width:320px;">'
                '<div style="background:#1565c0;color:#fff;padding:6px 10px;'
                'border-radius:6px 6px 0 0;font-weight:700;font-size:14px;">'
                '%s'
                '</div>'
                '<div style="background:#fff;border:1px solid #ddd;'
                'border-top:none;border-radius:0 0 6px 6px;padding:8px;">'
                '<div style="font-family:monospace;font-size:10px;color:#666;'
                'margin-bottom:6px;">'
                '%s... &middot; AVI %04d&ndash;%04d<br>'
                '%.0f&ndash;%.0fh &middot; %d clips<br>'
                'bright=%.0f&plusmn;%.0f &middot; sat=%.0f&plusmn;%.0f'
                '</div>'
                '<div style="max-height:600px;overflow-y:auto;">'
                '%s'
                '</div>'
                '</div>'
                '</div>' % (
                    pid,
                    uuid_short, win["avi_min"], win["avi_max"],
                    win["start_hr"], win["end_hr"], win["n_clips"],
                    win["mean_bright"], win.get("std_bright", 0),
                    win["mean_sat"], win.get("std_sat", 0),
                    "".join(clip_cards),
                )
            )

        set_sections.append(
            '<div style="margin-bottom:32px;padding:16px;background:#fafafa;'
            'border:1px solid #e0e0e0;border-radius:10px;">'
            '<div style="margin-bottom:12px;">'
            '<span style="font-weight:700;font-size:18px;color:#333;">'
            'Set %d'
            '</span>'
            '<span style="margin-left:14px;font-size:12px;color:#888;">'
            'match distance: %.4f &middot; '
            'avg brightness: %.0f &middot; '
            'avg saturation: %.0f &middot; '
            'total clips: %d'
            '</span>'
            '</div>'
            '<div style="display:flex;gap:14px;flex-wrap:wrap;'
            'align-items:flex-start;">'
            '%s'
            '</div>'
            '</div>' % (
                ms["set_idx"], ms["max_dist"],
                ms["avg_bright"], ms["avg_sat"], ms["total_clips"],
                "".join(columns_html),
            )
        )

    # Per-patient summary table
    patient_summary_rows = []
    for pid in patients:
        pat = df[df["patient_id"] == pid]
        n_uuids = pat["uuid"].nunique()
        total_clips = len(pat)
        gb = pat.get("file_size_bytes", pd.Series([0])).sum() / 1e9
        patient_summary_rows.append(
            '<tr><td style="font-weight:600;">%s</td>'
            '<td>%d</td><td>%d</td>'
            '<td>%.1f GB</td></tr>' % (pid, n_uuids, total_clips, gb)
        )

    # Detect window size from the matched sets
    if matched_sets and matched_sets[0]["windows"]:
        win0 = matched_sets[0]["windows"][0]
        win_size = win0["end_hr"] - win0["start_hr"]
    else:
        win_size = 2

    win_str = "%g" % win_size

    html = (
        '<!DOCTYPE html>\n<html>\n<head>\n<meta charset="utf-8">\n'
        '<title>Cross-Patient Matched Windows Review</title>\n'
        '<style>\n'
        '  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;\n'
        '         max-width: 1600px; margin: 0 auto; padding: 20px; background: #fff; }\n'
        '  h1 { color: #1565c0; margin-bottom: 4px; }\n'
        '  .subtitle { color: #666; margin-bottom: 20px; font-size: 14px; }\n'
        '  table { border-collapse: collapse; margin-bottom: 20px; }\n'
        '  th, td { padding: 6px 14px; text-align: left; border: 1px solid #ddd; }\n'
        '  th { background: #f5f5f5; font-size: 13px; }\n'
        '  td { font-size: 13px; }\n'
        '  .instructions { background: #e3f2fd; border-radius: 8px; padding: 14px;\n'
        '                   margin-bottom: 24px; font-size: 13px; line-height: 1.6; }\n'
        '</style>\n</head>\n<body>\n'
        '<h1>Cross-Patient Matched Contiguous Windows</h1>\n'
        '<div class="subtitle">\n'
        '  ' + str(len(matched_sets)) + ' matched sets across '
        + str(n_patients) + ' patients &middot;\n'
        '  Each set is a ' + win_str + '-hour contiguous window per patient, '
        'matched by brightness + saturation\n'
        '</div>\n\n'
        '<div class="instructions">\n'
        '  <b>How to use:</b> Each set shows a ' + win_str + '-hour contiguous '
        'window of AVIs per patient,\n'
        '  matched by similar brightness and color saturation. All clips within '
        'each window\n'
        '  are shown as a thumbnail strip so you can verify visual consistency. '
        'Sets are\n'
        '  ordered by match quality (Set 1 = closest match across patients). '
        'Scroll within\n'
        '  each column to see all clips.\n'
        '</div>\n\n'
        '<h3>Patient Summary (filtered dataset)</h3>\n'
        '<table>\n'
        '  <tr><th>Patient</th><th>Folders</th><th>Clips</th><th>Size</th></tr>\n'
        '  ' + "".join(patient_summary_rows) + '\n'
        '</table>\n\n'
        '<h3>Matched Windows</h3>\n'
        + "".join(set_sections) + '\n\n'
        '<div style="margin-top:30px;padding-top:16px;border-top:1px solid #eee;\n'
        '            color:#999;font-size:11px;">\n'
        '  Generated by 06_movement_screening.py --matched-review\n'
        '</div>\n</body>\n</html>'
    )

    output_html.write_text(html, encoding="utf-8")
    size_mb = output_html.stat().st_size / (1024 * 1024)
    print("  Wrote %s (%.1f MB)" % (output_html, size_mb))


def _run_matched_review(patient_ids: list[str], window_hr: float = MATCH_WINDOW_HR):
    """Load filtered copy list, compute matched contiguous windows, generate HTML."""
    filtered_csv = COPY_LIST_DIR / "copy_list_filtered.csv"
    if not filtered_csv.exists():
        print("ERROR: %s not found." % filtered_csv, file=sys.stderr)
        print("  Run --build-copy-list first, then apply exclusions.",
              file=sys.stderr)
        return

    df = pd.read_csv(filtered_csv)
    df = df[df["patient_id"].isin(patient_ids)]
    print("Loaded %d clips from %s" % (len(df), filtered_csv))
    print("  Patients: %s" % sorted(df["patient_id"].unique()))

    # Ensure offset_hr
    if "offset_hr" not in df.columns and "recording_offset_sec" in df.columns:
        df["offset_hr"] = df["recording_offset_sec"] / 3600

    # Compute matched windows
    print("\nComputing matched contiguous %d-hour windows..." % window_hr)
    matched_sets = _compute_matched_windows(df, n_sets=10, window_hr=window_hr)
    if not matched_sets:
        print("  No matched sets found.")
        return

    for ms in matched_sets:
        wins_str = "  |  ".join(
            "%s %s AVI %04d-%04d %.0f-%.0fh n=%d b=%.0f s=%.0f" % (
                w["patient_id"], w["uuid_short"],
                w["avi_min"], w["avi_max"],
                w["start_hr"], w["end_hr"], w["n_clips"],
                w["mean_bright"], w["mean_sat"],
            )
            for w in ms["windows"]
        )
        print("  Set %d: dist=%.3f  %s" % (ms["set_idx"], ms["max_dist"], wins_str))

    # Generate HTML
    thumbs_dir = COPY_LIST_DIR / "thumbs"
    output_html = COPY_LIST_DIR / "matched_sets_review.html"
    print("\nGenerating matched-windows review HTML...")
    generate_matched_review_html(matched_sets, df, thumbs_dir, output_html)
    print("\nDone. Open matched_sets_review.html in a browser to review.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-patient movement screening via frame differencing"
    )
    parser.add_argument(
        "--patients", required=True,
        help="Comma-separated patient IDs (e.g. EM1279,EM1269,EM1201,EM1287)"
    )
    parser.add_argument(
        "--triage", type=Path, default=DEFAULT_TRIAGE_CSV,
        help="Path to video_triage.csv"
    )
    parser.add_argument(
        "--metadata", type=Path, default=METADATA_CSV,
        help="Path to video_metadata.csv"
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG_YAML,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of AVIs to sample per folder (default {DEFAULT_SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds per AVI (default {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data_root from config.yaml"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running"
    )
    parser.add_argument(
        "--thumbnails", action="store_true",
        help="Extract thumbnails from screened AVIs and generate review HTML"
    )
    parser.add_argument(
        "--tag-brightness", action="store_true",
        help="Retroactively measure brightness for existing screening CSVs"
    )
    parser.add_argument(
        "--scan-all", action="store_true",
        help="Scan ALL AVIs (one frame each) for brightness/grayscale classification"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers for --scan-all / --build-copy-list (default 4)"
    )
    parser.add_argument(
        "--build-copy-list", action="store_true",
        help="Build filtered copy list with thumbnails and review HTML"
    )
    parser.add_argument(
        "--matched-review", action="store_true",
        help="Generate matched-pairs review HTML from copy_list_filtered.csv"
    )
    parser.add_argument(
        "--window-hours", type=float, default=MATCH_WINDOW_HR,
        help="Window size in hours for --matched-review (default: %.0f)" % MATCH_WINDOW_HR
    )
    args = parser.parse_args()

    patient_ids = [p.strip() for p in args.patients.split(",")]

    # --build-copy-list mode: filter + thumbnails + review HTML
    # --matched-review mode: generate matched-pairs HTML from filtered CSV
    if args.matched_review:
        _run_matched_review(patient_ids, window_hr=args.window_hours)
        return

    if args.build_copy_list:
        _run_build_copy_list(
            patient_ids, args.config, args.metadata,
            args.data_root, args.timeout, args.workers,
        )
        return

    # --scan-all mode: scan ALL AVIs for brightness/grayscale
    if args.scan_all:
        _run_scan_all(
            patient_ids, args.triage, args.metadata, args.config,
            args.data_root, args.timeout, args.workers,
        )
        return

    # --tag-brightness mode: retroactively add brightness to existing CSVs
    if args.tag_brightness:
        _run_tag_brightness(patient_ids, args.config, args.data_root, args.timeout)
        return

    # --thumbnails mode: extract thumbs from existing CSVs + build HTML
    if args.thumbnails:
        _run_thumbnails(patient_ids, args.config, args.data_root, args.timeout)
        return

    # Load data
    print("Loading config, triage, and metadata...")
    config = load_config(args.config)
    data_root = args.data_root or config.get("data_root", "")
    triage = load_triage(args.triage, patient_ids)
    metadata = load_metadata(args.metadata)

    print(f"  Patients: {patient_ids}")
    print(f"  Data root: {data_root}")
    print(f"  Usable folders: {len(triage)}")
    print(f"  Timeout per AVI: {args.timeout}s")

    # Build tasks
    tasks = build_folder_tasks(triage, metadata)
    print(f"  Folder tasks: {len(tasks)}")

    total_sampled = 0
    for t in tasks:
        n = min(args.sample_size, len(t["avi_list"]))
        total_sampled += n
    print(f"  AVIs to sample: {total_sampled} "
          f"(~{args.sample_size} per folder)")

    if args.dry_run:
        print("\n--- DRY RUN ---")
        for t in tasks:
            sampled = sample_avis(t["avi_list"], args.sample_size)
            print(f"\n  {t['patient_id']} / {t['uuid'][:10]}...")
            print(f"    Folder: {t['folder_name']}")
            print(f"    Total AVIs: {t['avi_count']}, "
                  f"Sampling: {len(sampled)}, "
                  f"Duration: {t['total_duration']/3600:.1f} hr")
            for s in sampled:
                print(f"      idx {s['avi_index']:>4}  "
                      f"offset {s['start_time']/3600:.2f} hr  "
                      f"{s['file_name']}")
        return

    # Group tasks by patient for sequential processing
    tasks_by_patient = {}
    for t in tasks:
        tasks_by_patient.setdefault(t["patient_id"], []).append(t)

    SCREENING_DIR.mkdir(parents=True, exist_ok=True)
    grand_t0 = time.time()

    for pid_idx, pid in enumerate(patient_ids, 1):
        patient_csv = SCREENING_DIR / f"screening_{pid}.csv"

        # Skip patients that already have results
        if patient_csv.exists():
            existing = pd.read_csv(patient_csv)
            ok_count = (existing["status"] == "ok").sum()
            print(f"\n{'='*80}")
            print(f"[{pid_idx}/{len(patient_ids)}] {pid} — SKIPPING, "
                  f"already have {patient_csv.name} ({len(existing)} rows, {ok_count} ok)")
            print(f"  Delete {patient_csv} to reprocess.")
            continue

        patient_tasks = tasks_by_patient.get(pid, [])
        if not patient_tasks:
            print(f"\n{'='*80}")
            print(f"[{pid_idx}/{len(patient_ids)}] {pid} — no usable folders, skipping")
            continue

        folder_count = len(patient_tasks)
        avi_count = sum(min(args.sample_size, len(t["avi_list"])) for t in patient_tasks)
        print(f"\n{'='*80}")
        print(f"[{pid_idx}/{len(patient_ids)}] {pid} — "
              f"{folder_count} folders, {avi_count} AVIs to sample")
        print("=" * 80)

        patient_t0 = time.time()
        patient_results = []

        for t_idx, task in enumerate(patient_tasks, 1):
            uuid_short = task["uuid"][:10]
            sampled_count = min(args.sample_size, len(task["avi_list"]))
            print(f"\n  Folder {t_idx}/{folder_count}: {uuid_short}... "
                  f"({sampled_count} AVIs, {task['total_duration']/3600:.1f} hr)")

            rows = process_folder(task, data_root, args.sample_size, args.timeout)
            patient_results.extend(rows)

            ok_count = sum(1 for r in rows if r.get("status") == "ok")
            fail_count = sum(1 for r in rows if r.get("status") == "failed")
            timeout_count = sum(1 for r in rows if r.get("status") == "timeout")
            parts = [f"{ok_count} ok"]
            if fail_count:
                parts.append(f"{fail_count} failed")
            if timeout_count:
                parts.append(f"{timeout_count} timeout")
            print(f"  Folder done: {', '.join(parts)}")

        patient_elapsed = time.time() - patient_t0
        print(f"\n  {pid} complete in {patient_elapsed:.1f}s")

        # Save per-patient CSV
        if patient_results:
            pdf = pd.DataFrame(patient_results)
            pdf = _reorder_columns(pdf)
            pdf.to_csv(patient_csv, index=False)
            print(f"  Saved: {patient_csv} ({len(pdf)} rows)")

    grand_elapsed = time.time() - grand_t0

    # Merge all per-patient CSVs into combined summary
    print(f"\n{'='*80}")
    print("MERGING PER-PATIENT RESULTS")
    print("=" * 80)

    per_patient_csvs = sorted(SCREENING_DIR.glob("screening_EM*.csv"))
    if not per_patient_csvs:
        print("No per-patient CSVs found.")
        return

    parts = []
    for csv_path in per_patient_csvs:
        pdf = pd.read_csv(csv_path)
        pid = csv_path.stem.replace("screening_", "")
        ok_count = (pdf["status"] == "ok").sum()
        print(f"  {pid}: {len(pdf)} rows ({ok_count} ok)")
        parts.append(pdf)

    df = pd.concat(parts, ignore_index=True)
    df = _reorder_columns(df)

    out_csv = SCREENING_DIR / "screening_summary.csv"
    df.to_csv(out_csv, index=False)

    status_counts = df["status"].value_counts().to_dict()
    print(f"\n  Combined: {out_csv}")
    print(f"  Total rows: {len(df)} ({status_counts})")
    print(f"\nALL DONE in {grand_elapsed:.1f}s")

    # Print cross-patient comparison
    print_cross_patient_summary(df)


if __name__ == "__main__":
    main()
