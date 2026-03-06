"""
04_build_detail_review.py — Build detailed review sheet for cross-patient comparison

Phase 2 of the M1 video-EEG pipeline. Reads video_triage.csv (usable folders from
Phase 1 contact sheet) and video_metadata.csv to extract timeline frames spread
across full recordings, compute motion heatmaps, detect quality issues, and generate
a self-contained HTML comparison sheet.

For each usable folder:
  - Extracts ~12 frames spread evenly across the FULL recording duration
  - Computes a motion heatmap from a mid-recording AVI sample
  - Flags quality issues (black frames, static video, short recordings, etc.)

Usage:
    python scripts/04_build_detail_review.py                            # default
    python scripts/04_build_detail_review.py --triage video_triage.csv  # custom triage path
    python scripts/04_build_detail_review.py --patient EM1334           # single patient
    python scripts/04_build_detail_review.py --parallel 4               # workers (default 4)
    python scripts/04_build_detail_review.py --dry-run                  # show plan only
    python scripts/04_build_detail_review.py --skip-motion              # timeline only (faster)
    python scripts/04_build_detail_review.py --resume                   # skip existing outputs
"""

import argparse
import base64
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
DETAIL_DIR = OUTPUT_DIR / "detail_review"
DETAIL_INDEX_CSV = DETAIL_DIR / "detail_index.csv"
OUTPUT_HTML = OUTPUT_DIR / "detail_review.html"

DEFAULT_PARALLEL = 4
TIMELINE_FRAMES = 12
MIN_TIMELINE_FRAMES = 3
SHORT_RECORDING_SEC = 300       # fewer frames below 5 min
THUMB_QUALITY = 85
MAX_FRAME_WIDTH = 640           # resize extracted frames for manageable HTML
MOTION_SAMPLE_FRAMES = 1800    # ~60 sec at 30 fps

# Quality flag thresholds
FLAG_SHORT_RECORDING_SEC = 7200   # 2 hours
FLAG_VERY_SHORT_AVI_SEC = 10
FLAG_SIZE_CV_THRESHOLD = 0.5
FLAG_BLACK_FRAME_MEAN = 15
FLAG_STATIC_DIFF_MEAN = 2
FLAG_LOW_MOTION_MEAN = 10
FLAG_HIGH_EDGE_RATIO = 0.6


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_triage(triage_path: Path, patient_filter: str | None = None) -> pd.DataFrame:
    """Load triage CSV, filter to usable folders."""
    df = pd.read_csv(triage_path)
    usable = df[df["usable"].str.lower() == "yes"].copy()
    if patient_filter:
        usable = usable[usable["patient_id"] == patient_filter]
    return usable


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, low_memory=False)


def load_electrode_counts(config: dict) -> dict[str, int]:
    return {
        pid: info.get("electrode_count", 0)
        for pid, info in config.get("patients", {}).items()
    }


# ---------------------------------------------------------------------------
# Build folder work list
# ---------------------------------------------------------------------------
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

        # Build per-AVI timeline
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
                "file_size_bytes": mrow.get("file_size_bytes", 0),
                "is_probed": bool(mrow.get("is_probed", False)),
                "error": str(mrow.get("error", "")) if pd.notna(mrow.get("error", "")) else "",
            })
            cumulative += dur

        total_duration = cumulative

        # Resolution / codec from probed row
        probed = folder_df[folder_df["is_probed"] == True]
        if not probed.empty:
            p = probed.iloc[0]
            width = int(p["width"])
            height = int(p["height"])
            fps = float(p["fps"])
            codec = str(p["codec_fourcc"])
            settings_changed = bool(p["settings_changed"])
        else:
            width, height, fps, codec = 0, 0, 30.0, "?"
            settings_changed = False

        tasks.append({
            "patient_id": pid,
            "uuid": uuid,
            "folder_name": folder_name,
            "avi_list": avi_list,
            "total_duration": total_duration,
            "avi_count": len(folder_df),
            "width": width,
            "height": height,
            "fps": fps,
            "codec": codec,
            "settings_changed": settings_changed,
            # Triage annotations (carry through)
            "camera_view": str(trow["camera_view"]) if pd.notna(trow.get("camera_view")) else "",
            "lighting": str(trow["lighting"]) if pd.notna(trow.get("lighting")) else "",
            "patient_visible": str(trow["patient_visible"]) if pd.notna(trow.get("patient_visible")) else "",
            "triage_notes": str(trow["notes"]) if pd.notna(trow.get("notes")) else "",
        })

    return tasks


# ---------------------------------------------------------------------------
# Timeline frame extraction
# ---------------------------------------------------------------------------
def pick_timeline_timestamps(total_duration: float) -> list[float]:
    """Pick evenly-spaced timestamps across the full duration."""
    n = MIN_TIMELINE_FRAMES if total_duration < SHORT_RECORDING_SEC else TIMELINE_FRAMES
    if n <= 1:
        return [total_duration / 2]
    margin = total_duration * 0.02
    span = total_duration - 2 * margin
    if span <= 0:
        return [total_duration / 2]
    return [margin + span * i / (n - 1) for i in range(n)]


def find_avi_for_timestamp(
    avi_list: list[dict], timestamp: float,
) -> tuple[str, float]:
    """Return (avi_file_name, local_offset_sec) for a global timestamp."""
    for avi in avi_list:
        end = avi["start_time"] + avi["duration"]
        if timestamp < end:
            return avi["file_name"], max(0.0, timestamp - avi["start_time"])
    if avi_list:
        last = avi_list[-1]
        return last["file_name"], last["duration"] * 0.9
    return "", 0.0


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Resize frame to max_width, preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_timeline_frame(
    folder_path: Path,
    avi_name: str,
    local_offset: float,
    fps: float,
    output_path: Path,
    max_width: int,
) -> dict:
    """Extract and save a single frame at a given time offset."""
    avi_path = folder_path / avi_name
    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return {"success": False, "error": f"cannot open {avi_name}",
                    "mean_pixel": 0}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        target = int(local_offset * actual_fps)
        target = max(0, min(target, frame_count - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return {"success": False, "error": f"read failed at frame {target}",
                    "mean_pixel": 0}

        frame = resize_frame(frame, max_width)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame,
                     [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])

        return {"success": True, "error": "",
                "mean_pixel": float(np.mean(frame))}

    except Exception as e:
        return {"success": False, "error": str(e), "mean_pixel": 0}


def extract_timeline_frames(
    data_root: str,
    task: dict,
    output_dir: Path,
    max_width: int,
) -> list[dict]:
    """Extract all timeline frames for one folder."""
    pid = task["patient_id"]
    uuid = task["uuid"]
    folder_path = Path(data_root) / task["folder_name"]

    timestamps = pick_timeline_timestamps(task["total_duration"])
    results = []

    for i, ts in enumerate(timestamps):
        avi_name, offset = find_avi_for_timestamp(task["avi_list"], ts)
        out_path = output_dir / pid / f"{uuid}_t{i:02d}.jpg"

        if not avi_name:
            results.append({"frame_index": i, "timestamp": ts,
                            "success": False, "error": "no AVI",
                            "mean_pixel": 0, "output_path": ""})
            continue

        r = extract_timeline_frame(
            folder_path, avi_name, offset, task["fps"], out_path, max_width)
        r["frame_index"] = i
        r["timestamp"] = ts
        r["output_path"] = str(out_path) if r["success"] else ""
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Motion heatmap
# ---------------------------------------------------------------------------
def compute_motion_heatmap(
    data_root: str,
    task: dict,
    output_dir: Path,
    max_width: int,
) -> dict:
    """Accumulate frame diffs from mid-recording AVI → blended heatmap."""
    pid = task["patient_id"]
    uuid = task["uuid"]
    folder_path = Path(data_root) / task["folder_name"]
    out_path = output_dir / pid / f"{uuid}_motion.jpg"

    mid_time = task["total_duration"] / 2
    avi_name, local_offset = find_avi_for_timestamp(task["avi_list"], mid_time)
    if not avi_name:
        return _motion_fail("no AVI for mid-point")

    avi_path = folder_path / avi_name
    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return _motion_fail(f"cannot open {avi_name}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or task["fps"]

        start = int(local_offset * fps) - MOTION_SAMPLE_FRAMES // 2
        start = max(0, min(start, frame_count - MOTION_SAMPLE_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        ret, prev = cap.read()
        if not ret or prev is None:
            cap.release()
            return _motion_fail("cannot read first frame")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        background = prev.copy()
        h, w = prev_gray.shape
        accum = np.zeros((h, w), dtype=np.float64)

        frames_read = 0
        for _ in range(MOTION_SAMPLE_FRAMES - 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            accum += cv2.absdiff(gray, prev_gray).astype(np.float64)
            prev_gray = gray
            frames_read += 1

        cap.release()

        if frames_read < 10:
            return _motion_fail(f"only {frames_read} frames")

        # Normalize → colormap → blend
        mx = accum.max()
        norm = (accum / mx * 255).astype(np.uint8) if mx > 0 else np.zeros_like(prev_gray)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(background, 0.5, heatmap, 0.5, 0)
        blended = resize_frame(blended, max_width)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), blended,
                     [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])

        # Stats for quality flags
        mean_intensity = float(np.mean(norm))
        border = int(min(h, w) * 0.15)
        mask = np.zeros((h, w), dtype=bool)
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True
        total_energy = float(np.sum(accum))
        edge_energy = float(np.sum(accum[mask]))
        edge_ratio = edge_energy / total_energy if total_energy > 0 else 0

        return {"success": True, "error": "", "output_path": str(out_path),
                "mean_intensity": mean_intensity, "edge_ratio": edge_ratio,
                "frames_read": frames_read}

    except Exception as e:
        return _motion_fail(str(e))


def _motion_fail(msg: str) -> dict:
    return {"success": False, "error": msg, "output_path": "",
            "mean_intensity": 0, "edge_ratio": 0, "frames_read": 0}


# ---------------------------------------------------------------------------
# Static-frame check (post-extraction)
# ---------------------------------------------------------------------------
def check_static_frames(timeline_results: list[dict]) -> bool:
    """Load consecutive extracted JPEGs and check for near-identical pairs."""
    ok = sorted(
        [r for r in timeline_results if r["success"] and r["output_path"]],
        key=lambda r: r["frame_index"],
    )
    for i in range(len(ok) - 1):
        try:
            a = cv2.imread(ok[i]["output_path"], cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(ok[i + 1]["output_path"], cv2.IMREAD_GRAYSCALE)
            if a is None or b is None or a.shape != b.shape:
                continue
            if float(np.mean(cv2.absdiff(a, b))) < FLAG_STATIC_DIFF_MEAN:
                return True
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------
FLAG_META = {
    "short_recording":  ("warn",  "Total duration < 2 hours"),
    "very_short_avis":  ("warn",  "AVI shorter than 10 seconds detected"),
    "size_variance":    ("warn",  "File-size CV > 0.5 across AVIs"),
    "settings_changed": ("warn",  "Resolution or FPS changed mid-recording"),
    "probe_errors":     ("error", "cv2 failed to open an AVI during probe"),
    "black_frames":     ("error", "Timeline frame mostly black (mean < 15)"),
    "static_frames":    ("warn",  "Consecutive timeline frames near-identical"),
    "low_motion":       ("warn",  "Very little movement in heatmap sample"),
    "high_motion_edge": ("warn",  "Motion concentrated at frame edges"),
}


def compute_quality_flags(
    task: dict,
    timeline_results: list[dict],
    motion_result: dict | None,
    static: bool,
) -> dict[str, bool]:
    flags: dict[str, bool] = {}

    # Metadata-derived
    flags["short_recording"] = task["total_duration"] < FLAG_SHORT_RECORDING_SEC
    flags["very_short_avis"] = any(
        a["duration"] < FLAG_VERY_SHORT_AVI_SEC for a in task["avi_list"])

    sizes = [a["file_size_bytes"] for a in task["avi_list"]
             if a["file_size_bytes"] and not pd.isna(a["file_size_bytes"])]
    if len(sizes) >= 2:
        m, s = np.mean(sizes), np.std(sizes)
        flags["size_variance"] = (s / m > FLAG_SIZE_CV_THRESHOLD) if m > 0 else False
    else:
        flags["size_variance"] = False

    flags["settings_changed"] = task["settings_changed"]
    flags["probe_errors"] = any(bool(a["error"]) for a in task["avi_list"])

    # Extraction-derived
    ok_frames = [r for r in timeline_results if r["success"]]
    flags["black_frames"] = any(
        r.get("mean_pixel", 255) < FLAG_BLACK_FRAME_MEAN for r in ok_frames)
    flags["static_frames"] = static

    if motion_result and motion_result.get("success"):
        flags["low_motion"] = motion_result["mean_intensity"] < FLAG_LOW_MOTION_MEAN
        flags["high_motion_edge"] = motion_result["edge_ratio"] > FLAG_HIGH_EDGE_RATIO
    else:
        flags["low_motion"] = False
        flags["high_motion_edge"] = False

    return flags


# ---------------------------------------------------------------------------
# Process single folder
# ---------------------------------------------------------------------------
def process_folder(
    data_root: str,
    task: dict,
    output_dir: Path,
    skip_motion: bool,
    max_width: int,
) -> dict:
    pid = task["patient_id"]
    uuid = task["uuid"]

    timeline = extract_timeline_frames(data_root, task, output_dir, max_width)
    static = check_static_frames(timeline)

    if skip_motion:
        motion = _motion_fail("skipped")
    else:
        motion = compute_motion_heatmap(data_root, task, output_dir, max_width)

    flags = compute_quality_flags(task, timeline, motion, static)

    return {
        "patient_id": pid,
        "uuid": uuid,
        "folder_name": task["folder_name"],
        "total_duration": task["total_duration"],
        "avi_count": task["avi_count"],
        "width": task["width"],
        "height": task["height"],
        "fps": task["fps"],
        "codec": task["codec"],
        "camera_view": task["camera_view"],
        "lighting": task["lighting"],
        "patient_visible": task["patient_visible"],
        "triage_notes": task["triage_notes"],
        "timeline": timeline,
        "timeline_ok": sum(1 for r in timeline if r["success"]),
        "timeline_fail": sum(1 for r in timeline if not r["success"]),
        "motion": motion,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def find_completed_uuids(detail_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not detail_dir.exists():
        return completed
    for pdir in detail_dir.iterdir():
        if not pdir.is_dir():
            continue
        for f in pdir.iterdir():
            if f.suffix == ".jpg" and "_t00" in f.name:
                completed.add(f.stem.split("_t00")[0])
    return completed


# ---------------------------------------------------------------------------
# Detail index CSV
# ---------------------------------------------------------------------------
def write_detail_index(results: list[dict], path: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "patient_id": r["patient_id"],
            "uuid": r["uuid"],
            "folder_name": r["folder_name"],
            "total_duration": r["total_duration"],
            "avi_count": r["avi_count"],
            "width": r["width"],
            "height": r["height"],
            "timeline_ok": r["timeline_ok"],
            "timeline_fail": r["timeline_fail"],
            "motion_ok": r["motion"]["success"],
            "motion_mean_intensity": r["motion"].get("mean_intensity", 0),
            "motion_edge_ratio": r["motion"].get("edge_ratio", 0),
        }
        for fname, fval in r["flags"].items():
            row[f"flag_{fname}"] = fval
        rows.append(row)

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def encode_image_base64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('ascii')}"
    except (OSError, IOError):
        return ""


def fmt_dur(sec: float) -> str:
    if sec != sec:
        return "?"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


def fmt_ts(sec: float) -> str:
    """Short timestamp label like '0h', '2.5h'."""
    h = sec / 3600
    if h < 1:
        return f"{int(sec // 60)}m"
    if h == int(h):
        return f"{int(h)}h"
    return f"{h:.1f}h"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def generate_html(
    results: list[dict],
    electrode_counts: dict[str, int],
) -> str:
    """Build self-contained HTML comparison sheet."""

    # Organise by patient
    by_patient: dict[str, list[dict]] = {}
    for r in results:
        by_patient.setdefault(r["patient_id"], []).append(r)

    # Overall stats
    total_patients = len(by_patient)
    total_folders = len(results)
    total_hours = sum(r["total_duration"] for r in results) / 3600
    any_flags = sum(1 for r in results if any(r["flags"].values()))

    # ---- Patient summary cards (top grid) ----
    summary_cards_html = []
    for pid in sorted(by_patient):
        folders = by_patient[pid]
        n_elec = electrode_counts.get(pid, 0)
        n_folders = len(folders)
        p_hours = sum(f["total_duration"] for f in folders) / 3600

        # Representative images: mid-frame + motion from longest folder
        longest = max(folders, key=lambda f: f["total_duration"])
        mid_idx = len(longest["timeline"]) // 2
        mid_frames = [t for t in longest["timeline"] if t["success"]]
        rep_frame_b64 = ""
        if mid_frames:
            mid_frame = min(mid_frames, key=lambda t: abs(t["frame_index"] - mid_idx))
            rep_frame_b64 = encode_image_base64(mid_frame["output_path"])
        motion_b64 = ""
        if longest["motion"]["success"]:
            motion_b64 = encode_image_base64(longest["motion"]["output_path"])

        # Flag count
        flag_count = sum(
            1 for f in folders for v in f["flags"].values() if v)

        flag_html = ""
        if flag_count:
            flag_html = f'<span class="flag-count">{flag_count} flags</span>'

        rep_img = (f'<img src="{rep_frame_b64}" loading="lazy">'
                   if rep_frame_b64 else '<div class="no-img">no frame</div>')
        mot_img = (f'<img src="{motion_b64}" loading="lazy">'
                   if motion_b64 else '<div class="no-img">no motion</div>')

        summary_cards_html.append(f"""
        <a href="#pat-{pid}" class="summary-card">
          <div class="sc-header">{pid}</div>
          <div class="sc-stats">{n_elec} electrodes &middot; {n_folders} folders &middot; {p_hours:.0f}h</div>
          <div class="sc-images">{rep_img}{mot_img}</div>
          {flag_html}
        </a>""")

    # ---- Per-folder detail sections ----
    nav_items = []
    patient_sections = []

    for pid in sorted(by_patient):
        folders = sorted(by_patient[pid], key=lambda f: f["folder_name"])
        n_elec = electrode_counts.get(pid, 0)
        p_hours = sum(f["total_duration"] for f in folders) / 3600

        nav_items.append(
            f'<a href="#pat-{pid}" class="nav-item">'
            f'{pid} <span class="nav-badge">{len(folders)}</span></a>')

        folder_cards = []
        for fld in folders:
            uuid = fld["uuid"]
            res = f"{fld['width']}x{fld['height']}" if fld["width"] else "?"
            dur = fmt_dur(fld["total_duration"])

            # Triage annotations (read-only display)
            annot_parts = []
            if fld["camera_view"]:
                annot_parts.append(fld["camera_view"])
            if fld["lighting"]:
                annot_parts.append(fld["lighting"])
            if fld["patient_visible"]:
                annot_parts.append(fld["patient_visible"])
            annot_str = " &middot; ".join(annot_parts) if annot_parts else ""

            # Timeline strip
            timeline_imgs = []
            for t in fld["timeline"]:
                ts_label = fmt_ts(t["timestamp"])
                if t["success"]:
                    b64 = encode_image_base64(t["output_path"])
                    cls = ""
                    if t.get("mean_pixel", 255) < FLAG_BLACK_FRAME_MEAN:
                        cls = ' class="frame-flagged"'
                    timeline_imgs.append(
                        f'<div class="tl-cell">'
                        f'<img src="{b64}"{cls} loading="lazy">'
                        f'<div class="tl-label">{ts_label}</div></div>')
                else:
                    timeline_imgs.append(
                        f'<div class="tl-cell">'
                        f'<div class="tl-placeholder">fail</div>'
                        f'<div class="tl-label">{ts_label}</div></div>')

            # Motion heatmap
            motion_html = ""
            if fld["motion"]["success"]:
                mb64 = encode_image_base64(fld["motion"]["output_path"])
                motion_html = (
                    f'<div class="motion-panel">'
                    f'<div class="motion-title">Motion heatmap (60 s mid-recording)</div>'
                    f'<img src="{mb64}" loading="lazy"></div>')

            # Quality flag badges
            badge_html = []
            for fname, fval in fld["flags"].items():
                if fval:
                    level, tip = FLAG_META.get(fname, ("warn", fname))
                    bcls = "badge-error" if level == "error" else "badge-warn"
                    badge_html.append(
                        f'<span class="badge {bcls}">'
                        f'<span class="flag-name">{fname}</span>'
                        f'<span class="flag-desc">{tip}</span></span>')
            badges_str = " ".join(badge_html)

            has_flags = any(fld["flags"].values())
            card_extra_cls = " flagged" if has_flags else ""

            # Build data attributes for JS export
            flag_data = " ".join(
                f'data-flag-{k}="{"1" if v else "0"}"'
                for k, v in fld["flags"].items())

            folder_cards.append(f"""
        <div class="folder-detail{card_extra_cls}" id="folder-{uuid}"
             data-uuid="{uuid}" data-patient="{pid}"
             data-folder="{fld['folder_name']}" data-res="{res}"
             data-avi-count="{fld['avi_count']}"
             data-duration="{fld['total_duration']:.0f}"
             data-camera="{fld['camera_view']}"
             data-lighting="{fld['lighting']}"
             data-visibility="{fld['patient_visible']}"
             data-triage-notes="{_esc(fld['triage_notes'])}"
             {flag_data}>
          <div class="fd-header">
            <span class="uuid" title="{fld['folder_name']}">{uuid}</span>
            <span class="fd-meta">
              {res} &middot; {fld['fps']:.0f} fps &middot; {fld['codec']} &middot;
              {fld['avi_count']} AVIs &middot; {dur}
            </span>
            {f'<span class="fd-annot">{annot_str}</span>' if annot_str else ""}
          </div>
          <div class="fd-badges">{badges_str}</div>
          <div class="timeline-strip">{''.join(timeline_imgs)}</div>
          {motion_html}
          <div class="review-controls" data-uuid="{uuid}">
            <label>Include:
              <select class="sel-include" onchange="onReviewChange(this)">
                <option value="">—</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
                <option value="maybe">Maybe</option>
              </select>
            </label>
            <input type="text" class="inp-notes" placeholder="Notes..."
                   onchange="onReviewChange(this)">
          </div>
        </div>""")

        patient_sections.append(f"""
      <div class="patient-section" id="pat-{pid}">
        <div class="patient-header">
          <h2>{pid}</h2>
          <span class="patient-stats">
            {n_elec} M1 electrodes &middot; {len(folders)} folders &middot;
            ~{p_hours:.0f}h
          </span>
        </div>
        {''.join(folder_cards)}
      </div>""")

    # ---- Assemble page ----
    html = _HTML_TEMPLATE.format(
        total_patients=total_patients,
        total_folders=total_folders,
        total_hours=f"{total_hours:,.0f}",
        any_flags=any_flags,
        summary_cards=''.join(summary_cards_html),
        nav_items=''.join(nav_items),
        patient_sections=''.join(patient_sections),
    )
    return html


def _esc(s: str) -> str:
    """Escape for HTML attributes."""
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


# ---------------------------------------------------------------------------
# HTML template (CSS + JS)
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Detail Review — M1 Pipeline Phase 2</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; color: #333; line-height: 1.4;
  }}

  /* ---- Header ---- */
  .header {{
    background: #1a1a2e; color: #fff; padding: 14px 24px;
    position: sticky; top: 0; z-index: 100;
  }}
  .header-top {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
  .header h1 {{ font-size: 20px; font-weight: 600; }}
  .header-stats {{ font-size: 13px; color: #aaa; margin-top: 2px; }}
  .header-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
  .progress-text {{ font-size: 13px; color: #aaa; }}
  .progress-bar {{ width: 120px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }}
  .progress-fill {{ height: 100%; background: #4caf50; border-radius: 3px; transition: width 0.3s; }}
  .btn {{
    padding: 7px 16px; border: none; border-radius: 4px;
    font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  .btn-export {{ background: #4caf50; color: #fff; }}
  .btn-export:hover {{ background: #43a047; }}
  .btn-filter {{ background: transparent; color: #aaa; border: 1px solid #555; }}
  .btn-filter:hover {{ color: #fff; border-color: #888; }}
  .btn-filter.active {{ background: #e65100; color: #fff; border-color: #e65100; }}
  .btn-clear {{ background: transparent; color: #aaa; border: 1px solid #555; }}
  .btn-clear:hover {{ color: #fff; border-color: #888; }}

  /* ---- Layout ---- */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 170px; background: #fff; border-right: 1px solid #ddd;
    position: sticky; top: 80px; height: calc(100vh - 80px);
    overflow-y: auto; flex-shrink: 0; padding: 8px 0;
  }}
  .nav-item {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 14px; text-decoration: none; color: #333;
    font-size: 13px; font-weight: 500;
  }}
  .nav-item:hover {{ background: #f0f0f0; }}
  .nav-badge {{
    background: #e0e0e0; border-radius: 10px; padding: 1px 6px;
    font-size: 11px; color: #666;
  }}
  .main {{ flex: 1; padding: 16px 24px; max-width: 1400px; }}

  /* ---- Summary grid ---- */
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; margin-bottom: 32px;
  }}
  .summary-card {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 10px 14px; text-decoration: none; color: inherit;
    transition: border-color 0.2s, box-shadow 0.2s;
  }}
  .summary-card:hover {{ border-color: #999; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .sc-header {{ font-size: 16px; font-weight: 700; color: #1a1a2e; }}
  .sc-stats {{ font-size: 12px; color: #888; margin: 2px 0 6px; }}
  .sc-images {{ display: flex; gap: 6px; }}
  .sc-images img {{ width: 48%; height: auto; border-radius: 3px; border: 1px solid #eee; }}
  .no-img {{
    width: 48%; height: 60px; background: #f0f0f0; border-radius: 3px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; color: #bbb;
  }}
  .flag-count {{
    display: inline-block; margin-top: 6px; font-size: 11px; font-weight: 600;
    color: #e65100; background: #fff3e0; padding: 1px 8px; border-radius: 3px;
  }}

  /* ---- Section heading ---- */
  .section-title {{
    font-size: 22px; font-weight: 700; color: #1a1a2e;
    border-bottom: 3px solid #1a1a2e; padding-bottom: 4px;
    margin: 32px 0 16px;
  }}

  /* ---- Patient sections ---- */
  .patient-section {{ margin-bottom: 28px; }}
  .patient-header {{
    display: flex; align-items: baseline; gap: 14px;
    border-bottom: 2px solid #1a1a2e; padding-bottom: 4px; margin-bottom: 10px;
  }}
  .patient-header h2 {{ font-size: 18px; color: #1a1a2e; }}
  .patient-stats {{ font-size: 13px; color: #777; }}

  /* ---- Folder detail card ---- */
  .folder-detail {{
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    margin-bottom: 12px; padding: 12px 16px;
    transition: border-color 0.2s;
  }}
  .folder-detail:hover {{ border-color: #999; }}
  .folder-detail.included {{ border-left: 4px solid #4caf50; }}
  .folder-detail.excluded {{ border-left: 4px solid #ccc; background: #fafafa; opacity: 0.7; }}
  .folder-detail.maybe-include {{ border-left: 4px solid #ff9800; }}
  .folder-detail.hidden-by-filter {{ display: none; }}
  .fd-header {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 4px; }}
  .uuid {{ font-family: monospace; font-size: 12px; color: #555; }}
  .fd-meta {{ font-size: 12px; color: #888; }}
  .fd-annot {{ font-size: 12px; color: #5c6bc0; font-weight: 500; }}
  .fd-badges {{ margin-bottom: 6px; display: flex; gap: 4px; flex-wrap: wrap; }}
  .badge {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; padding: 3px 8px;
    border-radius: 3px; font-weight: 600;
  }}
  .badge-warn {{ background: #fff3cd; color: #856404; }}
  .badge-error {{ background: #f8d7da; color: #721c24; }}
  .badge .flag-name {{ font-weight: 700; }}
  .badge .flag-desc {{ font-weight: 400; opacity: 0.85; }}

  /* ---- Timeline strip ---- */
  .timeline-strip {{
    display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px;
    scrollbar-width: thin;
  }}
  .tl-cell {{ flex-shrink: 0; text-align: center; }}
  .tl-cell img {{
    width: 220px; height: auto; border-radius: 2px;
    border: 1px solid #eee; display: block;
  }}
  .tl-cell img.frame-flagged {{ border: 2px solid #dc3545; }}
  .tl-label {{ font-size: 10px; color: #aaa; margin-top: 1px; }}
  .tl-placeholder {{
    width: 220px; height: 124px; background: #f0f0f0; border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; color: #ccc;
  }}

  /* ---- Motion panel ---- */
  .motion-panel {{ margin-top: 8px; }}
  .motion-title {{ font-size: 11px; color: #888; margin-bottom: 2px; }}
  .motion-panel img {{ max-width: 400px; height: auto; border-radius: 3px; border: 1px solid #eee; }}

  /* ---- Review controls ---- */
  .review-controls {{
    display: flex; align-items: center; gap: 10px; margin-top: 8px;
    padding-top: 8px; border-top: 1px solid #f0f0f0; flex-wrap: wrap;
  }}
  .review-controls label {{ font-size: 13px; font-weight: 600; color: #555; }}
  .review-controls select {{
    padding: 4px 8px; border: 1px solid #ddd; border-radius: 3px;
    font-size: 12px; background: #fff; cursor: pointer;
  }}
  .inp-notes {{
    flex: 1; min-width: 200px; padding: 5px 8px; border: 1px solid #ddd;
    border-radius: 3px; font-size: 12px;
  }}
  .inp-notes:focus, .review-controls select:focus {{ border-color: #999; outline: none; }}

  /* ---- Toast ---- */
  .toast {{
    position: fixed; bottom: 24px; right: 24px; background: #333;
    color: #fff; padding: 12px 20px; border-radius: 6px;
    font-size: 13px; opacity: 0; transition: opacity 0.3s;
    z-index: 200; pointer-events: none;
  }}
  .toast.show {{ opacity: 1; }}

  @media (max-width: 900px) {{
    .sidebar {{ display: none; }}
    .tl-cell img {{ width: 160px; }}
    .tl-placeholder {{ width: 160px; height: 90px; }}
    .summary-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-top">
      <div>
        <h1>Detail Review &mdash; M1 Pipeline Phase 2</h1>
        <div class="header-stats">
          {total_patients} patients &middot; {total_folders} folders &middot;
          ~{total_hours}h &middot; {any_flags} flagged
        </div>
      </div>
      <div class="header-actions">
        <span class="progress-text" id="progress-text">0 / {total_folders} reviewed</span>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
        <button class="btn btn-filter" id="btn-filter" onclick="toggleFilter()">Show flagged only</button>
        <button class="btn btn-export" onclick="exportCSV()">Export CSV</button>
        <button class="btn btn-clear" onclick="clearAll()">Clear all</button>
      </div>
    </div>
  </div>

  <div class="layout">
    <div class="sidebar">
      {nav_items}
    </div>
    <div class="main">
      <div class="section-title">Cross-Patient Comparison</div>
      <div class="summary-grid">
        {summary_cards}
      </div>

      <div class="section-title">Per-Folder Detail</div>
      {patient_sections}
    </div>
  </div>
  <div class="toast" id="toast"></div>

<script>
const STORAGE_KEY = 'detail_review_v1';
const TOTAL = {total_folders};
let filterActive = false;

// ---- State ----
function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {{}}; }}
  catch {{ return {{}}; }}
}}
function saveState(s) {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(s)); }}

// ---- Init ----
function init() {{
  const state = loadState();
  document.querySelectorAll('.review-controls').forEach(el => {{
    const uuid = el.dataset.uuid;
    const s = state[uuid];
    if (!s) return;
    if (s.include) el.querySelector('.sel-include').value = s.include;
    if (s.notes) el.querySelector('.inp-notes').value = s.notes;
    updateCardStyle(el);
  }});
  updateProgress();
}}

function onReviewChange(input) {{
  const ctl = input.closest('.review-controls');
  const uuid = ctl.dataset.uuid;
  const state = loadState();
  state[uuid] = {{
    include: ctl.querySelector('.sel-include').value,
    notes: ctl.querySelector('.inp-notes').value,
  }};
  saveState(state);
  updateCardStyle(ctl);
  updateProgress();
}}

function updateCardStyle(ctl) {{
  const card = ctl.closest('.folder-detail');
  const v = ctl.querySelector('.sel-include').value;
  card.classList.remove('included', 'excluded', 'maybe-include');
  if (v === 'yes') card.classList.add('included');
  else if (v === 'no') card.classList.add('excluded');
  else if (v === 'maybe') card.classList.add('maybe-include');
}}

function updateProgress() {{
  const state = loadState();
  const n = Object.values(state).filter(s => s.include).length;
  document.getElementById('progress-text').textContent = n + ' / ' + TOTAL + ' reviewed';
  document.getElementById('progress-fill').style.width = (n / TOTAL * 100) + '%';
}}

// ---- Filter ----
function toggleFilter() {{
  filterActive = !filterActive;
  const btn = document.getElementById('btn-filter');
  btn.classList.toggle('active', filterActive);
  btn.textContent = filterActive ? 'Show all' : 'Show flagged only';
  document.querySelectorAll('.folder-detail').forEach(card => {{
    if (filterActive) {{
      const isFlagged = card.classList.contains('flagged') ||
        Array.from(card.attributes).some(a => a.name.startsWith('data-flag-') && a.value === '1');
      card.classList.toggle('hidden-by-filter', !isFlagged);
    }} else {{
      card.classList.remove('hidden-by-filter');
    }}
  }});
}}

// ---- Export CSV ----
function exportCSV() {{
  const state = loadState();
  const flagNames = [{flag_names_js}];
  const hdr = ['patient_id','uuid','folder_name','resolution','avi_count',
               'duration_sec','camera_view','lighting','patient_visible',
               'triage_notes','include','review_notes'].concat(flagNames.map(f=>'flag_'+f));
  const rows = [hdr.join(',')];

  document.querySelectorAll('.folder-detail').forEach(card => {{
    const uuid = card.dataset.uuid;
    const s = state[uuid] || {{}};
    const vals = [
      card.dataset.patient,
      uuid,
      '"' + (card.dataset.folder||'').replace(/"/g,'""') + '"',
      card.dataset.res,
      card.dataset.aviCount,
      card.dataset.duration,
      card.dataset.camera || '',
      card.dataset.lighting || '',
      card.dataset.visibility || '',
      '"' + (card.dataset.triageNotes||'').replace(/"/g,'""') + '"',
      s.include || '',
      '"' + (s.notes||'').replace(/"/g,'""') + '"',
    ];
    flagNames.forEach(f => vals.push(card.getAttribute('data-flag-'+f) || '0'));
    rows.push(vals.join(','));
  }});

  const csv = rows.join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'video_triage_detailed.csv'; a.click();
  URL.revokeObjectURL(url);
  showToast('Exported video_triage_detailed.csv');
}}

// ---- Clear ----
function clearAll() {{
  if (!confirm('Clear all review annotations?')) return;
  localStorage.removeItem(STORAGE_KEY);
  document.querySelectorAll('.sel-include').forEach(s => s.selectedIndex = 0);
  document.querySelectorAll('.inp-notes').forEach(i => i.value = '');
  document.querySelectorAll('.folder-detail').forEach(c =>
    c.classList.remove('included','excluded','maybe-include'));
  updateProgress();
  showToast('Annotations cleared');
}}

function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}}

document.addEventListener('keydown', e => {{
  if ((e.ctrlKey||e.metaKey) && e.key === 'e') {{ e.preventDefault(); exportCSV(); }}
}});

init();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Build detailed review sheet for cross-patient video comparison",
    )
    parser.add_argument(
        "--triage", type=str, default=str(DEFAULT_TRIAGE_CSV),
        help=f"Path to video_triage.csv (default: {DEFAULT_TRIAGE_CSV})",
    )
    parser.add_argument(
        "--metadata", type=str, default=str(METADATA_CSV),
        help=f"Path to video_metadata.csv (default: {METADATA_CSV})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DETAIL_DIR),
        help=f"Output directory for frames/heatmaps (default: {DETAIL_DIR})",
    )
    parser.add_argument(
        "--output-html", type=str, default=str(OUTPUT_HTML),
        help=f"Output HTML path (default: {OUTPUT_HTML})",
    )
    parser.add_argument(
        "--patient", type=str, default=None, metavar="EM_ID",
        help="Process a single patient (for testing)",
    )
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL, metavar="N",
        help=f"Number of concurrent workers (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without extracting anything",
    )
    parser.add_argument(
        "--skip-motion", action="store_true",
        help="Skip motion heatmap computation (faster)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folders that already have outputs",
    )
    parser.add_argument(
        "--max-width", type=int, default=MAX_FRAME_WIDTH, metavar="PX",
        help=f"Max frame width in pixels (default: {MAX_FRAME_WIDTH})",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data_root from config.yaml",
    )
    args = parser.parse_args()

    triage_path = Path(args.triage)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_html = Path(args.output_html)

    # 1. Load data
    print(f"Loading triage from {triage_path} ...")
    triage = load_triage(triage_path, args.patient)
    print(f"  {len(triage)} usable folders"
          + (f" (patient={args.patient})" if args.patient else ""))

    if triage.empty:
        print("No usable folders found. Check triage CSV and --patient filter.")
        return

    print(f"Loading metadata from {metadata_path} ...")
    metadata = load_metadata(metadata_path)
    print(f"  {len(metadata):,} rows")

    config = load_config(CONFIG_YAML)
    electrode_counts = load_electrode_counts(config)
    data_root = args.data_root or config["data_root"]
    print(f"  Data root: {data_root}")

    # 2. Build work list
    tasks = build_folder_tasks(triage, metadata)
    if not tasks:
        print("No folders to process after matching triage with metadata.")
        return

    # Resume
    if args.resume:
        done = find_completed_uuids(output_dir)
        before = len(tasks)
        tasks = [t for t in tasks if t["uuid"] not in done]
        print(f"  Resume: skipping {before - len(tasks)} done, {len(tasks)} remaining")

    if not tasks:
        print("All folders already processed. Nothing to do.")
        # Still generate HTML from existing outputs
        _generate_html_from_existing(
            triage_path, metadata_path, output_dir, output_html,
            electrode_counts, args.patient)
        return

    n_patients = len(set(t["patient_id"] for t in tasks))
    total_dur_h = sum(t["total_duration"] for t in tasks) / 3600

    # 3. Summary
    print(f"\n{'=' * 60}")
    print(f"Detail review extraction plan:")
    print(f"  Patients:         {n_patients}")
    print(f"  Folders:          {len(tasks)}")
    print(f"  Total duration:   ~{total_dur_h:.0f}h of recording")
    print(f"  Timeline frames:  up to {TIMELINE_FRAMES}/folder")
    print(f"  Motion heatmaps:  {'SKIP' if args.skip_motion else 'yes (60s sample each)'}")
    print(f"  Frame max width:  {args.max_width}px")
    print(f"  Parallel:         {args.parallel} workers")
    print(f"  Output:           {output_dir}")
    print(f"{'=' * 60}")

    if args.dry_run:
        print(f"\n[DRY RUN] Per-patient breakdown:")
        by_p: dict[str, list] = {}
        for t in tasks:
            by_p.setdefault(t["patient_id"], []).append(t)
        for pid in sorted(by_p):
            fl = by_p[pid]
            n_frames = sum(
                len(pick_timeline_timestamps(t["total_duration"])) for t in fl)
            print(f"  {pid}: {len(fl)} folders, ~{n_frames} timeline frames"
                  + ("" if args.skip_motion else f", {len(fl)} motion heatmaps"))
        est_min = len(tasks) * (0.15 + (0 if args.skip_motion else 0.2)) / args.parallel
        print(f"\nEstimated time: ~{est_min:.0f} min with {args.parallel} workers")
        print(f"Remove --dry-run to start extraction.")
        return

    # 4. Process folders in parallel
    print(f"\nProcessing folders ...")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_results: list[dict] = []
    done_count = 0
    err_count = 0

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(
                process_folder, data_root, task, output_dir,
                args.skip_motion, args.max_width,
            ): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            done_count += 1
            try:
                result = future.result()
                all_results.append(result)
                n_err = result["timeline_fail"]
                if not result["motion"]["success"] and not args.skip_motion:
                    n_err += 1
                if n_err:
                    err_count += n_err
            except Exception as e:
                err_count += 1
                print(f"  ERROR [{task['patient_id']}/{task['uuid']}]: {e}",
                      file=sys.stderr)

            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                print(f"  {done_count}/{len(tasks)} folders "
                      f"({elapsed:.1f}s, {rate:.1f}/s)")

    elapsed = time.time() - t0
    print(f"\nExtraction complete in {elapsed:.1f}s")

    # 5. Write detail index CSV
    index_path = output_dir / "detail_index.csv"
    write_detail_index(all_results, index_path)
    print(f"Wrote {index_path}")

    # 6. Compute output size
    total_size = 0
    for r in all_results:
        for t in r["timeline"]:
            if t["success"] and t["output_path"]:
                try:
                    total_size += os.path.getsize(t["output_path"])
                except OSError:
                    pass
        if r["motion"]["success"] and r["motion"]["output_path"]:
            try:
                total_size += os.path.getsize(r["motion"]["output_path"])
            except OSError:
                pass

    # 7. Generate HTML (always from ALL outputs on disk, not just new ones)
    _generate_html_from_existing(
        triage_path, metadata_path, output_dir, output_html,
        electrode_counts, args.patient)

    # 8. Summary
    total_tl = sum(r["timeline_ok"] for r in all_results)
    total_motion = sum(1 for r in all_results if r["motion"]["success"])
    total_flagged = sum(1 for r in all_results if any(r["flags"].values()))

    html_size = output_html.stat().st_size / (1024 * 1024) if output_html.exists() else 0

    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Folders processed:  {done_count}")
    print(f"  Timeline frames:    {total_tl} extracted (this run)")
    print(f"  Motion heatmaps:    {total_motion} computed (this run)")
    print(f"  Flagged folders:    {total_flagged} (this run)")
    print(f"  Output size:        {total_size / (1024*1024):.1f} MB (images, this run)")
    print(f"  HTML:               {output_html} ({html_size:.1f} MB)")
    print(f"{'=' * 60}")

    if err_count:
        print(f"\n  {err_count} errors during extraction (check stderr)")


def _generate_html_from_existing(
    triage_path: Path,
    metadata_path: Path,
    output_dir: Path,
    output_html: Path,
    electrode_counts: dict[str, int],
    patient_filter: str | None,
):
    """Regenerate HTML from previously extracted outputs (for --resume with all done)."""
    index_path = output_dir / "detail_index.csv"
    if not index_path.exists():
        print("No detail_index.csv found. Run without --resume first.")
        return

    print("Regenerating HTML from existing outputs ...")
    triage = load_triage(triage_path, patient_filter)
    metadata = load_metadata(metadata_path)
    tasks = build_folder_tasks(triage, metadata)

    # Rebuild results by scanning existing files
    results = []
    for task in tasks:
        pid = task["patient_id"]
        uuid = task["uuid"]
        patient_dir = output_dir / pid

        # Reconstruct timeline results
        timeline = []
        timestamps = pick_timeline_timestamps(task["total_duration"])
        for i, ts in enumerate(timestamps):
            fpath = patient_dir / f"{uuid}_t{i:02d}.jpg"
            if fpath.exists():
                try:
                    img = cv2.imread(str(fpath))
                    mp = float(np.mean(img)) if img is not None else 0
                except Exception:
                    mp = 0
                timeline.append({
                    "frame_index": i, "timestamp": ts, "success": True,
                    "error": "", "mean_pixel": mp, "output_path": str(fpath),
                })
            else:
                timeline.append({
                    "frame_index": i, "timestamp": ts, "success": False,
                    "error": "file missing", "mean_pixel": 0, "output_path": "",
                })

        static = check_static_frames(timeline)

        # Motion
        mpath = patient_dir / f"{uuid}_motion.jpg"
        if mpath.exists():
            motion = {"success": True, "error": "", "output_path": str(mpath),
                       "mean_intensity": 0, "edge_ratio": 0, "frames_read": 0}
        else:
            motion = _motion_fail("not found")

        flags = compute_quality_flags(task, timeline, motion, static)

        results.append({
            "patient_id": pid,
            "uuid": uuid,
            "folder_name": task["folder_name"],
            "total_duration": task["total_duration"],
            "avi_count": task["avi_count"],
            "width": task["width"],
            "height": task["height"],
            "fps": task["fps"],
            "codec": task["codec"],
            "camera_view": task["camera_view"],
            "lighting": task["lighting"],
            "patient_visible": task["patient_visible"],
            "triage_notes": task["triage_notes"],
            "timeline": timeline,
            "timeline_ok": sum(1 for t in timeline if t["success"]),
            "timeline_fail": sum(1 for t in timeline if not t["success"]),
            "motion": motion,
            "flags": flags,
        })

    if not results:
        print("No results to generate HTML from.")
        return

    global _HTML_TEMPLATE
    flag_names_js = ",".join(f"'{k}'" for k in FLAG_META)
    _HTML_TEMPLATE = _HTML_TEMPLATE.replace("{flag_names_js}", flag_names_js)

    html = generate_html(results, electrode_counts)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {output_html} ({output_html.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
