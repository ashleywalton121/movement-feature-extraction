"""
05_framing_comparison.py — Cross-patient camera framing comparison

Phase 3 of the M1 video-EEG pipeline. For each usable folder, computes a
"motion profile" (where the patient moves in the frame) by accumulating frame
diffs from a 60-second mid-recording sample. The active region bounding box
and a downscaled 64x64 motion fingerprint enable pairwise comparison of
camera framing across patients.

Per-patient representatives (longest usable folder) are compared via:
  - Bounding box IoU (overlap of active regions)
  - Cosine similarity of 64x64 motion fingerprints (spatial distribution)

Patients are ranked by mean similarity and clustered into groups with
comparable framing, producing a self-contained HTML report.

Usage:
    python scripts/05_framing_comparison.py                    # default
    python scripts/05_framing_comparison.py --patient EM1337   # single patient test
    python scripts/05_framing_comparison.py --parallel 4       # workers (default 4)
    python scripts/05_framing_comparison.py --dry-run          # show plan only
    python scripts/05_framing_comparison.py --resume           # skip computed folders
    python scripts/05_framing_comparison.py --threshold 0.5    # similarity threshold
"""

import argparse
import base64
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
FRAMING_DIR = OUTPUT_DIR / "framing_comparison"
DETAIL_DIR = OUTPUT_DIR / "detail_review"
OUTPUT_HTML = OUTPUT_DIR / "framing_comparison.html"

DEFAULT_PARALLEL = 4
MOTION_SAMPLE_FRAMES = 1800   # ~60 sec at 30 fps
FINGERPRINT_SIZE = 64          # downscale motion map to 64x64
ACTIVE_THRESHOLD_PCT = 0.30    # top 30% of max intensity = active
MAX_FRAME_WIDTH = 640
THUMB_QUALITY = 85
DEFAULT_SIMILARITY_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Data loading (reuses patterns from 04)
# ---------------------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_triage(triage_path: Path, patient_filter: str | None = None) -> pd.DataFrame:
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
                "file_size_bytes": mrow.get("file_size_bytes", 0),
                "is_probed": bool(mrow.get("is_probed", False)),
            })
            cumulative += dur

        probed = folder_df[folder_df["is_probed"] == True]
        if not probed.empty:
            p = probed.iloc[0]
            width = int(p["width"])
            height = int(p["height"])
            fps = float(p["fps"])
        else:
            width, height, fps = 0, 0, 30.0

        tasks.append({
            "patient_id": pid,
            "uuid": uuid,
            "folder_name": folder_name,
            "avi_list": avi_list,
            "total_duration": cumulative,
            "avi_count": len(folder_df),
            "width": width,
            "height": height,
            "fps": fps,
        })

    return tasks


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
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Motion profile computation
# ---------------------------------------------------------------------------
REP_CANDIDATE_COUNT = 8    # sample this many frames across the recording


def _grab_single_frame(
    folder_path: Path, avi_list: list[dict], timestamp: float, fps: float,
) -> tuple[np.ndarray | None, float]:
    """Read one frame at a global timestamp. Returns (frame, mean_brightness)."""
    avi_name, local_offset = find_avi_for_timestamp(avi_list, timestamp)
    if not avi_name:
        return None, 0.0
    avi_path = folder_path / avi_name
    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return None, 0.0
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = max(0, min(int(local_offset * actual_fps), frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None, 0.0
        brightness = float(np.mean(frame))
        return frame, brightness
    except Exception:
        return None, 0.0


def _find_brightest_representative(
    folder_path: Path, task: dict,
) -> np.ndarray | None:
    """Sample frames across the full recording and return the brightest one."""
    total = task["total_duration"]
    fps = task["fps"]
    avi_list = task["avi_list"]

    # Space candidates evenly, avoiding the very start/end
    margin = total * 0.05
    span = total - 2 * margin
    if span <= 0:
        timestamps = [total / 2]
    else:
        n = REP_CANDIDATE_COUNT
        timestamps = [margin + span * i / max(n - 1, 1) for i in range(n)]

    best_frame = None
    best_brightness = -1.0
    for ts in timestamps:
        frame, brightness = _grab_single_frame(folder_path, avi_list, ts, fps)
        if frame is not None and brightness > best_brightness:
            best_brightness = brightness
            best_frame = frame

    return best_frame


def compute_motion_profile(
    data_root: str,
    task: dict,
    output_dir: Path,
) -> dict:
    """
    Accumulate frame diffs from mid-recording → motion map.
    Returns bounding box, 64x64 fingerprint, and representative frame
    (brightest frame sampled across the full recording).
    """
    pid = task["patient_id"]
    uuid = task["uuid"]
    folder_path = Path(data_root) / task["folder_name"]

    mid_time = task["total_duration"] / 2
    avi_name, local_offset = find_avi_for_timestamp(task["avi_list"], mid_time)
    if not avi_name:
        return _profile_fail("no AVI for mid-point")

    avi_path = folder_path / avi_name
    try:
        cap = cv2.VideoCapture(str(avi_path))
        if not cap.isOpened():
            return _profile_fail(f"cannot open {avi_name}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or task["fps"]

        start = int(local_offset * fps) - MOTION_SAMPLE_FRAMES // 2
        start = max(0, min(start, frame_count - MOTION_SAMPLE_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        ret, prev = cap.read()
        if not ret or prev is None:
            cap.release()
            return _profile_fail("cannot read first frame")

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
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
            return _profile_fail(f"only {frames_read} frames")

        # Threshold → binary active mask (top 30% of max intensity)
        mx = accum.max()
        if mx == 0:
            return _profile_fail("no motion detected")

        threshold = mx * ACTIVE_THRESHOLD_PCT
        active_mask = accum >= threshold

        # Bounding box of active region (normalized 0–1)
        rows = np.any(active_mask, axis=1)
        cols = np.any(active_mask, axis=0)
        if not rows.any() or not cols.any():
            return _profile_fail("empty active region")

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = {
            "x1": float(cmin / w),
            "y1": float(rmin / h),
            "x2": float((cmax + 1) / w),
            "y2": float((rmax + 1) / h),
        }

        # Active region stats
        active_area = float(np.sum(active_mask)) / (h * w)
        center_x = (bbox["x1"] + bbox["x2"]) / 2
        center_y = (bbox["y1"] + bbox["y2"]) / 2

        # 64x64 fingerprint (downscaled accumulator, normalized)
        norm = (accum / mx * 255).astype(np.uint8)
        fingerprint = cv2.resize(
            norm, (FINGERPRINT_SIZE, FINGERPRINT_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # Find brightest representative frame across the full recording
        rep_frame = _find_brightest_representative(folder_path, task)
        if rep_frame is None:
            # Fallback: use first frame from motion sample AVI
            cap2 = cv2.VideoCapture(str(avi_path))
            if cap2.isOpened():
                cap2.set(cv2.CAP_PROP_POS_FRAMES, start)
                ret2, rep_frame = cap2.read()
                cap2.release()
                if not ret2:
                    rep_frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                rep_frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Save outputs
        out_dir = output_dir / pid
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save representative frame
        rep_resized = resize_frame(rep_frame, MAX_FRAME_WIDTH)
        rep_path = out_dir / f"{uuid}_rep.jpg"
        cv2.imwrite(str(rep_path), rep_resized,
                     [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])

        # Save profile data (fingerprint + bbox)
        npz_path = out_dir / f"{uuid}_profile.npz"
        np.savez_compressed(
            str(npz_path),
            fingerprint=fingerprint,
            bbox=np.array([bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]),
        )

        return {
            "success": True,
            "error": "",
            "patient_id": pid,
            "uuid": uuid,
            "folder_name": task["folder_name"],
            "total_duration": task["total_duration"],
            "width": task["width"],
            "height": task["height"],
            "bbox": bbox,
            "active_area_pct": active_area * 100,
            "center_x": center_x,
            "center_y": center_y,
            "fingerprint": fingerprint,
            "rep_frame_path": str(rep_path),
            "npz_path": str(npz_path),
            "frames_read": frames_read,
        }

    except Exception as e:
        return _profile_fail(str(e))


def _profile_fail(msg: str) -> dict:
    return {
        "success": False,
        "error": msg,
        "patient_id": "",
        "uuid": "",
        "folder_name": "",
        "total_duration": 0,
        "width": 0,
        "height": 0,
        "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
        "active_area_pct": 0,
        "center_x": 0,
        "center_y": 0,
        "fingerprint": np.zeros((FINGERPRINT_SIZE, FINGERPRINT_SIZE), dtype=np.uint8),
        "rep_frame_path": "",
        "npz_path": "",
        "frames_read": 0,
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def find_completed_uuids(framing_dir: Path) -> set[str]:
    completed: set[str] = set()
    if not framing_dir.exists():
        return completed
    for pdir in framing_dir.iterdir():
        if not pdir.is_dir():
            continue
        for f in pdir.iterdir():
            if f.suffix == ".npz" and f.name.endswith("_profile.npz"):
                completed.add(f.stem.replace("_profile", ""))
    return completed


def load_existing_profile(npz_path: str, task: dict) -> dict:
    """Reconstruct a profile result from a saved .npz file."""
    data = np.load(npz_path)
    fingerprint = data["fingerprint"]
    bbox_arr = data["bbox"]
    bbox = {
        "x1": float(bbox_arr[0]),
        "y1": float(bbox_arr[1]),
        "x2": float(bbox_arr[2]),
        "y2": float(bbox_arr[3]),
    }
    active_area = (bbox["x2"] - bbox["x1"]) * (bbox["y2"] - bbox["y1"]) * 100
    center_x = (bbox["x1"] + bbox["x2"]) / 2
    center_y = (bbox["y1"] + bbox["y2"]) / 2

    rep_path = str(Path(npz_path).parent / f"{task['uuid']}_rep.jpg")

    return {
        "success": True,
        "error": "",
        "patient_id": task["patient_id"],
        "uuid": task["uuid"],
        "folder_name": task["folder_name"],
        "total_duration": task["total_duration"],
        "width": task["width"],
        "height": task["height"],
        "bbox": bbox,
        "active_area_pct": active_area,
        "center_x": center_x,
        "center_y": center_y,
        "fingerprint": fingerprint,
        "rep_frame_path": rep_path,
        "npz_path": npz_path,
        "frames_read": 0,
    }


# ---------------------------------------------------------------------------
# Per-patient representative selection
# ---------------------------------------------------------------------------
def pick_representatives(profiles: list[dict]) -> dict[str, dict]:
    """For each patient, pick the longest-duration usable folder."""
    by_patient: dict[str, list[dict]] = {}
    for p in profiles:
        if p["success"]:
            by_patient.setdefault(p["patient_id"], []).append(p)

    reps = {}
    for pid, folder_profiles in by_patient.items():
        reps[pid] = max(folder_profiles, key=lambda p: p["total_duration"])

    return reps


# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------
def compute_iou(bbox_a: dict, bbox_b: dict) -> float:
    """Intersection-over-union of two normalized bounding boxes."""
    x1 = max(bbox_a["x1"], bbox_b["x1"])
    y1 = max(bbox_a["y1"], bbox_b["y1"])
    x2 = min(bbox_a["x2"], bbox_b["x2"])
    y2 = min(bbox_a["y2"], bbox_b["y2"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = (bbox_a["x2"] - bbox_a["x1"]) * (bbox_a["y2"] - bbox_a["y1"])
    area_b = (bbox_b["x2"] - bbox_b["x1"]) * (bbox_b["y2"] - bbox_b["y1"])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def compute_cosine_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Cosine similarity of flattened fingerprint vectors."""
    a = fp_a.flatten().astype(np.float64)
    b = fp_b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarity_matrix(
    reps: dict[str, dict],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise IoU, cosine, and combined similarity matrices.
    Returns (patient_ids, iou_matrix, cosine_matrix, combined_matrix).
    """
    pids = sorted(reps.keys())
    n = len(pids)
    iou_mat = np.zeros((n, n))
    cos_mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                iou_mat[i, j] = 1.0
                cos_mat[i, j] = 1.0
            elif j > i:
                iou_val = compute_iou(reps[pids[i]]["bbox"], reps[pids[j]]["bbox"])
                cos_val = compute_cosine_similarity(
                    reps[pids[i]]["fingerprint"], reps[pids[j]]["fingerprint"])
                iou_mat[i, j] = iou_val
                iou_mat[j, i] = iou_val
                cos_mat[i, j] = cos_val
                cos_mat[j, i] = cos_val

    combined = (iou_mat + cos_mat) / 2
    return pids, iou_mat, cos_mat, combined


# ---------------------------------------------------------------------------
# Ranking and clustering
# ---------------------------------------------------------------------------
def rank_and_cluster(
    pids: list[str],
    combined: np.ndarray,
    threshold: float,
) -> tuple[list[str], list[list[str]]]:
    """
    Rank patients by mean similarity to all others.
    Greedy clustering: start with most typical, add if combined > threshold.
    Returns (ranked_pids, clusters).
    """
    n = len(pids)
    # Mean similarity to all others (exclude self)
    mean_sim = np.array([
        (combined[i].sum() - 1.0) / max(n - 1, 1) for i in range(n)
    ])
    ranked_indices = np.argsort(-mean_sim)
    ranked_pids = [pids[i] for i in ranked_indices]

    # Greedy clustering
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    assigned = set()
    clusters = []

    for pid in ranked_pids:
        if pid in assigned:
            continue
        cluster = [pid]
        assigned.add(pid)
        idx = pid_to_idx[pid]

        for other_pid in ranked_pids:
            if other_pid in assigned:
                continue
            other_idx = pid_to_idx[other_pid]
            # Check similarity to ALL current cluster members
            if all(combined[pid_to_idx[cp], other_idx] >= threshold
                   for cp in cluster):
                cluster.append(other_pid)
                assigned.add(other_pid)

        clusters.append(cluster)

    return ranked_pids, clusters


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------
def write_framing_metrics(profiles: list[dict], path: Path):
    """Write per-folder framing metrics CSV."""
    rows = []
    for p in profiles:
        if not p["success"]:
            continue
        rows.append({
            "patient_id": p["patient_id"],
            "uuid": p["uuid"],
            "folder_name": p["folder_name"],
            "total_duration": p["total_duration"],
            "width": p["width"],
            "height": p["height"],
            "bbox_x1": p["bbox"]["x1"],
            "bbox_y1": p["bbox"]["y1"],
            "bbox_x2": p["bbox"]["x2"],
            "bbox_y2": p["bbox"]["y2"],
            "active_area_pct": p["active_area_pct"],
            "center_x": p["center_x"],
            "center_y": p["center_y"],
            "frames_read": p["frames_read"],
        })
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


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def fmt_dur(sec: float) -> str:
    if sec != sec:
        return "?"
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    return f"{h}h {m}m" if h else f"{m}m"


# ---------------------------------------------------------------------------
# HTML: draw bounding box on representative frame
# ---------------------------------------------------------------------------
def create_bbox_overlay_image(rep_frame_path: str, bbox: dict) -> str:
    """Load rep frame, draw bounding box, return as base64 JPEG."""
    img = cv2.imread(rep_frame_path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    x1 = int(bbox["x1"] * w)
    y1 = int(bbox["y1"] * h)
    x2 = int(bbox["x2"] * w)
    y2 = int(bbox["y2"] * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('ascii')}"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
# Distinct colors for bounding box overlay (20 colors)
OVERLAY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def generate_html(
    reps: dict[str, dict],
    all_profiles: list[dict],
    pids: list[str],
    iou_mat: np.ndarray,
    cos_mat: np.ndarray,
    combined: np.ndarray,
    ranked_pids: list[str],
    clusters: list[list[str]],
    electrode_counts: dict[str, int],
    threshold: float,
) -> str:
    """Build self-contained HTML comparison report."""
    n = len(pids)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    # Cluster membership lookup
    pid_to_cluster = {}
    for ci, cluster in enumerate(clusters):
        for pid in cluster:
            pid_to_cluster[pid] = ci

    # Sort pids by cluster membership then rank for matrix display
    sorted_pids = []
    for cluster in clusters:
        sorted_pids.extend(cluster)

    # Folder counts per patient
    folder_counts: dict[str, int] = {}
    for p in all_profiles:
        if p["success"]:
            folder_counts[p["patient_id"]] = folder_counts.get(p["patient_id"], 0) + 1

    # ================================================================
    # Section 1: Active region overview — patient cards
    # ================================================================
    card_html_parts = []
    for pid in sorted_pids:
        rep = reps[pid]
        ci = pid_to_cluster.get(pid, 0)
        border_color = OVERLAY_COLORS[ci % len(OVERLAY_COLORS)]
        n_elec = electrode_counts.get(pid, 0)
        n_folders = folder_counts.get(pid, 0)
        bbox_img = create_bbox_overlay_image(rep["rep_frame_path"], rep["bbox"])
        area_pct = rep["active_area_pct"]
        cx = rep["center_x"]
        cy = rep["center_y"]

        img_tag = (f'<img src="{bbox_img}" loading="lazy">'
                   if bbox_img else '<div class="no-img">no frame</div>')

        card_html_parts.append(f"""
        <div class="patient-card" style="border-color: {border_color};"
             data-patient="{pid}" data-cluster="{ci}">
          <div class="pc-img">{img_tag}</div>
          <div class="pc-info">
            <div class="pc-pid">{pid}</div>
            <div class="pc-stats">
              {n_elec} electrodes &middot; {n_folders} folder{'s' if n_folders != 1 else ''}
              &middot; {fmt_dur(rep['total_duration'])}
            </div>
            <div class="pc-region">
              Active: {area_pct:.1f}% of frame &middot;
              Center: ({cx:.2f}, {cy:.2f})
            </div>
          </div>
          <div class="pc-cluster" style="background: {border_color};">
            Cluster {ci + 1}
          </div>
        </div>""")

    # ================================================================
    # Section 2: Similarity matrix
    # ================================================================
    # Build heatmap table sorted by cluster
    matrix_header = '<th></th>' + ''.join(
        f'<th class="matrix-pid" title="{pid}">{pid}</th>'
        for pid in sorted_pids
    )

    matrix_rows = []
    for pid_row in sorted_pids:
        i = pid_to_idx[pid_row]
        cells = []
        for pid_col in sorted_pids:
            j = pid_to_idx[pid_col]
            val = combined[i, j]
            iou_val = iou_mat[i, j]
            cos_val = cos_mat[i, j]
            if i == j:
                bg = "#e8e8e8"
                cells.append(f'<td class="matrix-cell diag" style="background:{bg};" '
                             f'title="{pid_row}">—</td>')
            else:
                # Green (similar) → Red (different)
                r = int(255 * (1 - val))
                g = int(200 * val)
                b = int(50 * val)
                bg = f"rgb({r},{g},{b})"
                fg = "#fff" if val < 0.5 else "#000"
                cells.append(
                    f'<td class="matrix-cell" style="background:{bg};color:{fg};" '
                    f'title="{pid_row} vs {pid_col}: IoU={iou_val:.2f}, '
                    f'Cos={cos_val:.2f}, Combined={val:.2f}" '
                    f'data-row="{pid_row}" data-col="{pid_col}">'
                    f'{val:.2f}</td>')

        matrix_rows.append(
            f'<tr><th class="matrix-pid">{pid_row}</th>{"".join(cells)}</tr>')

    # ================================================================
    # Section 3: Bounding box overlay (SVG)
    # ================================================================
    # SVG uses viewBox "0 0 100 56.25" (16:9 aspect ratio matching 1280x720)
    # All coordinates are absolute viewBox units (not percentages)
    SVG_W = 100
    SVG_H = 56.25
    svg_rects = []
    for idx, pid in enumerate(sorted_pids):
        rep = reps[pid]
        bbox = rep["bbox"]
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        x = bbox["x1"] * SVG_W
        y = bbox["y1"] * SVG_H
        w = (bbox["x2"] - bbox["x1"]) * SVG_W
        h = (bbox["y2"] - bbox["y1"]) * SVG_H
        svg_rects.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{color}" fill-opacity="0.15" stroke="{color}" stroke-width="0.5" '
            f'stroke-opacity="0.8" data-pid="{pid}"/>')

    svg_labels = []
    for idx, pid in enumerate(sorted_pids):
        rep = reps[pid]
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        cx = rep["center_x"] * SVG_W
        cy = rep["center_y"] * SVG_H
        svg_labels.append(
            f'<text x="{cx:.2f}" y="{cy:.2f}" fill="{color}" '
            f'font-size="3" text-anchor="middle" font-weight="bold" '
            f'stroke="#000" stroke-width="0.1">{pid[-4:]}</text>')

    # Legend for overlay
    legend_items = []
    for idx, pid in enumerate(sorted_pids):
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        legend_items.append(
            f'<span class="legend-item">'
            f'<span class="legend-color" style="background:{color};"></span>'
            f'{pid}</span>')

    # ================================================================
    # Section 4: Recommended subset
    # ================================================================
    # Largest cluster
    largest_cluster = max(clusters, key=len) if clusters else []
    subset_cards = []
    for pid in largest_cluster:
        rep = reps[pid]
        n_elec = electrode_counts.get(pid, 0)
        bbox_img = create_bbox_overlay_image(rep["rep_frame_path"], rep["bbox"])
        img_tag = (f'<img src="{bbox_img}" loading="lazy">'
                   if bbox_img else '<div class="no-img">no frame</div>')

        # Mean similarity to rest of cluster
        idx = pid_to_idx[pid]
        cluster_sims = [combined[idx, pid_to_idx[other]]
                        for other in largest_cluster if other != pid]
        mean_sim = np.mean(cluster_sims) if cluster_sims else 1.0

        subset_cards.append(f"""
        <div class="subset-card" data-patient="{pid}">
          <div class="sc-img">{img_tag}</div>
          <div class="sc-info">
            <span class="sc-pid">{pid}</span>
            <span class="sc-detail">{n_elec} electrodes &middot; {fmt_dur(rep['total_duration'])}</span>
            <span class="sc-sim">Cluster sim: {mean_sim:.2f}</span>
          </div>
          <label class="sc-toggle">
            <input type="checkbox" checked class="cb-include"
                   data-patient="{pid}" onchange="onSubsetChange()">
            Include
          </label>
        </div>""")

    # Stats for header
    total_patients = len(pids)
    total_profiles = sum(1 for p in all_profiles if p["success"])
    largest_cluster_size = len(largest_cluster)

    # ================================================================
    # Assemble HTML
    # ================================================================
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Framing Comparison — M1 Pipeline Phase 3</title>
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

  /* ---- Layout ---- */
  .layout {{ display: flex; min-height: 100vh; }}
  .sidebar {{
    width: 180px; background: #fff; border-right: 1px solid #ddd;
    position: sticky; top: 70px; height: calc(100vh - 70px);
    overflow-y: auto; flex-shrink: 0; padding: 8px 0;
  }}
  .nav-item {{
    display: block; padding: 8px 16px; text-decoration: none; color: #333;
    font-size: 13px; font-weight: 500; border-left: 3px solid transparent;
  }}
  .nav-item:hover {{ background: #f0f0f0; }}
  .nav-item.active {{ border-left-color: #1a1a2e; background: #f0f0f0; font-weight: 700; }}
  .main {{ flex: 1; padding: 16px 24px; max-width: 1400px; }}

  /* ---- Section titles ---- */
  .section-title {{
    font-size: 20px; font-weight: 700; color: #1a1a2e;
    border-bottom: 3px solid #1a1a2e; padding-bottom: 4px;
    margin: 32px 0 16px;
  }}
  .section-title:first-child {{ margin-top: 0; }}
  .section-desc {{ font-size: 13px; color: #777; margin-bottom: 16px; }}

  /* ---- Section 1: Patient cards grid ---- */
  .cards-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px; margin-bottom: 24px;
  }}
  .patient-card {{
    background: #fff; border: 2px solid #ddd; border-radius: 8px;
    overflow: hidden; transition: box-shadow 0.2s;
    position: relative;
  }}
  .patient-card:hover {{ box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
  .pc-img {{ position: relative; }}
  .pc-img img {{ width: 100%; height: auto; display: block; }}
  .no-img {{
    width: 100%; height: 120px; background: #f0f0f0;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; color: #bbb;
  }}
  .pc-info {{ padding: 8px 12px; }}
  .pc-pid {{ font-size: 16px; font-weight: 700; color: #1a1a2e; }}
  .pc-stats {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .pc-region {{ font-size: 11px; color: #666; margin-top: 4px; font-family: monospace; }}
  .pc-cluster {{
    position: absolute; top: 8px; right: 8px;
    color: #fff; font-size: 10px; font-weight: 700;
    padding: 2px 8px; border-radius: 3px;
  }}

  /* ---- Section 2: Similarity matrix ---- */
  .matrix-container {{ overflow-x: auto; margin-bottom: 24px; }}
  .matrix-table {{
    border-collapse: collapse; font-size: 11px;
    margin: 0 auto;
  }}
  .matrix-table th, .matrix-table td {{
    padding: 4px 6px; text-align: center;
    border: 1px solid #e0e0e0;
  }}
  .matrix-pid {{
    font-weight: 700; font-size: 10px; color: #333;
    white-space: nowrap; background: #f8f8f8;
  }}
  .matrix-cell {{
    cursor: default; font-family: monospace; font-size: 10px;
    min-width: 40px;
  }}
  .matrix-cell.diag {{ font-size: 10px; color: #999; }}
  .matrix-legend {{
    text-align: center; margin-top: 8px; font-size: 12px; color: #888;
  }}

  /* ---- Section 3: Overlay ---- */
  .overlay-container {{
    background: #1a1a2e; border-radius: 8px; padding: 16px;
    margin-bottom: 16px;
  }}
  .overlay-svg {{
    width: 100%; max-width: 800px; margin: 0 auto; display: block;
    background: #2a2a3e; border-radius: 4px;
  }}
  .overlay-legend {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
    justify-content: center;
  }}
  .legend-item {{
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; color: #ccc;
  }}
  .legend-color {{
    width: 12px; height: 12px; border-radius: 2px; display: inline-block;
  }}

  /* ---- Section 4: Recommended subset ---- */
  .subset-header {{
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 10px; margin-bottom: 12px;
  }}
  .subset-count {{
    font-size: 15px; font-weight: 600; color: #4caf50;
  }}
  .btn {{
    padding: 7px 16px; border: none; border-radius: 4px;
    font-size: 13px; font-weight: 600; cursor: pointer;
  }}
  .btn-export {{ background: #4caf50; color: #fff; }}
  .btn-export:hover {{ background: #43a047; }}
  .subset-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 10px;
  }}
  .subset-card {{
    display: flex; align-items: center; gap: 10px;
    background: #fff; border: 1px solid #ddd; border-radius: 6px;
    padding: 8px 12px;
  }}
  .sc-img {{ flex-shrink: 0; }}
  .sc-img img {{ width: 120px; height: auto; border-radius: 3px; }}
  .sc-info {{ flex: 1; }}
  .sc-pid {{ font-weight: 700; color: #1a1a2e; display: block; }}
  .sc-detail {{ font-size: 12px; color: #888; display: block; }}
  .sc-sim {{ font-size: 11px; color: #666; font-family: monospace; display: block; }}
  .sc-toggle {{
    display: flex; align-items: center; gap: 4px;
    font-size: 12px; font-weight: 600; color: #555; cursor: pointer;
    white-space: nowrap;
  }}
  .sc-toggle input {{ accent-color: #4caf50; width: 16px; height: 16px; cursor: pointer; }}

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
    .cards-grid {{ grid-template-columns: 1fr; }}
    .subset-grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <div class="header-top">
      <div>
        <h1>Camera Framing Comparison &mdash; M1 Pipeline Phase 3</h1>
        <div class="header-stats">
          {total_patients} patients &middot; {total_profiles} folder profiles &middot;
          Largest cluster: {largest_cluster_size} patients &middot;
          Threshold: {threshold:.2f}
        </div>
      </div>
    </div>
  </div>

  <div class="layout">
    <div class="sidebar">
      <a href="#sec-overview" class="nav-item active">Active Regions</a>
      <a href="#sec-matrix" class="nav-item">Similarity Matrix</a>
      <a href="#sec-overlay" class="nav-item">Region Overlay</a>
      <a href="#sec-subset" class="nav-item">Recommended Subset</a>
    </div>
    <div class="main">

      <!-- Section 1: Active region overview -->
      <div class="section-title" id="sec-overview">Active Region Overview</div>
      <div class="section-desc">
        Each card shows the representative frame with the detected active region
        (green bounding box). Border color indicates cluster membership.
        Sorted by cluster: similar framing is grouped together.
      </div>
      <div class="cards-grid">
        {''.join(card_html_parts)}
      </div>

      <!-- Section 2: Similarity matrix -->
      <div class="section-title" id="sec-matrix">Pairwise Similarity Matrix</div>
      <div class="section-desc">
        Combined score = average of bounding box IoU and 64&times;64 motion
        fingerprint cosine similarity. Green = similar framing, red = different.
        Hover cells for breakdowns.
      </div>
      <div class="matrix-container">
        <table class="matrix-table">
          <thead><tr>{matrix_header}</tr></thead>
          <tbody>{''.join(matrix_rows)}</tbody>
        </table>
      </div>
      <div class="matrix-legend">
        <span style="color:#c62828;">&#9632;</span> 0.0 (different) &nbsp;&mdash;&nbsp;
        <span style="color:#388e3c;">&#9632;</span> 1.0 (identical)
      </div>

      <!-- Section 3: Bounding box overlay -->
      <div class="section-title" id="sec-overlay">Active Region Overlay</div>
      <div class="section-desc">
        All patients' active-region bounding boxes overlaid on a normalized canvas.
        Shows at a glance how much camera framing varies.
      </div>
      <div class="overlay-container">
        <svg class="overlay-svg" viewBox="0 0 100 56.25" preserveAspectRatio="xMidYMid meet">
          <rect x="0" y="0" width="100" height="56.25" fill="#2a2a3e"/>
          {''.join(svg_rects)}
          {''.join(svg_labels)}
        </svg>
        <div class="overlay-legend">
          {''.join(legend_items)}
        </div>
      </div>

      <!-- Section 4: Recommended subset -->
      <div class="section-title" id="sec-subset">Recommended Subset</div>
      <div class="section-desc">
        Largest cluster of patients with mutually similar framing
        (combined similarity &ge; {threshold:.2f}).
        Use checkboxes to include/exclude patients, then export.
      </div>
      <div class="subset-header">
        <span class="subset-count" id="subset-count">
          {largest_cluster_size} / {largest_cluster_size} patients selected
        </span>
        <button class="btn btn-export" onclick="exportSubset()">Export Selected</button>
      </div>
      <div class="subset-grid">
        {''.join(subset_cards)}
      </div>

    </div>
  </div>

  <div class="toast" id="toast"></div>

<script>
// ---- Sidebar active state ----
const sections = document.querySelectorAll('.section-title[id]');
const navLinks = document.querySelectorAll('.nav-item');
window.addEventListener('scroll', () => {{
  let current = '';
  sections.forEach(sec => {{
    if (window.scrollY >= sec.offsetTop - 120) current = sec.id;
  }});
  navLinks.forEach(link => {{
    link.classList.toggle('active', link.getAttribute('href') === '#' + current);
  }});
}});

// ---- Subset export ----
function onSubsetChange() {{
  const boxes = document.querySelectorAll('.cb-include');
  const checked = Array.from(boxes).filter(b => b.checked).length;
  const total = boxes.length;
  document.getElementById('subset-count').textContent =
    checked + ' / ' + total + ' patients selected';
}}

function exportSubset() {{
  const boxes = document.querySelectorAll('.cb-include');
  const selected = Array.from(boxes)
    .filter(b => b.checked)
    .map(b => b.dataset.patient);

  if (selected.length === 0) {{
    showToast('No patients selected');
    return;
  }}

  const rows = ['patient_id'];
  selected.forEach(pid => rows.push(pid));
  const csv = rows.join('\\n');
  const blob = new Blob([csv], {{ type: 'text/csv' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'framing_subset.csv';
  a.click();
  URL.revokeObjectURL(url);
  showToast('Exported ' + selected.length + ' patients to framing_subset.csv');
}}

// ---- Toast ----
function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2500);
}}

// ---- Keyboard: Ctrl+E to export ----
document.addEventListener('keydown', e => {{
  if ((e.ctrlKey || e.metaKey) && e.key === 'e') {{
    e.preventDefault();
    exportSubset();
  }}
}});
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-patient camera framing comparison",
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
        "--patient", type=str, default=None, metavar="EM_ID",
        help="Process a single patient (for testing)",
    )
    parser.add_argument(
        "--parallel", type=int, default=DEFAULT_PARALLEL, metavar="N",
        help=f"Number of concurrent workers (default: {DEFAULT_PARALLEL})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without processing anything",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folders that already have profiles",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
        help=f"Similarity threshold for clustering (default: {DEFAULT_SIMILARITY_THRESHOLD})",
    )
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Override data_root from config.yaml",
    )
    args = parser.parse_args()

    triage_path = Path(args.triage)
    metadata_path = Path(args.metadata)

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

    # Track existing profiles for resume and HTML rebuild
    existing_profiles: list[dict] = []

    if args.resume:
        done_uuids = find_completed_uuids(FRAMING_DIR)
        new_tasks = []
        for t in tasks:
            if t["uuid"] in done_uuids:
                # Load existing profile
                npz_path = str(FRAMING_DIR / t["patient_id"] / f"{t['uuid']}_profile.npz")
                try:
                    existing_profiles.append(load_existing_profile(npz_path, t))
                except Exception as e:
                    print(f"  WARNING: cannot load {npz_path}: {e}", file=sys.stderr)
                    new_tasks.append(t)
            else:
                new_tasks.append(t)
        print(f"  Resume: {len(tasks) - len(new_tasks)} cached, "
              f"{len(new_tasks)} remaining")
        tasks = new_tasks

    n_patients = len(set(t["patient_id"] for t in tasks))
    total_dur_h = sum(t["total_duration"] for t in tasks) / 3600

    # 3. Summary
    print(f"\n{'=' * 60}")
    print(f"Framing comparison plan:")
    print(f"  Patients to process:  {n_patients}")
    print(f"  Folders to process:   {len(tasks)}")
    print(f"  Total duration:       ~{total_dur_h:.0f}h of recording")
    print(f"  Cached profiles:      {len(existing_profiles)}")
    print(f"  Motion sample:        {MOTION_SAMPLE_FRAMES} frames (~60s)")
    print(f"  Fingerprint size:     {FINGERPRINT_SIZE}x{FINGERPRINT_SIZE}")
    print(f"  Parallel:             {args.parallel} workers")
    print(f"  Threshold:            {args.threshold}")
    print(f"  Output:               {FRAMING_DIR}")
    print(f"{'=' * 60}")

    if args.dry_run:
        by_p: dict[str, list] = {}
        for t in tasks:
            by_p.setdefault(t["patient_id"], []).append(t)
        print(f"\n[DRY RUN] Per-patient breakdown:")
        for pid in sorted(by_p):
            fl = by_p[pid]
            print(f"  {pid}: {len(fl)} folders to profile")
        est_sec = len(tasks) * 10 / args.parallel
        print(f"\nEstimated time: ~{est_sec / 60:.0f} min with {args.parallel} workers")
        print(f"Remove --dry-run to start.")
        return

    # 4. Process folders in parallel
    all_profiles = list(existing_profiles)  # start with cached

    if tasks:
        print(f"\nComputing motion profiles ...")
        FRAMING_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        done_count = 0
        err_count = 0

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(compute_motion_profile, data_root, task, FRAMING_DIR): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                done_count += 1
                try:
                    result = future.result()
                    if result["success"]:
                        all_profiles.append(result)
                    else:
                        err_count += 1
                        print(f"  SKIP [{task['patient_id']}/{task['uuid']}]: "
                              f"{result['error']}", file=sys.stderr)
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
        print(f"\nProfile computation complete in {elapsed:.1f}s")
        if err_count:
            print(f"  {err_count} errors (check stderr)")

    # 5. Write metrics CSV
    metrics_path = FRAMING_DIR / "framing_metrics.csv"
    write_framing_metrics(all_profiles, metrics_path)
    print(f"Wrote {metrics_path}")

    # 6. Pick per-patient representatives
    successful = [p for p in all_profiles if p["success"]]
    if not successful:
        print("No successful profiles. Cannot generate comparison.")
        return

    reps = pick_representatives(successful)
    print(f"\n{len(reps)} patients with representative profiles")

    if len(reps) < 2:
        print("Need at least 2 patients for comparison. Use --patient to test single.")
        # Still generate HTML for the single patient
        if len(reps) == 1:
            pid = list(reps.keys())[0]
            pids = [pid]
            iou_mat = np.array([[1.0]])
            cos_mat = np.array([[1.0]])
            combined = np.array([[1.0]])
            ranked_pids = [pid]
            clusters = [[pid]]
        else:
            return
    else:
        # 7. Compute similarity matrix
        pids, iou_mat, cos_mat, combined = compute_similarity_matrix(reps)
        print(f"\nSimilarity matrix ({len(pids)}x{len(pids)}) computed")

        # 8. Rank and cluster
        ranked_pids, clusters = rank_and_cluster(pids, combined, args.threshold)

        print(f"\nClusters (threshold={args.threshold}):")
        for ci, cluster in enumerate(clusters):
            print(f"  Cluster {ci + 1}: {', '.join(cluster)} ({len(cluster)} patients)")

        largest = max(clusters, key=len)
        print(f"\nLargest cluster: {len(largest)} patients")

    # 9. Generate HTML
    print(f"\nGenerating HTML report ...")
    t0 = time.time()
    html = generate_html(
        reps, all_profiles, pids, iou_mat, cos_mat, combined,
        ranked_pids, clusters, electrode_counts, args.threshold,
    )
    elapsed = time.time() - t0
    print(f"  HTML generated in {elapsed:.1f}s")

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = OUTPUT_HTML.stat().st_size / (1024 * 1024)
    print(f"\nWrote {OUTPUT_HTML} ({size_mb:.1f} MB)")

    # 10. Summary
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Profiles computed:    {len(all_profiles)}")
    print(f"  Patients compared:    {len(reps)}")
    print(f"  Clusters found:       {len(clusters)}")
    print(f"  Largest cluster:      {len(max(clusters, key=len))} patients")
    print(f"  Metrics CSV:          {metrics_path}")
    print(f"  HTML report:          {OUTPUT_HTML}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
