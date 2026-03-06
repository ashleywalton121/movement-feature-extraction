"""
Side-by-side video: original video (with skeleton overlay) on left,
5 stacked scrolling time-series plots on right — one per extraction method.

Outputs one MP4 per AVI. Run from the repo venv:
    venv\\Scripts\\python scripts\\visualize_all_methods_video.py
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
import time


# ---------------------------------------------------------------------------
# Skeleton overlay (reused from visualize_video.py)
# ---------------------------------------------------------------------------
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]

KEYPOINT_COLORS = {
    "nose": (255, 0, 255),
    "left_eye": (255, 200, 0),
    "right_eye": (255, 200, 0),
    "left_ear": (255, 200, 0),
    "right_ear": (255, 200, 0),
    "left_shoulder": (0, 255, 0),
    "right_shoulder": (0, 255, 0),
    "left_elbow": (0, 200, 255),
    "right_elbow": (0, 200, 255),
    "left_wrist": (0, 100, 255),
    "right_wrist": (0, 100, 255),
    "left_hip": (255, 100, 0),
    "right_hip": (255, 100, 0),
}

LANDMARK_NAMES = list(KEYPOINT_COLORS.keys())


def draw_skeleton(frame, mp_row):
    """Draw keypoints and skeleton connections on the frame."""
    overlay = frame.copy()
    for name_a, name_b in SKELETON_CONNECTIONS:
        xa, ya = mp_row.get(f"{name_a}_x"), mp_row.get(f"{name_a}_y")
        xb, yb = mp_row.get(f"{name_b}_x"), mp_row.get(f"{name_b}_y")
        vis_a = mp_row.get(f"{name_a}_visibility", 0)
        vis_b = mp_row.get(f"{name_b}_visibility", 0)
        if pd.notna(xa) and pd.notna(xb) and vis_a > 0.3 and vis_b > 0.3:
            cv2.line(overlay, (int(xa), int(ya)), (int(xb), int(yb)),
                     (200, 200, 200), 2)
    for name in LANDMARK_NAMES:
        x, y = mp_row.get(f"{name}_x"), mp_row.get(f"{name}_y")
        vis = mp_row.get(f"{name}_visibility", 0)
        if pd.notna(x) and vis > 0.3:
            color = KEYPOINT_COLORS.get(name, (255, 255, 255))
            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
            cv2.circle(overlay, (int(x), int(y)), 5, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


# ---------------------------------------------------------------------------
# Plot rendering — 5 stacked subplots
# ---------------------------------------------------------------------------
def render_plots(fig, axes, current_time, total_time,
                 df_fd, df_bg, df_mhi, df_dis, df_mp, window_sec=30):
    """Render 5 scrolling time-series subplots for the current time."""
    for ax in axes:
        ax.clear()

    # Scrolling time window
    t_start = max(0, current_time - window_sec / 2)
    t_end = t_start + window_sec
    if t_end > total_time:
        t_end = total_time
        t_start = max(0, t_end - window_sec)

    # --- 1. Frame Differencing ---
    ax = axes[0]
    m = (df_fd["time_sec"] >= t_start) & (df_fd["time_sec"] <= t_end)
    ax.plot(df_fd.loc[m, "time_sec"], df_fd.loc[m, "mean_diff_smooth"],
            color="steelblue", linewidth=1.2)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Frame Diff", fontsize=7)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("Frame Differencing", fontsize=8, loc="left", pad=2)

    # --- 2. Background Subtraction ---
    ax = axes[1]
    m = (df_bg["time_sec"] >= t_start) & (df_bg["time_sec"] <= t_end)
    ax.plot(df_bg.loc[m, "time_sec"], df_bg.loc[m, "foreground_frac_smooth"],
            color="green", linewidth=1.2)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("FG Frac", fontsize=7)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("BG Subtraction", fontsize=8, loc="left", pad=2)

    # --- 3. MEI / MHI ---
    ax = axes[2]
    m = (df_mhi["time_sec"] >= t_start) & (df_mhi["time_sec"] <= t_end)
    ax.plot(df_mhi.loc[m, "time_sec"], df_mhi.loc[m, "mhi_mean_smooth"],
            color="purple", linewidth=1.2)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("MHI Mean", fontsize=7)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("MEI / MHI", fontsize=8, loc="left", pad=2)

    # --- 4. DIS Optical Flow ---
    ax = axes[3]
    m = (df_dis["time_sec"] >= t_start) & (df_dis["time_sec"] <= t_end)
    ax.plot(df_dis.loc[m, "time_sec"], df_dis.loc[m, "mean_mag_smooth"],
            color="darkorange", linewidth=1.2)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Opt Flow", fontsize=7)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title("DIS Optical Flow", fontsize=8, loc="left", pad=2)

    # --- 5. MediaPipe Velocities ---
    ax = axes[4]
    m = (df_mp["time_sec"] >= t_start) & (df_mp["time_sec"] <= t_end)
    mp_sub = df_mp.loc[m]
    if "mean_wrist_velocity" in mp_sub.columns:
        ax.plot(mp_sub["time_sec"],
                mp_sub["mean_wrist_velocity"].rolling(15, center=True, min_periods=1).mean(),
                color="crimson", linewidth=1.2, label="Wrists")
    if "nose_velocity" in mp_sub.columns:
        ax.plot(mp_sub["time_sec"],
                mp_sub["nose_velocity"].rolling(15, center=True, min_periods=1).mean(),
                color="mediumpurple", linewidth=1.2, label="Head")
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Velocity", fontsize=7)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title("MediaPipe Keypoint Velocities", fontsize=8, loc="left", pad=2)

    fig.tight_layout(pad=0.5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    return plot_img


# ---------------------------------------------------------------------------
# Video creation
# ---------------------------------------------------------------------------
def create_side_by_side_video(video_path, csv_paths, output_path,
                               target_fps=15, window_sec=30):
    """Create side-by-side video with skeleton overlay and 5 time-series plots."""
    df_fd = pd.read_csv(csv_paths["frame_diff"])
    df_bg = pd.read_csv(csv_paths["bg_sub"])
    df_mhi = pd.read_csv(csv_paths["mei_mhi"])
    df_dis = pd.read_csv(csv_paths["dis_flow"])
    df_mp = pd.read_csv(csv_paths["mediapipe"])

    # Smooth keypoint positions to reduce skeleton jitter (5-frame rolling average)
    for name in LANDMARK_NAMES:
        for coord in ("x", "y"):
            col = f"{name}_{coord}"
            if col in df_mp.columns:
                df_mp[col] = df_mp[col].rolling(5, center=True, min_periods=1).mean()

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_time = total_frames / src_fps

    frame_skip = max(1, int(round(src_fps / target_fps)))

    # Output dimensions
    out_h = 720
    scale = out_h / vid_h
    scaled_w = int(vid_w * scale)
    plot_w = scaled_w
    out_w = scaled_w + plot_w

    # Matplotlib figure — 5 subplots stacked
    dpi = 100
    fig, axes = plt.subplots(5, 1, figsize=(plot_w / dpi, out_h / dpi), dpi=dpi)
    plt.subplots_adjust(hspace=0.55)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (out_w, out_h))

    frame_idx = -1
    written = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        current_time = frame_idx / src_fps

        # Skeleton overlay
        if frame_idx < len(df_mp):
            mp_row = df_mp.iloc[frame_idx]
            if mp_row.get("pose_detected", False):
                frame = draw_skeleton(frame, mp_row)

        # Timestamp
        cv2.putText(frame, f"t={current_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        video_panel = cv2.resize(frame, (scaled_w, out_h))

        plot_img = render_plots(fig, axes, current_time, total_time,
                                df_fd, df_bg, df_mhi, df_dis, df_mp, window_sec)
        plot_panel = cv2.resize(plot_img, (plot_w, out_h))

        combined = np.hstack([video_panel, plot_panel])
        writer.write(combined)
        written += 1

    cap.release()
    writer.release()
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"  Saved: {output_path.name} ({written} frames, {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Main — iterate over 3 patients × 5 AVIs
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Create 5-method side-by-side visualization videos")
    parser.add_argument("--patients", nargs="+",
                        default=["EM1201", "EM1269", "EM1279"],
                        help="Patient IDs to process")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--max-avis", type=int, default=5,
                        help="Max AVIs per patient (default 5)")
    args = parser.parse_args()

    base = Path(r"C:\Users\ashle\Documents\temp_review\Movement_Decoding")
    output_dir = base / "output" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Method directory name -> CSV suffix
    method_suffixes = {
        "frame_diff": "_frame_diff.csv",
        "bg_sub":     "_bg_sub.csv",       # in bg_subtraction/ folder
        "mei_mhi":    "_mei_mhi.csv",
        "dis_flow":   "_dis_flow.csv",
        "mediapipe":  "_mediapipe.csv",
    }
    # Map short key -> actual folder name under output/{patient}/
    method_folders = {
        "frame_diff": "frame_diff",
        "bg_sub":     "bg_subtraction",
        "mei_mhi":    "mei_mhi",
        "dis_flow":   "dis_flow",
        "mediapipe":  "mediapipe",
    }

    total_created = 0

    for patient_id in args.patients:
        csv_dir = base / "output" / patient_id
        if not csv_dir.exists():
            print(f"Skipping {patient_id} — no output folder")
            continue

        # Discover available stems from frame_diff CSVs (canonical list)
        fd_dir = csv_dir / "frame_diff"
        fd_csvs = sorted(fd_dir.glob("*_frame_diff.csv"))[:args.max_avis]

        if not fd_csvs:
            print(f"Skipping {patient_id} — no frame_diff CSVs found")
            continue

        print(f"\n=== {patient_id}: {len(fd_csvs)} AVIs ===")

        for fd_csv in fd_csvs:
            # Stem: e.g. EM1201_a5b17dea-5440-41ee-bc66-b0136b81bf8b_0180
            stem = fd_csv.stem.replace("_frame_diff", "")

            # Build CSV paths for all 5 methods
            csv_paths = {}
            missing = False
            for method_key, suffix in method_suffixes.items():
                folder = method_folders[method_key]
                csv_path = csv_dir / folder / f"{stem}{suffix}"
                if not csv_path.exists():
                    print(f"  Skipping {stem} — missing {folder} CSV")
                    missing = True
                    break
                csv_paths[method_key] = csv_path
            if missing:
                continue

            # Find AVI: base/{patient}/{stem_subfolder}/{stem}.avi
            # stem contains patient_uuid, subfolder is patient_uuid part
            parts = stem.split("_", 1)  # split into patient_id and rest
            subfolder = parts[1].rsplit("_", 1)[0]  # uuid without index
            avi_subfolder = f"{patient_id}_{subfolder}"
            avi_path = base / patient_id / avi_subfolder / f"{stem}.avi"

            if not avi_path.exists():
                print(f"  Skipping {stem} — AVI not found at {avi_path}")
                continue

            # Extract index for output naming
            avi_index = stem.rsplit("_", 1)[1]
            output_path = output_dir / f"{patient_id}_{avi_index}_5methods.mp4"

            print(f"  Processing {patient_id} clip {avi_index}...")
            create_side_by_side_video(avi_path, csv_paths, output_path,
                                       target_fps=args.target_fps)
            total_created += 1

    print(f"\nDone. Created {total_created} videos in {output_dir}")


if __name__ == "__main__":
    main()
