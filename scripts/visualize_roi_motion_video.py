"""
Side-by-side video: original video with ROI bounding boxes on left,
per-region motion energy time-series on right.

Shows pose-guided ROI bounding boxes (head, left arm, right arm, torso)
overlaid on the video, synced with scrolling motion energy plots.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
import argparse
import time


# MediaPipe pose landmark indices
LANDMARK_INDICES = {
    'nose': 0,
    'left_eye': 2, 'right_eye': 5,
    'left_ear': 7, 'right_ear': 8,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_hip': 23, 'right_hip': 24,
}

# Body region definitions
BODY_REGIONS = {
    'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
    'left_arm': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_arm': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
}

REGION_COLORS_BGR = {
    'head': (60, 76, 231),       # red in BGR
    'left_arm': (219, 152, 52),  # blue in BGR
    'right_arm': (113, 204, 46), # green in BGR
    'torso': (18, 156, 243),     # orange in BGR
}

REGION_COLORS_MPL = {
    'global': '#7f8c8d',
    'head': '#e74c3c',
    'left_arm': '#3498db',
    'right_arm': '#2ecc71',
    'torso': '#f39c12',
}

REGION_LABELS = {
    'global': 'Global',
    'head': 'Head',
    'left_arm': 'Left Arm',
    'right_arm': 'Right Arm',
    'torso': 'Torso',
}

MIN_VISIBILITY = 0.5
PADDING = 30


def get_keypoint_px(pose_row, name, frame_w, frame_h, use_named_cols=False):
    """Get pixel coordinates for a keypoint from a pose row."""
    if use_named_cols:
        x = pose_row.get(f"{name}_x", np.nan)
        y = pose_row.get(f"{name}_y", np.nan)
        vis = pose_row.get(f"{name}_visibility", 0)
    else:
        idx = LANDMARK_INDICES[name]
        x_norm = pose_row.get(f"pose_{idx}_x", np.nan)
        y_norm = pose_row.get(f"pose_{idx}_y", np.nan)
        vis = pose_row.get(f"pose_{idx}_vis", 0)
        if pd.notna(x_norm):
            x = x_norm * frame_w
            y = y_norm * frame_h
        else:
            x, y = np.nan, np.nan

    if pd.isna(x) or pd.isna(y) or vis < MIN_VISIBILITY:
        return None, None
    return float(x), float(y)


def get_region_bbox(pose_row, region, frame_w, frame_h, use_named_cols=False):
    """Compute bounding box for a body region."""
    xs, ys = [], []
    for name in BODY_REGIONS[region]:
        x, y = get_keypoint_px(pose_row, name, frame_w, frame_h, use_named_cols)
        if x is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        return None
    x1 = max(0, int(min(xs) - PADDING))
    y1 = max(0, int(min(ys) - PADDING))
    x2 = min(frame_w, int(max(xs) + PADDING))
    y2 = min(frame_h, int(max(ys) + PADDING))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return (x1, y1, x2, y2)


def draw_roi_boxes(frame, pose_row, frame_w, frame_h, use_named_cols=False):
    """Draw ROI bounding boxes and keypoints on frame."""
    overlay = frame.copy()

    for region, color in REGION_COLORS_BGR.items():
        bbox = get_region_bbox(pose_row, region, frame_w, frame_h, use_named_cols)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        # Label
        label = REGION_LABELS[region]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw keypoints
    for name in LANDMARK_INDICES:
        x, y = get_keypoint_px(pose_row, name, frame_w, frame_h, use_named_cols)
        if x is not None:
            cv2.circle(overlay, (int(x), int(y)), 4, (255, 255, 255), -1)
            cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 0), 1)

    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


def render_plots(fig, axes, current_time, total_time, df_roi, window_sec=30):
    """Render scrolling time-series plots for current frame."""
    for ax in axes:
        ax.clear()

    t_start = max(0, current_time - window_sec / 2)
    t_end = t_start + window_sec
    if t_end > total_time:
        t_end = total_time
        t_start = max(0, t_end - window_sec)

    plot_order = ['global', 'head', 'left_arm', 'right_arm', 'torso']

    for ax, region in zip(axes, plot_order):
        col = f"{region}_mean_diff_smooth"
        color = REGION_COLORS_MPL[region]
        label = REGION_LABELS[region]

        if col not in df_roi.columns:
            ax.set_ylabel(label, fontsize=8, color=color)
            continue

        mask = (df_roi["time_sec"] >= t_start) & (df_roi["time_sec"] <= t_end)
        subset = df_roi.loc[mask]
        t_vals = subset["time_sec"].values
        vals = subset[col].values

        # Drop NaN for clean line
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(t_vals[valid], vals[valid], color=color, linewidth=1.5)
            ax.fill_between(t_vals[valid], 0, vals[valid], color=color, alpha=0.15)

        ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
        ax.set_ylabel(label, fontsize=8, fontweight='bold', color=color)
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)", fontsize=8)
    axes[0].set_title("ROI Motion Energy (smoothed)", fontsize=9, loc="left", pad=2)

    fig.tight_layout(pad=0.5)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    return plot_img


def create_side_by_side_video(video_path, roi_csv, pose_csv, output_path,
                               patient_id=None, target_fps=15, window_sec=30):
    """Create side-by-side video with ROI boxes and motion energy plots."""
    df_roi = pd.read_csv(roi_csv)
    df_pose = pd.read_csv(pose_csv)

    # Detect pose CSV column format
    use_named_cols = "nose_x" in df_pose.columns

    # Build pose lookup by frame
    pose_lookup = {}
    for _, row in df_pose.iterrows():
        pose_lookup[int(row["frame"])] = row

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

    # Matplotlib figure
    dpi = 100
    fig, axes = plt.subplots(5, 1, figsize=(plot_w / dpi, out_h / dpi), dpi=dpi)
    plt.subplots_adjust(hspace=0.4)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (out_w, out_h))

    # Title for overlay
    clip_name = Path(video_path).stem
    if patient_id:
        import re
        clip_name = re.sub(r'^[^_]*~[^_]*_', f'{patient_id}_', clip_name)

    frame_idx = -1
    written = 0
    t0 = time.time()

    print(f"  Creating side-by-side: {Path(output_path).name}")
    print(f"  Video: {vid_w}x{vid_h} @ {src_fps:.0f}fps -> {target_fps}fps output")
    print(f"  Output: {out_w}x{out_h}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        current_time = frame_idx / src_fps

        # Draw ROI boxes if we have pose data for this frame
        pose_row = pose_lookup.get(frame_idx)
        if pose_row is not None:
            frame = draw_roi_boxes(frame, pose_row, vid_w, vid_h, use_named_cols)

        # Timestamp overlay
        cv2.putText(frame, f"t={current_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, clip_name, (10, vid_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Scale video frame
        video_panel = cv2.resize(frame, (scaled_w, out_h))

        # Render plots
        plot_img = render_plots(fig, axes, current_time, total_time,
                                df_roi, window_sec)
        plot_panel = cv2.resize(plot_img, (plot_w, out_h))

        combined = np.hstack([video_panel, plot_panel])
        writer.write(combined)
        written += 1

        if written % 100 == 0:
            elapsed = time.time() - t0
            pct = (frame_idx / total_frames) * 100
            print(f"  Progress: {pct:.0f}% ({written} frames, {written/elapsed:.1f} fps)")

    cap.release()
    writer.release()
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"  Done: {written} frames in {elapsed:.1f}s ({written/elapsed:.1f} fps)")


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side video with ROI motion energy plots")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--roi-csv", required=True, help="ROI motion CSV from extract_roi_motion.py")
    parser.add_argument("--pose-csv", required=True, help="Pose CSV (mediapipe or pose_N format)")
    parser.add_argument("--output", "-o", help="Output MP4 path")
    parser.add_argument("--patient-id", default=None, help="Patient ID for title (e.g., EM1334)")
    parser.add_argument("--fps", type=int, default=15, help="Output video FPS (default: 15)")
    parser.add_argument("--window", type=int, default=30, help="Time window in seconds (default: 30)")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    if args.output:
        output_path = Path(args.output)
    else:
        clip_num = video_path.stem.split("_")[-1]
        output_path = Path("visualizations") / f"sidebyside_roi_{clip_num}.mp4"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    create_side_by_side_video(
        video_path=str(video_path),
        roi_csv=args.roi_csv,
        pose_csv=args.pose_csv,
        output_path=str(output_path),
        patient_id=args.patient_id,
        target_fps=args.fps,
        window_sec=args.window,
    )


if __name__ == "__main__":
    main()
