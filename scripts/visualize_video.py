"""
Side-by-side video: original video on left, time-series plots on right.
Exports as MP4 for presentations. Includes skeleton overlay on the video.
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path
import time


# MediaPipe skeleton connections (subset for upper body)
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

    # Draw connections
    for name_a, name_b in SKELETON_CONNECTIONS:
        xa, ya = mp_row.get(f"{name_a}_x"), mp_row.get(f"{name_a}_y")
        xb, yb = mp_row.get(f"{name_b}_x"), mp_row.get(f"{name_b}_y")
        vis_a = mp_row.get(f"{name_a}_visibility", 0)
        vis_b = mp_row.get(f"{name_b}_visibility", 0)

        if pd.notna(xa) and pd.notna(xb) and vis_a > 0.3 and vis_b > 0.3:
            pt_a = (int(xa), int(ya))
            pt_b = (int(xb), int(yb))
            cv2.line(overlay, pt_a, pt_b, (200, 200, 200), 2)

    # Draw keypoints
    for name in LANDMARK_NAMES:
        x, y = mp_row.get(f"{name}_x"), mp_row.get(f"{name}_y")
        vis = mp_row.get(f"{name}_visibility", 0)
        if pd.notna(x) and vis > 0.3:
            color = KEYPOINT_COLORS.get(name, (255, 255, 255))
            cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
            cv2.circle(overlay, (int(x), int(y)), 5, (255, 255, 255), 1)

    # Blend overlay
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    return frame


def render_plots(fig, axes, current_time, total_time, df_fd, df_dis, df_mp, window_sec=30):
    """Render the time-series plots for the current frame."""
    for ax in axes:
        ax.clear()

    # Time window
    t_start = max(0, current_time - window_sec / 2)
    t_end = t_start + window_sec
    if t_end > total_time:
        t_end = total_time
        t_start = max(0, t_end - window_sec)

    # 1. Frame Differencing
    ax = axes[0]
    mask = (df_fd["time_sec"] >= t_start) & (df_fd["time_sec"] <= t_end)
    ax.plot(df_fd.loc[mask, "time_sec"], df_fd.loc[mask, "mean_diff_smooth"],
            color="steelblue", linewidth=1.5)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Frame Diff", fontsize=8)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Frame Differencing", fontsize=9, loc="left", pad=2)

    # 2. DIS Optical Flow
    ax = axes[1]
    mask = (df_dis["time_sec"] >= t_start) & (df_dis["time_sec"] <= t_end)
    ax.plot(df_dis.loc[mask, "time_sec"], df_dis.loc[mask, "mean_mag_smooth"],
            color="darkorange", linewidth=1.5)
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Optical Flow", fontsize=8)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("DIS Optical Flow", fontsize=9, loc="left", pad=2)

    # 3. MediaPipe Velocities
    ax = axes[2]
    mask = (df_mp["time_sec"] >= t_start) & (df_mp["time_sec"] <= t_end)
    mp_subset = df_mp.loc[mask]
    if "mean_wrist_velocity" in mp_subset.columns:
        ax.plot(mp_subset["time_sec"],
                mp_subset["mean_wrist_velocity"].rolling(15, center=True, min_periods=1).mean(),
                color="crimson", linewidth=1.2, label="Wrists")
    if "nose_velocity" in mp_subset.columns:
        ax.plot(mp_subset["time_sec"],
                mp_subset["nose_velocity"].rolling(15, center=True, min_periods=1).mean(),
                color="mediumpurple", linewidth=1.2, label="Head")
    ax.axvline(current_time, color="red", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Pose Velocity", fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_xlim(t_start, t_end)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("MediaPipe Keypoint Velocities", fontsize=9, loc="left", pad=2)

    fig.tight_layout(pad=0.5)

    # Render to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    plot_img = np.asarray(buf)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
    return plot_img


def create_side_by_side_video(video_path, fd_csv, dis_csv, mp_csv, output_path,
                               target_fps=15, window_sec=30):
    """Create side-by-side video with skeleton overlay and time-series plots."""
    # Load data
    df_fd = pd.read_csv(fd_csv)
    df_dis = pd.read_csv(dis_csv)
    df_mp = pd.read_csv(mp_csv)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_time = total_frames / src_fps

    # Frame skip for target fps
    frame_skip = max(1, int(round(src_fps / target_fps)))

    # Output dimensions: video on left, plots on right
    # Scale video to fit left half
    out_h = 720
    scale = out_h / vid_h
    scaled_w = int(vid_w * scale)
    plot_w = scaled_w  # same width for plots
    out_w = scaled_w + plot_w

    # Set up matplotlib figure for plots (match video height)
    dpi = 100
    fig_w = plot_w / dpi
    fig_h = out_h / dpi
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=dpi)
    plt.subplots_adjust(hspace=0.4)

    # Set up video writer
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

        # Skip frames to match target fps
        if frame_idx % frame_skip != 0:
            continue

        current_time = frame_idx / src_fps

        # Draw skeleton on video frame
        if frame_idx < len(df_mp):
            mp_row = df_mp.iloc[frame_idx]
            if mp_row.get("pose_detected", False):
                frame = draw_skeleton(frame, mp_row)

        # Add timestamp to video
        cv2.putText(frame, f"t={current_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Scale video frame
        video_panel = cv2.resize(frame, (scaled_w, out_h))

        # Render plots
        plot_img = render_plots(fig, axes, current_time, total_time,
                                df_fd, df_dis, df_mp, window_sec)
        plot_panel = cv2.resize(plot_img, (plot_w, out_h))

        # Combine side by side
        combined = np.hstack([video_panel, plot_panel])
        writer.write(combined)
        written += 1

    cap.release()
    writer.release()
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"  Saved: {output_path} ({written} frames, {elapsed:.1f}s)")


def main():
    base = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction")
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = base / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    fd_dir = base / "output" / "frame_diff"
    dis_dir = base / "output" / "dis_flow"
    mp_dir = base / "output" / "mediapipe"

    fd_files = sorted(fd_dir.glob("*_frame_diff.csv"))[:1]  # first video only

    print(f"Creating side-by-side video for first clip...\n")

    for fd_file in fd_files:
        stem = fd_file.stem.replace("_frame_diff", "")
        dis_file = dis_dir / f"{stem}_dis_flow.csv"
        mp_file = mp_dir / f"{stem}_mediapipe.csv"
        video_file = video_dir / f"{stem}.avi"

        if not all(f.exists() for f in [dis_file, mp_file, video_file]):
            print(f"  Skipping {stem} — missing files")
            continue

        clip_num = stem.split("_")[-1]
        output_path = output_dir / f"sidebyside_{clip_num}.mp4"
        print(f"  Processing clip {clip_num}...")
        create_side_by_side_video(video_file, fd_file, dis_file, mp_file, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
