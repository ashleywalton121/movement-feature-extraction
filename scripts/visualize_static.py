"""
Static comparison plots: all 3 extraction methods stacked vertically.
One PNG per video showing time-aligned motion signals.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_comparison(video_name, frame_diff_csv, dis_flow_csv, mediapipe_csv, output_path):
    """Create a stacked comparison plot for one video."""
    df_fd = pd.read_csv(frame_diff_csv)
    df_dis = pd.read_csv(dis_flow_csv)
    df_mp = pd.read_csv(mediapipe_csv)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"Movement Feature Comparison — {video_name}", fontsize=14, fontweight="bold")

    # 1. Frame Differencing
    ax = axes[0]
    ax.plot(df_fd["time_sec"], df_fd["mean_diff"], alpha=0.3, color="steelblue", linewidth=0.5)
    ax.plot(df_fd["time_sec"], df_fd["mean_diff_smooth"], color="steelblue", linewidth=1.5, label="Smoothed")
    ax.set_ylabel("Mean Pixel\nDifference")
    ax.set_title("Frame Differencing", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. DIS Optical Flow — Magnitude
    ax = axes[1]
    ax.plot(df_dis["time_sec"], df_dis["mean_magnitude"], alpha=0.3, color="darkorange", linewidth=0.5)
    ax.plot(df_dis["time_sec"], df_dis["mean_mag_smooth"], color="darkorange", linewidth=1.5, label="Smoothed")
    ax.set_ylabel("Mean Flow\nMagnitude (px)")
    ax.set_title("DIS Optical Flow — Motion Magnitude", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. DIS Optical Flow — Motion Area
    ax = axes[2]
    ax.fill_between(df_dis["time_sec"], df_dis["motion_area_frac"], alpha=0.4, color="coral")
    ax.plot(df_dis["time_sec"], df_dis["motion_area_frac"], color="coral", linewidth=0.8)
    ax.set_ylabel("Motion Area\nFraction")
    ax.set_title("DIS Optical Flow — Fraction of Frame with Motion (>1px)", fontsize=11, loc="left")
    ax.set_ylim(0, max(0.1, df_dis["motion_area_frac"].max() * 1.2))
    ax.grid(True, alpha=0.3)

    # 4. MediaPipe — Body Part Velocities
    ax = axes[3]
    if "mean_wrist_velocity" in df_mp.columns:
        ax.plot(df_mp["time_sec"], df_mp["mean_wrist_velocity"].rolling(15, center=True).mean(),
                color="crimson", linewidth=1.2, label="Wrists")
    if "mean_shoulder_velocity" in df_mp.columns:
        ax.plot(df_mp["time_sec"], df_mp["mean_shoulder_velocity"].rolling(15, center=True).mean(),
                color="forestgreen", linewidth=1.2, label="Shoulders")
    if "nose_velocity" in df_mp.columns:
        ax.plot(df_mp["time_sec"], df_mp["nose_velocity"].rolling(15, center=True).mean(),
                color="mediumpurple", linewidth=1.2, label="Nose/Head")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("MediaPipe Pose — Keypoint Velocities (0.5s rolling mean)", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    base = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction")
    output_dir = base / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    fd_dir = base / "output" / "frame_diff"
    dis_dir = base / "output" / "dis_flow"
    mp_dir = base / "output" / "mediapipe"

    # Match files across methods by video number suffix
    fd_files = sorted(fd_dir.glob("*_frame_diff.csv"))

    for fd_file in fd_files:
        # Extract the video stem (everything before _frame_diff)
        stem = fd_file.stem.replace("_frame_diff", "")
        dis_file = dis_dir / f"{stem}_dis_flow.csv"
        mp_file = mp_dir / f"{stem}_mediapipe.csv"

        if not dis_file.exists() or not mp_file.exists():
            print(f"  Skipping {stem} — missing data files")
            continue

        # Short name for display
        parts = stem.split("_")
        short_name = parts[-1]  # e.g., "0099"

        output_path = output_dir / f"comparison_{short_name}.png"
        plot_comparison(f"Clip {short_name}", fd_file, dis_file, mp_file, output_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
