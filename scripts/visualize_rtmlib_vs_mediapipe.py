"""
Comparison visualization: RTMLib Wholebody (133 kps) vs MediaPipe Pose (33 kps).
Shows velocity signals from both methods side-by-side for the first video.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_comparison(mp_csv, rtm_csv, output_path):
    """Create a comparison plot of RTMLib vs MediaPipe for one video."""
    df_mp = pd.read_csv(mp_csv)
    df_rtm = pd.read_csv(rtm_csv)

    smooth = 15  # rolling window for smoothing

    fig, axes = plt.subplots(6, 1, figsize=(16, 18), sharex=True)
    fig.suptitle("RTMLib Wholebody vs MediaPipe Pose — Clip 0099", fontsize=14, fontweight="bold")

    # 1. Wrist velocities — both methods
    ax = axes[0]
    ax.plot(df_mp["time_sec"],
            df_mp["mean_wrist_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="steelblue", linewidth=1.2, label="MediaPipe")
    ax.plot(df_rtm["time_sec"],
            df_rtm["mean_wrist_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="crimson", linewidth=1.2, label="RTMLib")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_title("Wrist Velocity Comparison", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Shoulder velocities — both methods
    ax = axes[1]
    ax.plot(df_mp["time_sec"],
            df_mp["mean_shoulder_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="steelblue", linewidth=1.2, label="MediaPipe")
    ax.plot(df_rtm["time_sec"],
            df_rtm["mean_shoulder_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="crimson", linewidth=1.2, label="RTMLib")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_title("Shoulder Velocity Comparison", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Head/nose velocity — both methods
    ax = axes[2]
    ax.plot(df_mp["time_sec"],
            df_mp["nose_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="steelblue", linewidth=1.2, label="MediaPipe")
    ax.plot(df_rtm["time_sec"],
            df_rtm["nose_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="crimson", linewidth=1.2, label="RTMLib")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_title("Head (Nose) Velocity Comparison", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 4. RTMLib-only: Hand fingertip velocity (not available in MediaPipe)
    ax = axes[3]
    ax.plot(df_rtm["time_sec"],
            df_rtm["mean_hand_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="darkorange", linewidth=1.2, label="Hand Fingertips")
    ax.plot(df_rtm["time_sec"],
            df_rtm["mean_wrist_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="crimson", linewidth=1.2, alpha=0.5, label="Wrists (reference)")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_title("RTMLib Only — Hand Fingertip vs Wrist Velocity", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5. RTMLib-only: Region confidence scores
    ax = axes[4]
    ax.plot(df_rtm["time_sec"], df_rtm["body_score"], color="forestgreen", linewidth=1, label="Body")
    ax.plot(df_rtm["time_sec"], df_rtm["face_score"], color="mediumpurple", linewidth=1, label="Face")
    ax.plot(df_rtm["time_sec"], df_rtm["left_hand_score"], color="darkorange", linewidth=1, label="Left Hand")
    ax.plot(df_rtm["time_sec"], df_rtm["right_hand_score"], color="crimson", linewidth=1, label="Right Hand")
    ax.plot(df_rtm["time_sec"], df_rtm["feet_score"], color="gray", linewidth=1, label="Feet")
    ax.set_ylabel("Confidence\nScore")
    ax.set_ylim(0, 1)
    ax.set_title("RTMLib Only — Region Confidence Scores Over Time", fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # 6. Overall body velocity — both methods
    ax = axes[5]
    ax.plot(df_mp["time_sec"],
            df_mp["mean_body_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="steelblue", linewidth=1.2, label="MediaPipe (13 landmarks)")
    ax.plot(df_rtm["time_sec"],
            df_rtm["mean_body_velocity"].rolling(smooth, center=True, min_periods=1).mean(),
            color="crimson", linewidth=1.2, label="RTMLib (21 landmarks)")
    ax.set_ylabel("Velocity\n(px/frame)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Overall Body Velocity Comparison", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    base = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction")
    output_dir = base / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    mp_dir = base / "output" / "mediapipe"
    rtm_dir = base / "output" / "rtmlib"

    # First video only
    mp_files = sorted(mp_dir.glob("*_mediapipe.csv"))[:1]

    for mp_file in mp_files:
        stem = mp_file.stem.replace("_mediapipe", "")
        rtm_file = rtm_dir / f"{stem}_rtmlib.csv"

        if not rtm_file.exists():
            print(f"Skipping {stem} — missing rtmlib data")
            continue

        clip_num = stem.split("_")[-1]
        output_path = output_dir / f"rtmlib_vs_mediapipe_{clip_num}.png"
        print(f"Creating comparison for clip {clip_num}...")
        plot_comparison(mp_file, rtm_file, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
