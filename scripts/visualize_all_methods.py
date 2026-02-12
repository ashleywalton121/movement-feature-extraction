"""
Compare all motion extraction methods on a single video clip.

Creates a stacked plot showing time-series from each method:
1. Frame Differencing (mean pixel diff)
2. DIS Optical Flow (mean magnitude)
3. Background Subtraction (foreground fraction)
4. MEI/MHI (motion history mean + motion energy)
5. MediaPipe Pose (mean wrist velocity)
6. ROI Motion (per-region mean diff)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys


def load_csv(path):
    """Load CSV, return None if missing."""
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_all_methods(clip_id, output_dir, data_dir, roi_csv_path=None):
    """Create comparison plot for one video clip."""
    base = Path(data_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load all available data
    df_fd = load_csv(base / "frame_diff" / f"{clip_id}_frame_diff.csv")
    df_dis = load_csv(base / "dis_flow" / f"{clip_id}_dis_flow.csv")
    df_bg = load_csv(base / "bg_subtraction" / f"{clip_id}_bg_subtraction.csv")
    df_mhi = load_csv(base / "mei_mhi" / f"{clip_id}_mei_mhi.csv")
    df_mp = load_csv(base / "mediapipe" / f"{clip_id}_mediapipe.csv")
    df_roi = load_csv(roi_csv_path) if roi_csv_path else None

    # Count available methods
    methods = []
    if df_fd is not None: methods.append("frame_diff")
    if df_dis is not None: methods.append("dis_flow")
    if df_bg is not None: methods.append("bg_sub")
    if df_mhi is not None: methods.append("mei_mhi")
    if df_mp is not None: methods.append("mediapipe")
    if df_roi is not None: methods.append("roi_motion")

    n_plots = len(methods)
    if n_plots == 0:
        print(f"  No data found for {clip_id}")
        return None

    # Short clip name for title
    parts = clip_id.split("_")
    short_name = f"{parts[0]}_{parts[-1]}" if len(parts) > 1 else clip_id

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2.2 * n_plots), sharex=True,
                             gridspec_kw={'hspace': 0.08})
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    # 1. Frame Differencing
    if df_fd is not None:
        ax = axes[ax_idx]
        t = df_fd["time_sec"].values
        col = "mean_diff_smooth" if "mean_diff_smooth" in df_fd.columns else "mean_diff"
        ax.plot(t, df_fd[col].values, color="#3498db", linewidth=0.8)
        ax.fill_between(t, 0, df_fd[col].values, color="#3498db", alpha=0.15)
        ax.set_ylabel("Frame Diff", fontsize=9, fontweight="bold", color="#3498db")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # 2. DIS Optical Flow
    if df_dis is not None:
        ax = axes[ax_idx]
        t = df_dis["time_sec"].values
        col = "mean_mag_smooth" if "mean_mag_smooth" in df_dis.columns else "mean_magnitude"
        ax.plot(t, df_dis[col].values, color="#e67e22", linewidth=0.8)
        ax.fill_between(t, 0, df_dis[col].values, color="#e67e22", alpha=0.15)
        ax.set_ylabel("Optical Flow", fontsize=9, fontweight="bold", color="#e67e22")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # 3. Background Subtraction
    if df_bg is not None:
        ax = axes[ax_idx]
        t = df_bg["time_sec"].values
        col = "foreground_frac_smooth" if "foreground_frac_smooth" in df_bg.columns else "foreground_frac"
        ax.plot(t, df_bg[col].values, color="#2ecc71", linewidth=0.8)
        ax.fill_between(t, 0, df_bg[col].values, color="#2ecc71", alpha=0.15)
        ax.set_ylabel("BG Sub\n(fg frac)", fontsize=9, fontweight="bold", color="#2ecc71")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # 4. MEI/MHI
    if df_mhi is not None:
        ax = axes[ax_idx]
        t = df_mhi["time_sec"].values
        col_energy = "motion_energy_smooth" if "motion_energy_smooth" in df_mhi.columns else "motion_energy"
        col_mhi = "mhi_mean_smooth" if "mhi_mean_smooth" in df_mhi.columns else "mhi_mean"
        ax.plot(t, df_mhi[col_energy].values, color="#9b59b6", linewidth=0.8, label="Motion Energy")
        ax.plot(t, df_mhi[col_mhi].values * df_mhi[col_energy].max(), color="#e74c3c",
                linewidth=0.8, alpha=0.7, label="MHI Mean (scaled)")
        ax.fill_between(t, 0, df_mhi[col_energy].values, color="#9b59b6", alpha=0.1)
        ax.set_ylabel("MEI/MHI", fontsize=9, fontweight="bold", color="#9b59b6")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # 5. MediaPipe Pose Velocity
    if df_mp is not None:
        ax = axes[ax_idx]
        t = df_mp["time_sec"].values
        if "mean_wrist_velocity" in df_mp.columns:
            wrist = df_mp["mean_wrist_velocity"].rolling(15, center=True, min_periods=1).mean()
            ax.plot(t, wrist.values, color="#e74c3c", linewidth=0.8, label="Wrists")
        if "nose_velocity" in df_mp.columns:
            head = df_mp["nose_velocity"].rolling(15, center=True, min_periods=1).mean()
            ax.plot(t, head.values, color="#8e44ad", linewidth=0.8, label="Head")
        ax.fill_between(t, 0, wrist.values if "mean_wrist_velocity" in df_mp.columns else 0,
                        color="#e74c3c", alpha=0.1)
        ax.set_ylabel("Pose Velocity", fontsize=9, fontweight="bold", color="#e74c3c")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # 6. ROI Motion
    if df_roi is not None:
        ax = axes[ax_idx]
        t = df_roi["time_sec"].values
        roi_colors = {'head': '#e74c3c', 'left_arm': '#3498db',
                      'right_arm': '#2ecc71', 'torso': '#f39c12'}
        roi_labels = {'head': 'Head', 'left_arm': 'L Arm',
                      'right_arm': 'R Arm', 'torso': 'Torso'}
        for region, color in roi_colors.items():
            col = f"{region}_mean_diff_smooth"
            if col in df_roi.columns:
                vals = df_roi[col].values
                mask = ~np.isnan(vals)
                if mask.any():
                    ax.plot(t[mask], vals[mask], color=color, linewidth=0.8,
                            label=roi_labels[region])
        ax.set_ylabel("ROI Motion", fontsize=9, fontweight="bold", color="#7f8c8d")
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        ax_idx += 1

    # X-axis and title
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    fig.suptitle(f"All Methods Comparison: {short_name}", fontsize=13, fontweight="bold", y=0.99)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.05, hspace=0.08)

    output_path = out / f"all_methods_{parts[-1] if len(parts) > 1 else clip_id}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare all motion extraction methods")
    parser.add_argument("--clip-id", required=True,
                        help="Clip identifier (filename stem without method suffix)")
    parser.add_argument("--data-dir", default=r"C:\Users\ashle\Documents\git\movement-feature-extraction\output",
                        help="Base output directory containing method subdirs")
    parser.add_argument("--output-dir", default=r"C:\Users\ashle\Documents\git\movement-feature-extraction\visualizations",
                        help="Output directory for visualization PNGs")
    parser.add_argument("--roi-csv", default=None,
                        help="Path to ROI motion CSV (if available)")

    args = parser.parse_args()
    plot_all_methods(args.clip_id, args.output_dir, args.data_dir, args.roi_csv)


if __name__ == "__main__":
    main()
