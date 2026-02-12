"""
Motion Energy Images (MEI) and Motion History Images (MHI) extraction.

MEI: Binary image showing WHERE motion occurred over a time window.
MHI: Grayscale image encoding WHEN motion occurred (brighter = more recent).

Outputs per-frame time series:
- mei_frac: fraction of pixels with any motion in the window (spatial extent)
- mhi_mean: mean MHI intensity (overall recency-weighted motion)
- mhi_max: max MHI intensity (strongest recent motion)
- motion_energy: instantaneous frame difference energy (sum of absolute diffs)
- smoothed versions

Also saves MEI/MHI snapshot images at regular intervals for visual inspection.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time


def process_video(video_path, mhi_duration_sec=2.0, diff_threshold=25,
                  save_snapshots=True, snapshot_interval_sec=30):
    """
    Extract MEI/MHI motion features from a single video.

    Args:
        video_path: Path to video file
        mhi_duration_sec: Time window for MHI decay (seconds)
        diff_threshold: Pixel difference threshold for motion detection
        save_snapshots: Whether to save MHI/MEI snapshot images
        snapshot_interval_sec: Interval between snapshots
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height

    mhi_duration = mhi_duration_sec * fps  # duration in frames

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None, []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # Initialize MHI (float32 required by cv2.motempl)
    mhi = np.zeros((height, width), dtype=np.float32)

    records = []
    snapshots = []
    frame_idx = 0
    snapshot_frame_interval = int(snapshot_interval_sec * fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        # Frame difference
        diff = cv2.absdiff(gray_blur, prev_gray)
        _, motion_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # Clean up motion mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

        # Update MHI: set moving pixels to current timestamp, decay old ones
        timestamp = np.float64(frame_idx)
        mhi_duration_f = np.float64(mhi_duration)
        cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, mhi_duration_f)

        # Compute MEI: any pixel that moved within the MHI window
        mei = (mhi > (timestamp - mhi_duration)).astype(np.uint8)
        mei_frac = float(np.sum(mei)) / total_pixels

        # Normalized MHI for metrics (0-255 range)
        mhi_vis = np.clip((mhi - (timestamp - mhi_duration)) / mhi_duration, 0, 1)
        mhi_mean = float(np.mean(mhi_vis))
        mhi_max = float(np.max(mhi_vis))

        # Instantaneous motion energy
        motion_energy = float(np.mean(diff))
        motion_area = float(np.sum(motion_mask > 0)) / total_pixels

        records.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 4),
            "motion_energy": round(motion_energy, 4),
            "motion_area": round(motion_area, 6),
            "mei_frac": round(mei_frac, 6),
            "mhi_mean": round(mhi_mean, 6),
            "mhi_max": round(mhi_max, 4),
        })

        # Save snapshots
        if save_snapshots and frame_idx % snapshot_frame_interval == 0:
            mhi_img = (mhi_vis * 255).astype(np.uint8)
            mhi_color = cv2.applyColorMap(mhi_img, cv2.COLORMAP_JET)
            mei_color = cv2.applyColorMap(mei * 255, cv2.COLORMAP_HOT)

            # Composite: original + MHI overlay
            frame_small = cv2.resize(frame, (width // 2, height // 2))
            mhi_small = cv2.resize(mhi_color, (width // 2, height // 2))
            mei_small = cv2.resize(mei_color, (width // 2, height // 2))

            # Top row: original + MHI, bottom row: MEI + blank
            top = np.hstack([frame_small, mhi_small])
            # Add label
            t_sec = frame_idx / fps
            cv2.putText(top, f"t={t_sec:.1f}s", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(top, "MHI", (width // 2 + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            snapshots.append({
                "frame": frame_idx,
                "time_sec": t_sec,
                "image": top,
            })

        prev_gray = gray_blur

    cap.release()

    if not records:
        return None, []

    df = pd.DataFrame(records)

    # Gaussian smoothing (0.5s FWHM)
    sigma = (0.5 * fps) / 2.355
    for col in ["motion_energy", "motion_area", "mei_frac", "mhi_mean"]:
        df[f"{col}_smooth"] = np.round(
            gaussian_filter1d(df[col].values, sigma=sigma), 6
        )

    return df, snapshots


def main():
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\mei_mhi")
    snapshot_dir = output_dir / "snapshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with MEI/MHI...\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df, snapshots = process_video(video_path)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_mei_mhi.csv"
            df.to_csv(out_file, index=False)

            # Save snapshots
            for snap in snapshots:
                snap_file = snapshot_dir / f"{video_path.stem}_t{snap['time_sec']:.0f}s.png"
                cv2.imwrite(str(snap_file), snap["image"])

            print(f"{len(df)} frames in {elapsed:.1f}s ({len(df)/elapsed:.0f} fps), {len(snapshots)} snapshots")
        else:
            print("FAILED")

    print(f"\nDone. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
