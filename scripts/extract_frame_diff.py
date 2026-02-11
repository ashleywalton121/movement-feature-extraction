"""
Method 1: Frame Differencing — Global motion energy extraction.
Simplest possible movement quantification from video.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time


def process_video(video_path):
    """Extract frame-differencing motion energy from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0).astype(np.float32)

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0).astype(np.float32)

        diff = np.abs(gray - prev_gray)

        records.append({
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "mean_diff": np.mean(diff),
            "max_diff": np.max(diff),
            "motion_area_frac": np.mean(diff > 10),  # fraction of pixels above threshold
        })

        prev_gray = gray

    cap.release()

    df = pd.DataFrame(records)

    # Add smoothed version (0.5s Gaussian)
    sigma = fps * 0.5
    df["mean_diff_smooth"] = gaussian_filter1d(df["mean_diff"].values, sigma=sigma)

    return df


def main():
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\frame_diff")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with frame differencing...\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df = process_video(video_path)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_frame_diff.csv"
            df.to_csv(out_file, index=False)
            print(f"{len(df)} frames in {elapsed:.1f}s ({len(df)/elapsed:.0f} fps)")
        else:
            print("FAILED")

    print("\nDone. Output saved to:", output_dir)


if __name__ == "__main__":
    main()
