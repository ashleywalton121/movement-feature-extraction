"""
Method 2: DIS Optical Flow — Dense motion vectors with magnitude and direction.
Best classical dense optical flow (faster and more accurate than Farneback).
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time


def process_video(video_path):
    """Extract DIS optical flow features from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # DIS optical flow in MEDIUM mode (good balance of speed and accuracy)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = dis.calc(prev_gray, gray, None)  # shape: (H, W, 2)

        # Extract magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Compute features
        records.append({
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "mean_magnitude": np.mean(magnitude),
            "median_magnitude": np.median(magnitude),
            "max_magnitude": np.max(magnitude),
            "std_magnitude": np.std(magnitude),
            "motion_area_frac": np.mean(magnitude > 1.0),
            # Dominant direction (magnitude-weighted circular mean)
            "mean_direction_rad": np.arctan2(
                np.mean(magnitude * np.sin(angle)),
                np.mean(magnitude * np.cos(angle))
            ),
            # Direction consistency (1 = all same direction, 0 = random)
            "direction_consistency": np.sqrt(
                np.mean(np.cos(angle) * magnitude)**2 +
                np.mean(np.sin(angle) * magnitude)**2
            ) / (np.mean(magnitude) + 1e-8),
        })

        prev_gray = gray

    cap.release()

    df = pd.DataFrame(records)

    # Smoothed versions (0.5s Gaussian)
    sigma = fps * 0.5
    df["mean_mag_smooth"] = gaussian_filter1d(df["mean_magnitude"].values, sigma=sigma)

    return df


def main():
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\dis_flow")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with DIS optical flow...\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df = process_video(video_path)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_dis_flow.csv"
            df.to_csv(out_file, index=False)
            print(f"{len(df)} frames in {elapsed:.1f}s ({len(df)/elapsed:.0f} fps)")
        else:
            print("FAILED")

    print("\nDone. Output saved to:", output_dir)


if __name__ == "__main__":
    main()
