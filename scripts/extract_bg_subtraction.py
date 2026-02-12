"""
Background Subtraction (MOG2) — motion extraction via learned background model.

Learns a statistical model of the static background and identifies foreground
(moving) pixels. Better than frame differencing at ignoring camera noise and
slow lighting changes, since it adapts over time.

Outputs per-frame:
- foreground_frac: fraction of pixels classified as foreground (moving)
- foreground_mean_intensity: mean intensity of foreground pixels
- foreground_area_px: total foreground pixel count
- smoothed versions of each
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import time


def process_video(video_path, history=500, var_threshold=50, detect_shadows=False):
    """
    Extract background-subtraction motion features from a single video.

    Args:
        video_path: Path to video file
        history: Number of frames for background model history
        var_threshold: Variance threshold for pixel classification (higher = less sensitive)
        detect_shadows: Whether to detect shadows (slower, marks shadows as gray)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height

    # Create MOG2 background subtractor
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply background subtraction
        fg_mask = bg_sub.apply(frame)

        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Binary: foreground pixels are 255
        fg_binary = (fg_mask == 255).astype(np.uint8)
        fg_count = int(np.sum(fg_binary))
        fg_frac = fg_count / total_pixels

        # Mean intensity of foreground region in grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if fg_count > 0:
            fg_mean_intensity = float(np.mean(gray[fg_binary == 1]))
        else:
            fg_mean_intensity = 0.0

        records.append({
            "frame": frame_idx,
            "time_sec": round(frame_idx / fps, 4),
            "foreground_frac": round(fg_frac, 6),
            "foreground_area_px": fg_count,
            "foreground_mean_intensity": round(fg_mean_intensity, 2),
        })

        frame_idx += 1

    cap.release()

    if not records:
        return None

    df = pd.DataFrame(records)

    # Gaussian smoothing (0.5s FWHM)
    sigma = (0.5 * fps) / 2.355
    for col in ["foreground_frac", "foreground_mean_intensity"]:
        df[f"{col}_smooth"] = np.round(
            gaussian_filter1d(df[col].values, sigma=sigma), 6
        )

    return df


def main():
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\bg_subtraction")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with MOG2 background subtraction...\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df = process_video(video_path)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_bg_subtraction.csv"
            df.to_csv(out_file, index=False)
            print(f"{len(df)} frames in {elapsed:.1f}s ({len(df)/elapsed:.0f} fps)")
        else:
            print("FAILED")

    print(f"\nDone. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
