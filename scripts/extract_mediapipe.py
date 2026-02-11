"""
Method 3: MediaPipe Pose — 33 body keypoints with visibility scores.
Extracts keypoint positions and derives velocity-based movement features.
Uses the new mediapipe.tasks API (v0.10.20+).
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import time
import urllib.request
import os


# Key landmark indices for upper body (most visible in EMU with blankets)
LANDMARK_NAMES = {
    0: "nose",
    2: "left_eye",
    5: "right_eye",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker_full.task"


def download_model():
    """Download the MediaPipe pose landmarker model if not present."""
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose landmarker model to {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")


def process_video(video_path):
    """Extract MediaPipe pose keypoints from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up pose landmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    records = []
    frame_idx = -1

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # Convert to MediaPipe Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Detect pose
            timestamp_ms = int(frame_idx * 1000 / fps)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "frame": frame_idx,
                "time_sec": frame_idx / fps,
                "pose_detected": len(results.pose_landmarks) > 0,
            }

            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]  # first person
                for idx, name in LANDMARK_NAMES.items():
                    lm = landmarks[idx]
                    row[f"{name}_x"] = lm.x * width
                    row[f"{name}_y"] = lm.y * height
                    row[f"{name}_visibility"] = lm.visibility
            else:
                for idx, name in LANDMARK_NAMES.items():
                    row[f"{name}_x"] = np.nan
                    row[f"{name}_y"] = np.nan
                    row[f"{name}_visibility"] = 0.0

            records.append(row)

    cap.release()

    df = pd.DataFrame(records)

    # Compute velocity for key landmarks (pixels/frame)
    for name in LANDMARK_NAMES.values():
        dx = df[f"{name}_x"].diff()
        dy = df[f"{name}_y"].diff()
        df[f"{name}_velocity"] = np.sqrt(dx**2 + dy**2)

    # Aggregate movement features
    wrist_cols = ["left_wrist_velocity", "right_wrist_velocity"]
    existing = [c for c in wrist_cols if c in df.columns]
    if existing:
        df["mean_wrist_velocity"] = df[existing].mean(axis=1)

    shoulder_cols = ["left_shoulder_velocity", "right_shoulder_velocity"]
    existing = [c for c in shoulder_cols if c in df.columns]
    if existing:
        df["mean_shoulder_velocity"] = df[existing].mean(axis=1)

    # Overall body movement (mean velocity across all tracked landmarks)
    vel_cols = [c for c in df.columns if c.endswith("_velocity")]
    if vel_cols:
        df["mean_body_velocity"] = df[vel_cols].mean(axis=1)

    return df


def main():
    download_model()

    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\mediapipe")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with MediaPipe Pose...\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df = process_video(video_path)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_mediapipe.csv"
            df.to_csv(out_file, index=False)
            detected = df["pose_detected"].sum()
            total = len(df)
            print(f"{total} frames in {elapsed:.1f}s ({total/elapsed:.0f} fps) — pose detected: {detected}/{total} ({100*detected/total:.0f}%)")
        else:
            print("FAILED")

    print("\nDone. Output saved to:", output_dir)


if __name__ == "__main__":
    main()
