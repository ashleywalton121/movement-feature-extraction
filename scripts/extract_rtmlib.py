"""
Method 4: RTMPose Wholebody via rtmlib — 133 keypoints (body + face + hands + feet).
Significantly more detailed than MediaPipe's 33 keypoints.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from rtmlib import Wholebody, draw_skeleton
import time


# 133 COCO-WholeBody keypoint layout:
#   0-16:   body (17 keypoints)
#   17-22:  feet (6 keypoints)
#   23-90:  face (68 keypoints)
#   91-111: left hand (21 keypoints)
#   112-132: right hand (21 keypoints)

# Key indices we'll track for movement features
KEY_INDICES = {
    # Body
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    # Feet
    "left_big_toe": 17,
    "right_big_toe": 20,
    # Hands (wrist/fingertip landmarks)
    "left_hand_middle_tip": 103,
    "right_hand_middle_tip": 124,
}


def process_video(video_path, wholebody_model):
    """Extract RTMPose wholebody keypoints from a single video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    records = []
    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Run wholebody estimation
        keypoints, scores = wholebody_model(frame)

        row = {
            "frame": frame_idx,
            "time_sec": frame_idx / fps,
            "num_people": len(keypoints),
        }

        if len(keypoints) > 0:
            # Use first detected person
            kps = keypoints[0]   # shape: (133, 2)
            scs = scores[0]      # shape: (133,)

            row["pose_detected"] = True
            row["total_keypoints_visible"] = int(np.sum(scs > 0.3))

            # Store key landmark positions and scores
            for name, idx in KEY_INDICES.items():
                row[f"{name}_x"] = kps[idx, 0]
                row[f"{name}_y"] = kps[idx, 1]
                row[f"{name}_score"] = scs[idx]

            # Region-level confidence (mean score for each body region)
            row["body_score"] = np.mean(scs[0:17])
            row["face_score"] = np.mean(scs[23:91])
            row["left_hand_score"] = np.mean(scs[91:112])
            row["right_hand_score"] = np.mean(scs[112:133])
            row["feet_score"] = np.mean(scs[17:23])
        else:
            row["pose_detected"] = False
            row["total_keypoints_visible"] = 0
            for name in KEY_INDICES:
                row[f"{name}_x"] = np.nan
                row[f"{name}_y"] = np.nan
                row[f"{name}_score"] = 0.0
            row["body_score"] = 0.0
            row["face_score"] = 0.0
            row["left_hand_score"] = 0.0
            row["right_hand_score"] = 0.0
            row["feet_score"] = 0.0

        records.append(row)

    cap.release()
    df = pd.DataFrame(records)

    # Compute velocities for key landmarks
    for name in KEY_INDICES:
        dx = df[f"{name}_x"].diff()
        dy = df[f"{name}_y"].diff()
        df[f"{name}_velocity"] = np.sqrt(dx**2 + dy**2)

    # Aggregate features
    df["mean_wrist_velocity"] = df[["left_wrist_velocity", "right_wrist_velocity"]].mean(axis=1)
    df["mean_shoulder_velocity"] = df[["left_shoulder_velocity", "right_shoulder_velocity"]].mean(axis=1)
    df["mean_hand_velocity"] = df[["left_hand_middle_tip_velocity", "right_hand_middle_tip_velocity"]].mean(axis=1)

    # Overall body movement
    body_vel_cols = [f"{name}_velocity" for name in
                     ["nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                      "left_wrist", "right_wrist", "left_hip", "right_hip"]]
    existing = [c for c in body_vel_cols if c in df.columns]
    df["mean_body_velocity"] = df[existing].mean(axis=1)

    return df


def main():
    video_dir = Path(r"C:\Users\ashle\Documents\temp_review\EM1334")
    output_dir = Path(r"C:\Users\ashle\Documents\git\movement-feature-extraction\output\rtmlib")
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.avi"))[:3]
    print(f"Processing {len(videos)} videos with RTMPose Wholebody (133 keypoints)...\n")

    # Initialize model once (downloads weights on first run)
    print("  Loading wholebody model...")
    t0 = time.time()
    wholebody = Wholebody(
        to_openpose=False,
        mode="balanced",       # balanced between speed and accuracy
        backend="onnxruntime",
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s\n")

    for video_path in videos:
        print(f"  {video_path.name}...", end=" ", flush=True)
        t0 = time.time()
        df = process_video(video_path, wholebody)
        elapsed = time.time() - t0

        if df is not None:
            out_file = output_dir / f"{video_path.stem}_rtmlib.csv"
            df.to_csv(out_file, index=False)
            detected = df["pose_detected"].sum()
            total = len(df)
            avg_kps = df.loc[df["pose_detected"], "total_keypoints_visible"].mean()
            print(f"{total} frames in {elapsed:.1f}s ({total/elapsed:.0f} fps) — "
                  f"pose: {detected}/{total} ({100*detected/total:.0f}%), "
                  f"avg visible keypoints: {avg_kps:.0f}/133")
        else:
            print("FAILED")

    print("\nDone. Output saved to:", output_dir)


if __name__ == "__main__":
    main()
