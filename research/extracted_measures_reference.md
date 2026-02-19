# Extracted Measures Reference

> A comprehensive catalog of every time-series measure extracted from EM1334 video data, organized by method.
>
> **Data source:** EM1334 study `1d25e418-8093-4f8e-871a-ad30bf7b56d1`, clips 0099-0101
> **Video specs:** 1280x720, ~30 fps, overhead ceiling camera, patient in hospital bed

---

## Methods at a Glance

| # | Method | Script | Speed (CPU) | Description |
|:-:|:-------|:-------|:-----------:|:------------|
| 1 | **Frame Differencing** | `extract_frame_diff.py` | ~100 fps | Absolute pixel difference between consecutive frames after Gaussian blur. The simplest motion baseline — one global intensity value per frame. |
| 2 | **DIS Optical Flow** | `extract_dis_flow.py` | ~11 fps | Dense Inverse Search optical flow. Estimates a 2D motion vector (direction + magnitude) for every pixel, capturing both *how fast* and *which direction* things moved. |
| 3 | **Background Subtraction (MOG2)** | `extract_bg_subtraction.py` | ~60-77 fps | Learns a statistical background model over time and classifies each pixel as foreground (moving) or background. Adapts to gradual lighting changes. |
| 4 | **MEI / MHI Motion Templates** | `extract_mei_mhi.py` | ~58-60 fps | Temporal motion templates over a sliding window. MEI shows *where* motion occurred; MHI encodes *when* (recent vs. old). Captures sustained vs. transient movement. |
| 5 | **MediaPipe Pose** | `extract_mediapipe.py` | ~35 fps | Google's pose estimator. Tracks 13 upper-body keypoints (x, y, visibility) per frame and computes per-joint velocities. Single-person, 33 keypoints total. |
| 6 | **RTMLib Wholebody Pose** | `extract_rtmlib.py` | ~1 fps | RTMPose 133-keypoint whole-body model (body + face + hands + feet). 4x the detail of MediaPipe. Includes region-level confidence scores and fingertip tracking. |
| 7 | **Pose-Guided ROI Motion** | `extract_roi_motion.py`* | ~120 fps | Combines pose estimation with frame differencing. Uses keypoint positions to define adaptive bounding boxes around 4 body regions, then computes per-region motion energy. |

*\*Located in the seizure-semiology-mapping repo*

---

## **1. Frame Differencing**

| | |
|:--|:--|
| **Script** | `scripts/extract_frame_diff.py` |
| **Speed** | ~100 fps (CPU) |
| **Output** | `output/frame_diff/` |
| **Depends on** | `opencv-python` only |

The simplest motion detection method. Computes the absolute pixel difference between consecutive frames after Gaussian blur (21x21 kernel to reduce noise). Pixels that changed significantly between frames indicate movement.

| | |
|:--|:--|
| **Strengths** | Extremely fast, no setup, works regardless of occlusion (detects blanket movement too) |
| **Weaknesses** | Cannot distinguish patient from nurse movement. Sensitive to lighting changes and compression artifacts. No spatial specificity — just one number per frame |

### Extracted Measures

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `mean_diff` | raw | Mean absolute pixel intensity change across the entire frame. Higher = more overall movement |
| `max_diff` | raw | Maximum pixel intensity change in the frame |
| `motion_area_frac` | raw | Fraction of pixels exceeding change threshold (>10). Indicates how much of the frame is moving |
| `mean_diff_smooth` | smooth | Gaussian-smoothed `mean_diff` (0.5s window) |

---

## **2. DIS Optical Flow**

| | |
|:--|:--|
| **Script** | `scripts/extract_dis_flow.py` |
| **Speed** | ~11 fps (CPU) |
| **Output** | `output/dis_flow/` |
| **Depends on** | `opencv-python` only |

Dense Inverse Search (DIS) is an efficient dense optical flow algorithm. Unlike frame differencing which only detects *that* pixels changed, optical flow estimates *how* they moved — producing a 2D motion vector (direction + magnitude) for every pixel.

| | |
|:--|:--|
| **Strengths** | Richer than frame differencing — captures both speed and direction. Can detect flow patterns (e.g., rhythmic back-and-forth for clonic movements). Good balance of accuracy and CPU speed |
| **Weaknesses** | Still global (no per-region breakdown by default). Slower than frame differencing. Classical method — less accurate than deep learning approaches on complex scenes |

### Extracted Measures

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `mean_magnitude` | raw | Mean motion vector length across all pixels (px/frame). Overall movement intensity |
| `median_magnitude` | raw | Median motion vector length. More robust to outliers than mean |
| `max_magnitude` | raw | Peak motion vector length in the frame |
| `std_magnitude` | raw | Standard deviation of motion magnitudes. High = mixture of fast and slow regions |
| `motion_area_frac` | raw | Fraction of pixels with flow magnitude > 1 px |
| `mean_direction_rad` | raw | Magnitude-weighted circular mean direction (radians). Dominant movement direction |
| `direction_consistency` | raw | Directional coherence (0-1). 1 = all same direction (tonic), low = chaotic (hyperkinetic) |
| `mean_mag_smooth` | smooth | Smoothed `mean_magnitude` |

---

## **3. Background Subtraction (MOG2)**

| | |
|:--|:--|
| **Script** | `scripts/extract_bg_subtraction.py` |
| **Speed** | ~60-77 fps (CPU) |
| **Output** | `output/bg_subtraction/` |
| **Depends on** | `opencv-python` only |
| **Parameters** | `history=500`, `varThreshold=50`, morphological open/close (5x5 ellipse) |

MOG2 (Mixture of Gaussians) learns a statistical model of the background over time. Each pixel is modeled as a mixture of Gaussian distributions. Pixels that don't fit the learned background model are classified as "foreground" (moving objects). Unlike frame differencing, this adapts to gradual changes (lighting shifts, IR mode switching) because the background model updates continuously.

| | |
|:--|:--|
| **Strengths** | Adapts to slow lighting changes (better than frame differencing for long recordings). Distinguishes sustained movement from transient noise. Fast on CPU |
| **Weaknesses** | Takes ~500 frames (~17s at 30 fps) to learn the initial background. Large scene changes (nurse entering) temporarily confuse the model. No per-body-region breakdown |

### Extracted Measures

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `foreground_frac` | raw | Fraction of pixels classified as foreground (moving) |
| `foreground_area_px` | raw | Raw pixel count of foreground region |
| `foreground_mean_intensity` | raw | Average brightness of foreground pixels. Bright = patient, dark = shadow |
| `foreground_frac_smooth` | smooth | Smoothed `foreground_frac` (FWHM-based Gaussian, 0.5s) |
| `foreground_mean_intensity_smooth` | smooth | Smoothed `foreground_mean_intensity` |

---

## **4. Motion Energy Images (MEI) / Motion History Images (MHI)**

| | |
|:--|:--|
| **Script** | `scripts/extract_mei_mhi.py` |
| **Speed** | ~58-60 fps (CPU) |
| **Output** | `output/mei_mhi/` (CSVs + snapshot PNGs every 30s) |
| **Depends on** | `opencv-contrib-python` (`cv2.motempl`) |
| **Parameters** | MHI window duration = 2 seconds (configurable) |

MEI and MHI are temporal motion templates that summarize movement over a sliding time window.

- **MEI (Motion Energy Image):** Binary image showing *where* any motion occurred within the window. Each pixel is 1 if it moved at any point, 0 otherwise.
- **MHI (Motion History Image):** Grayscale image encoding *when* motion occurred. Brighter pixels = more recent motion. Older motion fades over the window duration.

| | |
|:--|:--|
| **Strengths** | Captures temporal dynamics — not just "is there motion?" but "how has motion evolved over the last N seconds?" Useful for sustained movements (tonic posturing) vs. brief bursts (myoclonic jerks). Snapshot images provide intuitive visual summaries |
| **Weaknesses** | Window duration is a fixed parameter that may not suit all movement types. Doesn't separate patient from other movement sources |

### Extracted Measures

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `motion_energy` | raw | Instantaneous mean frame difference (per-frame) |
| `motion_area` | raw | Fraction of pixels with motion above threshold |
| `mei_frac` | raw | Fraction of the frame that moved within the 2s MHI window. High = widespread or sustained movement |
| `mhi_mean` | raw | Mean MHI intensity (0-1). High = lots of recent movement across the frame |
| `mhi_max` | raw | Peak MHI intensity. Indicates the most recently active pixel region |
| `motion_energy_smooth` | smooth | Smoothed `motion_energy` |
| `motion_area_smooth` | smooth | Smoothed `motion_area` |
| `mei_frac_smooth` | smooth | Smoothed `mei_frac` |
| `mhi_mean_smooth` | smooth | Smoothed `mhi_mean` |

---

## **5. MediaPipe Pose Estimation**

| | |
|:--|:--|
| **Script** | `scripts/extract_mediapipe.py` |
| **Speed** | ~35 fps (CPU) |
| **Output** | `output/mediapipe/` |
| **Depends on** | `mediapipe` (auto-downloads `pose_landmarker_full.task` model) |

Google's MediaPipe estimates 33 body keypoints (pose landmarks) per frame. Each keypoint has an (x, y) position in pixels and a visibility confidence score. The script extracts 13 upper-body keypoints and computes frame-to-frame velocities for each.

| | |
|:--|:--|
| **Strengths** | Body-part-specific motion signals (e.g., wrist velocity separately from head velocity). Critical for seizure semiology where the *location* of movement matters (lateralized arm jerking vs. head turning). Fast on CPU, trivial install |
| **Weaknesses** | Only 33 keypoints (no face mesh, no individual fingers). Single-person only — cannot handle nurse entering frame. Trained on standing people at eye level, so accuracy degrades with overhead camera angles and supine patients |

### Extracted Measures — Keypoint Positions and Velocities

13 keypoints, each with 4 columns (**52 columns total**). All positions are in pixels, velocities in px/frame.

| Keypoint | `{name}_x` | `{name}_y` | `{name}_visibility` | `{name}_velocity` |
|:---------|:----------:|:----------:|:-------------------:|:-----------------:|
| `nose` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_eye` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_eye` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_ear` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_ear` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_shoulder` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_shoulder` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_elbow` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_elbow` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_wrist` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_wrist` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_hip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_hip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |

### Extracted Measures — Aggregate and Metadata

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `pose_detected` | metadata | Boolean — was a person found in the frame? |
| `mean_wrist_velocity` | aggregate | Average of left + right wrist velocity (px/frame) |
| `mean_shoulder_velocity` | aggregate | Average of left + right shoulder velocity (px/frame) |
| `mean_body_velocity` | aggregate | Average velocity across all 13 keypoints (px/frame) |

---

## **6. RTMLib Wholebody Pose Estimation**

| | |
|:--|:--|
| **Script** | `scripts/extract_rtmlib.py` |
| **Speed** | ~1 fps (CPU) |
| **Output** | `output/rtmlib/` |
| **Depends on** | `rtmlib` (ONNX Runtime backend, `balanced` mode) |
| **Model** | RTMPose Wholebody — 133 COCO-WholeBody keypoints |

RTMPose via rtmlib provides 133 keypoints covering the entire body: 17 body, 68 face, 21 per hand (42 total), and 6 feet. This is 4x the detail of MediaPipe. Includes per-keypoint confidence scores and computes region-level motion scores and velocities.

| | |
|:--|:--|
| **Strengths** | Most detailed pose estimation available on CPU. Face mesh enables detection of facial automatisms. Hand keypoints can detect fine motor movements. Apache 2.0 license |
| **Weaknesses** | Very slow on CPU (~1 fps). Would benefit significantly from GPU acceleration. 35x slower than MediaPipe for ~4x more keypoints — tradeoff depends on whether the extra detail is needed |

### Extracted Measures — Keypoint Positions and Velocities

21 representative keypoints saved to CSV, each with 4 columns (**84 columns total**). All positions are in pixels, velocities in px/frame.

| Keypoint | `{name}_x` | `{name}_y` | `{name}_score` | `{name}_velocity` |
|:---------|:----------:|:----------:|:--------------:|:-----------------:|
| `nose` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_eye` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_eye` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_ear` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_ear` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_shoulder` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_shoulder` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_elbow` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_elbow` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_wrist` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_wrist` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_hip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_hip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_knee` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_knee` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_ankle` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_ankle` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_big_toe` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_big_toe` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `left_hand_middle_tip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |
| `right_hand_middle_tip` | X position | Y position | Confidence (0-1) | Frame-to-frame displacement |

*The full 133-keypoint model includes 68 face landmarks + 21 per hand, but only the above representative keypoints are saved to CSV.*

### Extracted Measures — Region Confidence Scores

| Column | Type | Description |
|:-------|:----:|:------------|
| `body_score` | region | Mean confidence across 17 body keypoints |
| `face_score` | region | Mean confidence across 68 face keypoints |
| `left_hand_score` | region | Mean confidence across 21 left hand keypoints |
| `right_hand_score` | region | Mean confidence across 21 right hand keypoints |
| `feet_score` | region | Mean confidence across 6 feet keypoints |

### Extracted Measures — Aggregate and Metadata

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `num_people` | metadata | Number of people detected in the frame |
| `pose_detected` | metadata | Boolean — was a person found? |
| `total_keypoints_visible` | metadata | Count of keypoints with score above threshold (out of 133) |
| `mean_wrist_velocity` | aggregate | Average of left + right wrist velocity (px/frame) |
| `mean_shoulder_velocity` | aggregate | Average of left + right shoulder velocity (px/frame) |
| `mean_hand_velocity` | aggregate | Average of left + right middle fingertip velocity (px/frame) |
| `mean_body_velocity` | aggregate | Average velocity across all saved keypoints (px/frame) |

---

## **7. Pose-Guided ROI Motion Energy**

| | |
|:--|:--|
| **Script** | `scripts/extract_roi_motion.py` (in seizure-semiology-mapping repo) |
| **Speed** | ~120 fps (CPU) |
| **Output** | `seizure-semiology-mapping/data/EM1334/` |
| **Depends on** | Pre-computed MediaPipe pose CSVs + `opencv-python` |

Combines pose estimation with frame differencing. Uses MediaPipe keypoint positions to define adaptive bounding boxes around 4 body regions, then computes frame differencing within each ROI separately. The ROIs move with the patient as they shift position.

| Region | Keypoints Used | Padding |
|:-------|:---------------|:-------:|
| **Head** | nose, left/right eye, left/right ear | +30 px |
| **Left arm** | left shoulder, left elbow, left wrist | +30 px |
| **Right arm** | right shoulder, right elbow, right wrist | +30 px |
| **Torso** | left/right shoulder, left/right hip | +30 px |

| | |
|:--|:--|
| **Strengths** | Best of both worlds — pixel-level motion sensitivity (works through blankets) with body-region specificity from pose estimation. Can detect lateralized movement (left arm vs. right arm). Adaptive ROIs follow the patient |
| **Weaknesses** | Requires pre-computed pose CSVs. Coverage limited by pose estimation sample rate. ROI accuracy depends on pose estimation accuracy. NaN for frames where pose data is unavailable |

### Extracted Measures — Global

| Column | Type | Description |
|:-------|:----:|:------------|
| `frame` | index | Frame number |
| `time_sec` | index | Timestamp in seconds |
| `global_mean_diff` | raw | Full-frame mean pixel difference (same as basic frame diff) |
| `global_motion_area` | raw | Full-frame fraction of pixels above motion threshold |
| `global_mean_diff_smooth` | smooth | Smoothed `global_mean_diff` (0.5s Gaussian FWHM) |
| `global_motion_area_smooth` | smooth | Smoothed `global_motion_area` |

### Extracted Measures — Per Region

4 regions, each with 4 columns (**16 columns total**):

| Column | Type | Regions | Description |
|:-------|:----:|:--------|:------------|
| `{region}_mean_diff` | raw | head, left_arm, right_arm, torso | Mean pixel difference within the region's bounding box |
| `{region}_motion_area` | raw | head, left_arm, right_arm, torso | Fraction of ROI pixels above motion threshold |
| `{region}_mean_diff_smooth` | smooth | head, left_arm, right_arm, torso | Smoothed `mean_diff` for the region |
| `{region}_motion_area_smooth` | smooth | head, left_arm, right_arm, torso | Smoothed `motion_area` for the region |

---

## Column Counts by Method

| # | Method | Raw | Smoothed | Aggregate | Metadata | Region | Keypoint (x, y, conf, vel) | **Total** |
|:-:|:-------|:---:|:--------:|:---------:|:--------:|:------:|:--------------------------:|:---------:|
| 1 | Frame Differencing | 3 | 1 | — | — | — | — | **4** |
| 2 | DIS Optical Flow | 7 | 1 | — | — | — | — | **8** |
| 3 | Background Subtraction | 3 | 2 | — | — | — | — | **5** |
| 4 | MEI/MHI | 5 | 4 | — | — | — | — | **9** |
| 5 | MediaPipe Pose | — | — | 3 | 1 | — | 13 x 4 = 52 | **56** |
| 6 | RTMLib Wholebody | — | — | 4 | 3 | 5 | 21 x 4 = 84 | **96** |
| 7 | ROI Motion Energy | 10 | 10 | — | — | — | — | **20** |
| | | | | | | | **Grand total** | **~198** |

*All columns exclude `frame` and `time_sec` index columns, which are present in every CSV.*
