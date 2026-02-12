# Movement Feature Extraction Methods — Overview

This document explains each motion extraction method tested on EM1334 video data, plus methods identified but not yet run due to hardware constraints.

## Methods Tested on EM1334

### 1. Frame Differencing
**Script:** `scripts/extract_frame_diff.py`
**Speed:** ~100 fps (CPU)

The simplest motion detection method. Computes the absolute pixel difference between consecutive frames after Gaussian blur (21x21 kernel to reduce noise). Pixels that changed significantly between frames indicate movement.

**Outputs:**
- `mean_diff` — Average pixel intensity change across the entire frame. Higher = more overall movement.
- `motion_area_frac` — Fraction of pixels exceeding a change threshold (>10). Indicates how much of the frame is moving.
- `mean_diff_smooth` — Gaussian-smoothed version (0.5s window) for cleaner signal.

**Strengths:** Extremely fast, no setup, works regardless of occlusion (detects blanket movement too).
**Weaknesses:** Cannot distinguish patient from nurse movement. Sensitive to lighting changes and compression artifacts. No spatial specificity — just one number per frame.

---

### 2. DIS Optical Flow
**Script:** `scripts/extract_dis_flow.py`
**Speed:** ~11 fps (CPU)

Dense Inverse Search (DIS) is an efficient dense optical flow algorithm. Unlike frame differencing which only detects *that* pixels changed, optical flow estimates *how* they moved — producing a 2D motion vector (direction + magnitude) for every pixel.

**Outputs:**
- `mean_magnitude` — Average motion vector length across all pixels. Captures overall movement intensity.
- `mean_angle` — Average motion direction (radians). Can indicate dominant movement direction.
- `motion_area` — Fraction of pixels with significant flow magnitude.
- `mean_mag_smooth` — Smoothed version.

**Strengths:** Richer than frame differencing — captures both speed and direction. Can detect flow patterns (e.g., rhythmic back-and-forth for clonic movements). Good balance of accuracy and CPU speed.
**Weaknesses:** Still global (no per-region breakdown by default). Slower than frame differencing. Classical method — less accurate than deep learning approaches on complex scenes.

---

### 3. Background Subtraction (MOG2)
**Script:** `scripts/extract_bg_subtraction.py`
**Speed:** ~60-77 fps (CPU)

MOG2 (Mixture of Gaussians) learns a statistical model of the background over time. Each pixel is modeled as a mixture of Gaussian distributions. Pixels that don't fit the learned background model are classified as "foreground" (moving objects). Unlike frame differencing, this adapts to gradual changes (lighting shifts, IR mode switching) because the background model updates continuously.

**Parameters:**
- `history=500` — Number of frames used to build the background model.
- `varThreshold=50` — Higher = less sensitive to small changes (reduces noise).
- Morphological open/close operations clean up the foreground mask.

**Outputs:**
- `foreground_frac` — Fraction of pixels classified as foreground (moving). Higher = more of the frame contains movement.
- `foreground_area_px` — Raw pixel count of foreground region.
- `foreground_mean_intensity` — Average brightness of foreground pixels. Can indicate whether the moving region is the patient (bright) or a shadow (dark).
- Smoothed versions of frac and intensity.

**Strengths:** Adapts to slow lighting changes (better than frame differencing for long recordings). Distinguishes sustained movement from transient noise. Fast on CPU.
**Weaknesses:** Takes ~500 frames (~17s at 30fps) to learn the initial background. Large scene changes (nurse entering) temporarily confuse the model. Still no per-body-region breakdown.

---

### 4. Motion Energy Images (MEI) / Motion History Images (MHI)
**Script:** `scripts/extract_mei_mhi.py`
**Speed:** ~58-60 fps (CPU)

MEI and MHI are temporal motion templates that summarize movement over a sliding time window.

- **MEI (Motion Energy Image):** Binary image showing *where* any motion occurred within the window. Each pixel is 1 if it moved at any point during the window, 0 otherwise.
- **MHI (Motion History Image):** Grayscale image encoding *when* motion occurred. Brighter pixels = more recent motion. Older motion fades out over the window duration.

The MHI window duration is configurable (default: 2 seconds). This means the MHI "remembers" the last 2 seconds of motion at any given time.

**Outputs:**
- `motion_energy` — Instantaneous mean frame difference (per-frame).
- `motion_area` — Fraction of pixels with motion above threshold.
- `mei_frac` — Fraction of the frame that moved within the MHI window. High values = widespread or sustained movement.
- `mhi_mean` — Mean MHI intensity (0-1). High values = lots of recent movement across the frame.
- `mhi_max` — Peak MHI intensity. Indicates the most recently active pixel region.
- Smoothed versions.
- **Snapshot images** saved every 30 seconds: side-by-side original frame + color-coded MHI heatmap (JET colormap: blue=old motion, red/yellow=recent motion).

**Strengths:** Captures temporal dynamics — not just "is there motion?" but "how has motion evolved over the last N seconds?" Useful for detecting sustained movements (tonic posturing) vs. brief bursts (myoclonic jerks). The snapshot images provide intuitive visual summaries.
**Weaknesses:** The window duration is a fixed parameter that may not suit all movement types. Doesn't separate patient from other movement sources.

---

### 5. MediaPipe Pose Estimation
**Script:** `scripts/extract_mediapipe.py`
**Speed:** ~35 fps (CPU)

Google's MediaPipe Holistic estimates 33 body keypoints (pose landmarks) per frame. Each keypoint has an (x, y) position in pixels and a visibility confidence score. The script also computes frame-to-frame velocities for each keypoint.

**Outputs (per frame):**
- 13 upper-body keypoints with x, y, visibility: nose, eyes, ears, shoulders, elbows, wrists, hips.
- Per-keypoint velocity (pixel distance moved since previous frame).
- Aggregate velocities: `mean_wrist_velocity`, `mean_shoulder_velocity`, `mean_body_velocity`.
- `pose_detected` — Boolean indicating whether a person was found.

**Strengths:** Provides body-part-specific motion signals (e.g., wrist velocity separately from head velocity). This is critical for seizure semiology where the *location* of movement matters (lateralized arm jerking vs. head turning). 33 keypoints, fast on CPU, trivial install.
**Weaknesses:** Only 33 keypoints (no face mesh, no individual fingers). Single-person only — cannot handle nurse entering frame. Trained on standing people at eye level, so accuracy degrades with overhead camera angles and supine patients.

---

### 6. RTMLib Wholebody Pose Estimation
**Script:** `scripts/extract_rtmlib.py`
**Speed:** ~1 fps (CPU)

RTMPose via rtmlib provides 133 keypoints covering the entire body: 17 body, 68 face, 21 per hand (42 total), and 6 feet. This is 4x the detail of MediaPipe. Includes per-keypoint confidence scores and computes region-level motion scores and velocities.

**Outputs (per frame):**
- 133 keypoints with x, y, confidence.
- Region scores: face, left/right hand, body — indicating tracking quality per region.
- Per-keypoint and aggregate velocities.
- Number of visible keypoints (out of 133).

**Strengths:** Most detailed pose estimation available on CPU. Face mesh enables detection of facial automatisms. Hand keypoints can detect fine motor movements. Apache 2.0 license.
**Weaknesses:** Very slow on CPU (~1 fps). Would benefit significantly from GPU acceleration. 35x slower than MediaPipe for ~4x more keypoints — tradeoff depends on whether the extra detail is needed.

---

### 7. Pose-Guided ROI Motion Energy
**Script:** `scripts/extract_roi_motion.py` (in seizure-semiology-mapping repo)
**Speed:** ~120 fps (CPU)

Combines pose estimation with frame differencing. Uses MediaPipe keypoint positions to define adaptive bounding boxes around 4 body regions (head, left arm, right arm, torso), then computes frame differencing within each ROI separately. The ROIs move with the patient as they shift position.

**Region definitions:**
- **Head:** Bounding box around nose, eyes, ears (+ 30px padding)
- **Left arm:** Left shoulder, elbow, wrist
- **Right arm:** Right shoulder, elbow, wrist
- **Torso:** Both shoulders and both hips

**Outputs (per frame):**
- `global_mean_diff` — Full-frame mean pixel difference (same as basic frame diff).
- `{region}_mean_diff` — Mean pixel difference within each body region's ROI.
- `{region}_motion_area` — Fraction of ROI pixels above motion threshold.
- Smoothed versions of all signals (0.5s Gaussian FWHM).
- NaN for frames where pose data is unavailable.

**Strengths:** Best of both worlds — pixel-level motion sensitivity (works through blankets) with body-region specificity from pose estimation. Can detect lateralized movement (left arm vs. right arm). Adaptive ROIs follow the patient.
**Weaknesses:** Requires pre-computed pose CSVs. Coverage limited by pose estimation sample rate (e.g., 33% if MediaPipe was run on every 3rd frame). ROI accuracy depends on pose estimation accuracy.

---

## Methods Not Yet Tested (Require GPU)

### 8. RAFT Deep Optical Flow
**What it does:** State-of-the-art learned optical flow. Uses a neural network to estimate dense pixel-level motion between frames, with iterative refinement. Significantly more accurate than classical methods (DIS, Farneback) especially for fast, complex movements.
**Why it matters:** Better handles the fast, irregular movements seen in seizures (tonic-clonic jerking, hyperkinetic episodes). Sintel benchmark error: 1.43 EPE vs. ~3-5 for DIS.
**Requirements:** NVIDIA GPU, PyTorch. BSD 3-Clause license.
**Speed:** ~10-15 fps on GPU.
**Status:** Not run — no GPU on current machine.

### 9. YOLO11-Pose Multi-Person Detection + Tracking
**What it does:** Detects and tracks multiple people in the frame simultaneously, with built-in ByteTrack/BoTSORT tracking. Assigns each person a persistent ID across frames and estimates 17 pose keypoints per person.
**Why it matters:** Critical for separating patient movement from nurse/visitor movement. Currently, all methods treat the entire frame as one — a nurse adjusting the bed would appear as patient movement.
**Requirements:** Works on CPU (Nano model, slower) or GPU (faster). AGPL-3.0 license.
**Speed:** ~2-4 fps (CPU, Nano) or ~30+ fps (GPU).
**Status:** Not run. Could potentially run on CPU with the Nano model but would be slow.

### 10. RAFT or GMFlow for Seizure-Specific Flow Features
**What it does:** Extends RAFT/GMFlow optical flow with seizure-relevant derived features: rhythmic frequency detection via FFT of flow magnitude (clonic seizures typically 3-8 Hz), lateralization index (right-left asymmetry), flow direction consistency (uniform = tonic, chaotic = hyperkinetic).
**Why it matters:** These derived features directly map to clinical seizure semiology descriptors used by epileptologists.
**Requirements:** GPU for the flow computation; feature derivation is CPU.
**Status:** Not run — depends on RAFT/GMFlow.

### 11. B-SOiD Unsupervised Behavior Discovery
**What it does:** Takes pose estimation time-series as input and uses unsupervised machine learning to automatically discover clusters of movement patterns. Originally developed for animal behavior, it has been applied to seizure analysis.
**Why it matters:** Could automatically discover seizure movement motifs (tonic posturing, clonic jerking, automatisms) without manual labeling. BSD 3-Clause license.
**Requirements:** CPU only (runs on pose data, not video). Needs existing pose keypoint CSVs as input.
**Speed:** Fast (operates on pose CSVs, not video).
**Status:** Not run. **Could run on current machine** using existing MediaPipe or RTMLib pose data. Next priority.

### 12. Frequency Analysis (FFT/Wavelet) of Motion Signals
**What it does:** Applies spectral analysis (Fast Fourier Transform or wavelet decomposition) to motion time-series to detect periodic/rhythmic movement. Clonic seizures have a characteristic 3-8 Hz frequency that can be detected this way.
**Why it matters:** Rhythmic movement detection is one of the most reliable automated seizure signatures. Works on any motion signal (frame diff, optical flow, pose velocity).
**Requirements:** CPU only. Runs on existing extracted CSVs.
**Speed:** Near-instant (operates on 1D time-series).
**Status:** Not run. **Could run on current machine** using existing extraction outputs. High priority for seizure detection.

---

## Methods Comparison Summary

| Method | Speed (CPU) | Spatial Detail | Temporal Detail | Body-Part Specific | Handles Occlusion | GPU Needed |
|--------|:-----------:|:--------------:|:---------------:|:------------------:|:-----------------:|:----------:|
| Frame Differencing | 100 fps | None (global) | Per-frame | No | Yes (any pixel change) | No |
| DIS Optical Flow | 11 fps | Per-pixel direction | Per-frame | No | Yes | No |
| Background Subtraction | 60-77 fps | Foreground mask | Adaptive model | No | Yes | No |
| MEI/MHI | 58-60 fps | Spatial template | 2s sliding window | No | Yes | No |
| MediaPipe Pose | 35 fps | 33 keypoints | Per-frame velocity | Yes (13 upper body) | No | No |
| RTMLib Wholebody | 1 fps | 133 keypoints | Per-frame velocity | Yes (face+hands+body) | No | No |
| ROI Motion Energy | 120 fps | Per-region ROI | Per-frame + smoothed | Yes (4 regions) | Partial (within ROI) | No |
| RAFT Optical Flow | 10-15 fps | Per-pixel direction | Per-frame | No | Yes | **Yes** |
| YOLO11 Multi-Person | 2-30 fps | 17 keypoints/person | Tracking IDs | Yes + person separation | No | Optional |
| B-SOiD | N/A (post-hoc) | From pose input | Behavior segments | Yes (from pose) | No | No |
| Frequency Analysis | N/A (post-hoc) | From input signal | Spectral decomposition | Depends on input | Depends on input | No |

## Recommended Next Steps

1. **Frequency analysis** on existing motion signals — can run immediately, high value for seizure detection
2. **B-SOiD** on existing pose data — unsupervised behavior discovery, no GPU needed
3. **YOLO11-Pose Nano** on CPU — multi-person tracking to separate patient from staff
4. **RAFT** when GPU is available — highest quality optical flow for detailed seizure analysis
