# Movement Feature Extraction Tools — Comprehensive Comparison

## Overview

This document surveys tools for extracting movement features from video for EMU (Epilepsy Monitoring Unit) seizure semiology analysis. The target environment: fixed overhead camera, patient in hospital bed under blankets (significant lower-body occlusion), variable lighting (day/night/IR), nurses and visitors entering frame, and long continuous recordings (hours to days).

Tools are organized into four categories:
1. [Pose Estimation](#1-pose-estimation)
2. [Optical Flow](#2-optical-flow)
3. [Motion Energy / Pixel-Based](#3-motion-energy--pixel-based)
4. [Body Tracking & Action Recognition](#4-body-tracking--action-recognition)

---

## 1. Pose Estimation

### Summary Table

| Tool | Keypoints | Multi-Person | 3D | Install | CPU Real-time | License | Accuracy (COCO AP) | Active Dev | EMU Rating |
|------|-----------|:---:|:---:|---------|:---:|---------|:---:|:---:|:---:|
| **RTMPose/rtmlib** | 17-133 | Yes | Via RTMW | `pip install rtmlib` | Yes (90+ FPS) | Apache 2.0 | 75.8 | Yes | **Best** |
| **MediaPipe Pose** | 33 | No | Relative Z | `pip install mediapipe` | Yes (16-18 FPS) | Apache 2.0 | ~84%* | Yes | Good prototype |
| **ViTPose** | 17-133 | Via detector | No | HuggingFace or MMPose | No | Apache 2.0 | 80.9 | Yes | Best accuracy |
| **YOLO11-Pose** | 17 | Yes (built-in) | No | `pip install ultralytics` | Yes (Nano) | AGPL-3.0 | ~75 | Very Active | Good detection |
| **MMPose** | 17-133+ | Yes | Via models | `mim install mmpose` | Via RTMPose | Apache 2.0 | Up to 80.9 | Yes | Best framework |
| **OpenPose** | 25+face+hands | Yes | No | C++ build | No (1-3 FPS) | Non-commercial | ~70 | Stalled | Legacy |
| **AlphaPose** | 17-136 | Yes | Via HybrIK | Build from source | No | GPL-3.0 | 75 | Stalled | 136-kp model |
| **HRNet** | 17 | Via detector | No | Research repo | No | MIT | ~75.5 | Archival | Use via MMPose |
| **MoveNet** | 17 | Up to 6 | No | `pip install tensorflow` | Yes (30+ FPS) | Apache 2.0 | Competitive | Stable | Lightweight |
| **Sapiens** | Variable | No | Depth map | Research repo | No | CC-BY-NC | SOTA | Yes | Research only |

*MediaPipe uses different evaluation methodology; not directly comparable to COCO AP.*

### Top Recommendations

**Primary: RTMPose/RTMW via rtmlib**
- 133-keypoint whole-body estimation (body + face + hands + feet)
- Real-time on CPU (90+ FPS), trivial install (`pip install rtmlib`)
- Apache 2.0 license, actively maintained
- Best overall balance for this project

**For detection/tracking: YOLO11-Pose**
- Best multi-person detection with built-in tracking (ByteTrack/BoTSORT)
- Handles nurses/visitors entering frame
- Use for person detection, then feed crops to RTMPose for detailed keypoints

**For maximum accuracy (offline): ViTPose via HuggingFace**
- State-of-the-art accuracy (80.9 AP on COCO)
- Best for detailed analysis of flagged seizure segments
- Requires GPU

**For rapid prototyping: MediaPipe Pose**
- 3 lines of code to get running, no GPU needed
- Good for initial feasibility testing

### Key Consideration: Blanket Occlusion
No pose estimator handles blanket occlusion well out-of-the-box. Strategies:
- Focus on visible upper body (head, face, arms, hands)
- Use temporal tracking to interpolate across brief occlusions
- Consider custom training (DeepLabCut, SLEAP) on EMU-specific data
- Combine with motion energy for occluded regions (blanket movement as proxy)

---

## 2. Optical Flow

### Summary Table

| Method | Type | Dense/Sparse | Sintel Clean EPE | Speed (FPS) | GPU Required | Install | License | EMU Rating |
|--------|------|:---:|:---:|:---:|:---:|---------|---------|:---:|
| **DIS (OpenCV)** | Classical | Dense | ~3-5 | 15-50 (CPU) | No | `pip install opencv-python` | Apache 2.0 | **Best classical** |
| **Farneback** | Classical | Dense | ~5-7 | 5-15 (CPU) | No | `pip install opencv-python` | Apache 2.0 | Good prototype |
| **Lucas-Kanade** | Classical | Sparse | N/A | >100 (CPU) | No | `pip install opencv-python` | Apache 2.0 | Point tracking |
| **RAFT** | Deep | Dense | 1.43 | 10-15 (GPU) | Yes | PyTorch repo | BSD 3-Clause | **Best deep** |
| **RAFT-Small** | Deep | Dense | 2.21 | 25-30 (GPU) | Yes | PyTorch repo | BSD 3-Clause | Fast deep |
| **GMFlow** | Deep | Dense | 1.08 | 10-15 (GPU) | Yes | PyTorch repo | Apache 2.0 | Large displacements |
| **UniMatch** | Deep | Dense | ~1.05 | 7-12 (GPU) | Yes | PyTorch repo | Apache 2.0 | Maximum accuracy |
| **GMA** | Deep | Dense | 1.30 | 8-12 (GPU) | Yes | PyTorch repo | Check repo | Occlusion handling |
| **PWC-Net** | Deep | Dense | 2.55 | 30-35 (GPU) | Yes | PyTorch repo | Non-commercial | Superseded |
| **FlowNet2** | Deep | Dense | 2.02 | 8-10 (GPU) | Yes | Complex build | Mixed | Legacy only |

### Top Recommendations

**Start here: DIS Optical Flow (OpenCV)**
- Best classical dense flow — faster and more accurate than Farneback
- Zero setup beyond `pip install opencv-python`
- CPU-only, processes 24h of video in ~15-20 minutes
- Perfect for building and validating the feature extraction pipeline

**Upgrade for accuracy: RAFT**
- Best balance of accuracy, ease of use, and license (BSD 3-Clause)
- Use `raft-things.pth` weights for novel domains like EMU video
- Clean PyTorch codebase, no custom CUDA ops

**For fast seizure movements: GMFlow or UniMatch**
- Global matching handles large displacements better than iterative methods
- Apache 2.0 license

### Deriving Movement Features from Flow

From a dense flow field `(H, W, 2)` per frame pair, extract:

| Feature | Clinical Relevance |
|---------|-------------------|
| Mean magnitude | Overall movement intensity |
| Regional magnitude (per ROI) | Which body region is moving |
| Lateralization index | (right - left) / (right + left) — lateralizing sign |
| Dominant direction | Direction of movement |
| Direction consistency | Uniform (tonic) vs. chaotic (hyperkinetic) |
| Rhythmic frequency (FFT) | Clonic frequency (typically 3-8 Hz) |
| Motion area fraction | Focal vs. generalized |
| Magnitude derivative | Acceleration / deceleration |

### Processing Strategy for Long Recordings
Use a two-pass approach:
1. **Fast pass (CPU):** Frame differencing or DIS to identify time windows with significant motion
2. **Detailed pass (GPU):** RAFT on flagged segments only — reduces computation 10-100x

---

## 3. Motion Energy / Pixel-Based

### Methods

| Method | How It Works | Speed | Setup | Best For |
|--------|-------------|-------|-------|----------|
| **Frame Differencing** | `|F(t) - F(t-1)|` | 200-500 FPS | 5 lines of code | Simplest baseline |
| **MEI/MHI** | Temporal accumulation of motion | 150-400 FPS | `opencv-contrib-python` | Where + when motion occurred |
| **Motion Energy Analysis** | Frame diff + ROIs + smoothing | 200-500 FPS | ~30 lines | Regional movement quantification |
| **Background Subtraction (MOG2)** | Statistical background model | 80-200 FPS | OpenCV built-in | Separating patient from background |
| **Background Subtraction (KNN)** | Sample-based background model | 60-150 FPS | OpenCV built-in | Handling gradual lighting changes |

### Key Strengths for EMU
- **Works regardless of occlusion** — detects any pixel change, including blanket movement
- **Extremely fast** — can process 24h of video in minutes
- **No training required**
- **ROI-based analysis** provides body-region specificity without pose estimation

### Key Weaknesses
- Cannot distinguish patient movement from nurse/visitor movement
- Cannot distinguish movement type (tonic vs. clonic vs. artifact)
- Sensitive to lighting changes, especially IR mode switching
- Absolute values depend on resolution/exposure — cross-session comparison requires normalization

### Noise Mitigation for EMU Video

| Noise Source | Mitigation |
|-------------|------------|
| Camera sensor noise | Gaussian blur (21x21) before differencing |
| Compression artifacts | Threshold above noise floor |
| Lighting changes | Background subtraction (MOG2); detect global intensity shifts |
| IR mode switching | Detect transition frames, interpolate across |
| Nurses entering frame | Mask bed region only; flag large-area motion as artifact |
| Monitor/display flicker | Exclude monitor region from ROI |

### Recommended Tiered Approach
1. **Tier 1:** Global motion energy (frame differencing + Gaussian smoothing)
2. **Tier 2:** ROI-based motion energy (head, arms, torso regions)
3. **Tier 3:** Frequency analysis (FFT/wavelet of motion energy for clonic detection)
4. **Tier 4:** Pose estimation on candidate segments identified by motion energy

---

## 4. Body Tracking & Action Recognition

### Research-Oriented Tracking

| Tool | What It Does | Custom Training | License | EMU Rating |
|------|-------------|:---:|---------|:---:|
| **DeepLabCut** | Custom keypoint tracking | Yes (100-200 labeled frames) | LGPL v3 | Moderate-High |
| **SLEAP** | Multi-animal/person pose tracking | Yes (fast: 15-60 min) | BSD 3-Clause | Moderate-High |
| **B-SOiD** | Unsupervised behavior segmentation from pose | Auto-unsupervised | BSD 3-Clause | High (exploratory) |

### Action Recognition Models

| Tool | Architecture | Long-Range Temporal | Efficiency | License | EMU Rating |
|------|-------------|:---:|:---:|---------|:---:|
| **SlowFast** | Dual-pathway CNN | Moderate (64 frames) | Moderate | Apache 2.0 | Moderate-High |
| **TimeSformer** | Pure Transformer | **High (96+ frames)** | High | CC-NC 4.0 | High (NC limit) |
| **Video Swin** | Shifted-window Transformer | Moderate (32 frames) | Good | MIT | Moderate-High |
| **X3D** | Efficient 3D CNN | Moderate | **Very High** | Apache 2.0 | High |
| **MMAction2** | Framework (30+ models) | Varies | Varies | Apache 2.0 | **Very High** |
| **PyTorchVideo** | Framework (models + data) | Varies | Varies | Apache 2.0 | High |

### Seizure-Specific Published Research

| Approach | Accuracy | Reference |
|----------|----------|-----------|
| ViTPose + MotionBERT (5-class) | 198 seizures, 74 patients | ScienceDirect 2025 |
| CNN + LSTM (tonic-clonic) | 88% sensitivity, 92% specificity | Multiple papers |
| 3D CNN I3D/SlowFast | Hyperkinetic, tonic, tonic-clonic | Nature Sci. Reports 2022 |
| Optical flow + catch22 clustering | 91% hyperkinetic, 88% tonic | Frontiers Neurol. 2023 |
| Privacy-preserving (flow + normals) | GTCS vs. PNES | Springer 2025 |
| **Nelli (commercial)** | 93.7% major convulsive | FDA cleared Dec 2025 |

### B-SOiD: Unsupervised Behavior Discovery
Particularly promising for seizure analysis:
- Takes pose estimation output, discovers behavioral categories automatically
- No manual behavior labeling needed
- Already applied to seizure behavior in animal models (identified 63 behavior groups)
- Could discover movement motifs (tonic posturing, clonic jerking, automatisms) from pose trajectories
- BSD 3-Clause license

---

## 5. Overall Recommendations

### Recommended Pipeline for EMU Seizure Video

```
EMU Video (AVI/MP4)
    |
    v
[1. Motion Energy Screening] ---- Frame differencing (CPU, fast)
    |                              Identify active time windows
    v
[2. Person Detection + Tracking] - YOLO11-Pose (multi-person, built-in tracking)
    |                              Isolate patient vs. staff
    v
[3. Pose Estimation] ------------ RTMPose/DWPose via rtmlib (133 keypoints)
    |                              On active windows only
    v
[4. Feature Extraction] ---------- Velocity, joint angles, symmetry,
    |                              periodicity (FFT), lateralization
    v
[5. Movement Classification] ---- B-SOiD (unsupervised discovery)
    |                              OR MMAction2 skeleton-based (supervised)
    v
[6. Output] ---------------------- Time-series features + event labels
                                   Aligned with EEG timestamps
```

### Tool Priority for This Project

| Priority | Tool | Purpose | Why |
|:---:|------|---------|-----|
| 1 | **Motion energy (OpenCV)** | Baseline movement quantification | Instant setup, works with occlusion, fast |
| 2 | **MediaPipe Pose** | Quick pose estimation prototype | 3-line setup, CPU-only, 33 keypoints |
| 3 | **RTMPose via rtmlib** | Production pose estimation | 133 keypoints, real-time CPU, Apache 2.0 |
| 4 | **DIS Optical Flow (OpenCV)** | Dense motion fields | Best classical flow, CPU-only |
| 5 | **RAFT** | High-accuracy optical flow | Best deep flow, BSD license |
| 6 | **YOLO11-Pose** | Multi-person detection + tracking | Handle nurses/visitors |
| 7 | **B-SOiD** | Behavior discovery | Unsupervised seizure motif detection |
| 8 | **MMAction2** | Action classification | Full framework for supervised classification |

### Hardware Requirements

| Approach | GPU Needed | Processing Time (24h video) |
|----------|:---:|:---:|
| Motion energy | No | ~5-10 minutes |
| MediaPipe Pose | No | ~24-48 hours (CPU) |
| RTMPose (rtmlib) | No (CPU OK) | ~4-8 hours (CPU) |
| DIS optical flow | No | ~15-20 minutes |
| RAFT | Yes | ~20-40 minutes (GPU) |
| YOLO11-Pose | Optional | ~2-4 hours (CPU) / ~30 min (GPU) |

---

## 6. Key References

### Seizure Video Analysis
- Brown et al. (2024) "Computer vision for automated seizure detection" — *Epilepsia*
- "Deep learning approaches for seizure video analysis" — *Epilepsy & Behavior*, Mar 2024
- "Automated seizure detection in video" — *Frontiers in Neuroinformatics*, 2024
- ViTPose + MotionBERT 5-class seizure classification — *ScienceDirect*, 2025
- 3D CNN seizure classification — *Nature Scientific Reports*, 2022
- Motion feature clustering for seizure types — *Frontiers in Neurology*, 2023
- Privacy-preserving seizure detection — *Springer*, 2025
- B-SOiD seizure behavior decoding — *Annals of Neurology*, 2025
- Nelli (Neuro Event Labs) — FDA cleared Dec 2025

### Tool Documentation
- [RTMPose / rtmlib](https://github.com/Tau-J/rtmlib)
- [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [YOLO11-Pose](https://github.com/ultralytics/ultralytics)
- [MMPose](https://github.com/open-mmlab/mmpose)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMFlow](https://github.com/haofeixu/gmflow)
- [UniMatch](https://github.com/autonomousvision/unimatch)
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)
- [SLEAP](https://sleap.ai/)
- [B-SOiD](https://github.com/YttriLab/B-SOID)
- [MMAction2](https://github.com/open-mmlab/mmaction2)
- [PyTorchVideo](https://pytorchvideo.org/)

### Motion Energy in Clinical Research
- Ramseyer & Tschacher (2011) — Motion energy analysis methodology
- Kalitzin et al. (2012) — Motion energy for clonic seizure detection
- Geertsema et al. (2018) — Automated video-based nocturnal seizure detection
- Karayiannis et al. (2005-2006) — Neonatal seizure motion features
