# Movement Feature Extraction — Progress Log

## 2026-02-11 — Phase 2 Continued: Additional Methods + Visualizations

### Completed
- Added 3 new extraction methods on EM1334 clips 0099-0101:

| Method | Script | Speed | Output |
|--------|--------|-------|--------|
| Background Subtraction (MOG2) | `extract_bg_subtraction.py` | ~60-77 fps | Foreground fraction, area, mean intensity |
| MEI/MHI Motion Templates | `extract_mei_mhi.py` | ~58-60 fps | Motion energy, MEI fraction, MHI mean/max + snapshot images |
| Pose-Guided ROI Motion | `extract_roi_motion.py` | ~120 fps | Per-region motion (head, L arm, R arm, torso) |

- Created side-by-side video for ROI motion: `visualizations/sidebyside_roi_0099.mp4`
  - Left panel: video with colored ROI bounding boxes around body regions
  - Right panel: scrolling per-region motion energy plots with red time cursor
- Created all-methods comparison plots for all 3 clips: `visualizations/all_methods_0099.png` etc.
  - 6 stacked subplots: frame diff, optical flow, BG subtraction, MEI/MHI, pose velocity, ROI motion
- Wrote comprehensive methods overview: `research/methods_overview.md`
  - Explains all 7 tested methods + 5 methods not yet run (RAFT, YOLO11, B-SOiD, frequency analysis, etc.)
- Renamed all output files from original patient name to EM1334
- Created dedicated venv for this repo with opencv-contrib-python, numpy, pandas, matplotlib, scipy

### All Methods Tested on EM1334 (7 total)

| # | Method | Script | Speed | Body-Part Specific | GPU |
|---|--------|--------|:-----:|:------------------:|:---:|
| 1 | Frame Differencing | `extract_frame_diff.py` | ~100 fps | No | No |
| 2 | DIS Optical Flow | `extract_dis_flow.py` | ~11 fps | No | No |
| 3 | MediaPipe Pose | `extract_mediapipe.py` | ~35 fps | Yes (33 kp) | No |
| 4 | RTMLib Wholebody | `extract_rtmlib.py` | ~1 fps | Yes (133 kp) | No |
| 5 | Background Subtraction | `extract_bg_subtraction.py` | ~60-77 fps | No | No |
| 6 | MEI/MHI Templates | `extract_mei_mhi.py` | ~58-60 fps | No | No |
| 7 | ROI Motion Energy | `extract_roi_motion.py` | ~120 fps | Yes (4 regions) | No |

### Key Observations
- All methods agree on overall activity pattern: quiet first ~40s, then increasing movement
- Background subtraction adapts to slow lighting changes better than frame differencing
- MHI provides temporal memory — captures sustained vs. transient motion
- ROI motion + pose velocity provide the most clinically relevant signals (body-part-specific)
- No GPU available on this machine — RAFT, YOLO11 full, and other GPU methods deferred

### Next Steps
- [ ] Frequency analysis (FFT/wavelet) on existing motion signals — detect clonic rhythms
- [ ] B-SOiD unsupervised behavior discovery on existing pose data
- [ ] YOLO11-Pose Nano on CPU for multi-person tracking
- [ ] Test on more diverse video segments (night mode, nurse visits, seizure events)
- [ ] RAFT deep optical flow when GPU is available

---

## 2026-02-11 — Phase 2: Testing Extraction Methods

### Completed
- Copied 104 AVI files from EM1334 to local temp folder for testing
- Created virtual environment with opencv-python, numpy, pandas, matplotlib, scipy, mediapipe, rtmlib
- Tested 4 extraction methods on first 3 videos (clips 0099-0101, 1280x720, 30fps, ~2 min each):

| Method | Script | Speed | Output |
|--------|--------|-------|--------|
| Frame Differencing | `extract_frame_diff.py` | ~100 fps | Mean pixel diff, motion area fraction |
| DIS Optical Flow | `extract_dis_flow.py` | ~11 fps | Flow magnitude, direction, motion area |
| MediaPipe Pose | `extract_mediapipe.py` | ~35 fps | 33 keypoints, velocities, visibility |
| RTMLib Wholebody | `extract_rtmlib.py` | ~1 fps | 133 keypoints (body+face+hands+feet), velocities, region scores |

### Key Results
- **Frame Differencing**: Fastest, good for screening. Detects all movement including camera noise.
- **DIS Optical Flow**: Good balance of speed and detail. Dense motion vectors per pixel.
- **MediaPipe Pose**: 33 body keypoints, 99-100% pose detection rate, fast on CPU.
- **RTMLib Wholebody**: 133 keypoints, 100% detection rate, 127/133 avg visible keypoints. 4x more detail than MediaPipe but 35x slower on CPU.

### Visualizations Created
- Static comparison plots (PNG) for all 3 videos: frame diff + DIS flow + MediaPipe velocities
- Side-by-side video (MP4) for first clip: skeleton overlay + scrolling time series
- RTMLib vs MediaPipe comparison visualization

---

## 2026-02-11 — Phase 1 Research Complete
- Created repository and project structure
- Completed comprehensive research across 4 tool categories:
  - **Pose estimation:** RTMPose/rtmlib (recommended), MediaPipe, ViTPose, YOLO11-Pose, MMPose, OpenPose, AlphaPose, HRNet, MoveNet, Sapiens
  - **Optical flow:** DIS (recommended classical), Farneback, Lucas-Kanade, RAFT (recommended deep), GMFlow, UniMatch, GMA, PWC-Net, FlowNet2
  - **Motion energy:** Frame differencing, MEI/MHI, Motion Energy Analysis with ROIs, MOG2/KNN background subtraction
  - **Body tracking & action recognition:** DeepLabCut, SLEAP, B-SOiD, SlowFast, TimeSformer, Video Swin, X3D, MMAction2, PyTorchVideo, Nelli (commercial)
- Identified key constraint: blanket occlusion means upper body (head, face, arms) is primary tracking target
- Found relevant seizure-specific published research (2022-2025)
- Wrote comprehensive comparison: `research/tool_comparison.md`

### Key Findings
- **Motion energy** is the fastest path to a working baseline (CPU-only, minutes to process 24h)
- **RTMPose via rtmlib** is the best pose tool (133 keypoints, CPU real-time, Apache 2.0, pip install)
- **RAFT** is the best deep optical flow (BSD license, clean code, no custom CUDA ops)
- **B-SOiD** is promising for unsupervised seizure behavior discovery from pose data
- Recommended pipeline: motion energy screening → person detection → pose estimation → feature extraction → classification
