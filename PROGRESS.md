# Movement Feature Extraction — Progress Log

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

### Next Steps
- Create rtmlib vs MediaPipe comparison visualization
- Evaluate which methods are most useful for seizure detection
- Consider GPU acceleration for rtmlib if scaling to full dataset
- Test on more diverse video segments (night mode, nurse visits, seizure events)

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
