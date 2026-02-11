# Movement Feature Extraction — Progress Log

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

### Next Steps
- Phase 2: Install top candidate tools and test on sample video
  - Start with motion energy (simplest, no dependencies)
  - Then MediaPipe or RTMPose for pose estimation
  - Then DIS or RAFT for optical flow
- Need sample EMU video to test against
