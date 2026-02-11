# Movement Feature Extraction — Project Plan

## Goal
Evaluate tools for extracting movement features from video, test them on sample data, and build side-by-side visualizations comparing raw video with extracted movement time series.

## Phases

### Phase 1: Research Tools [COMPLETE]
- [x] Survey pose estimation tools (MediaPipe, OpenPose, MMPose, ViTPose, RTMPose, YOLO11-Pose, etc.)
- [x] Survey optical flow methods (DIS, Farneback, RAFT, GMFlow, UniMatch, etc.)
- [x] Survey motion energy / pixel-change approaches
- [x] Survey full-body tracking and action recognition tools (DeepLabCut, SLEAP, B-SOiD, SlowFast, MMAction2, etc.)
- [x] Document each tool: capabilities, input/output format, ease of use, GPU requirements, license
- [x] Write comparison summary in `research/tool_comparison.md`

### Phase 2: Set Up and Test Tools
- [ ] Install top candidate tools
- [ ] Run each tool on sample video(s)
- [ ] Extract movement feature time series from each tool
- [ ] Validate outputs — check for dropped frames, artifacts, scaling issues
- [ ] Save extracted features in standardized format (CSV or Parquet)

### Phase 3: Build Visualizations
- [ ] Build script to play video synced with time-series plots
- [ ] Overlay skeleton/keypoints on video frames (for pose tools)
- [ ] Plot movement magnitude, velocity, joint angles over time
- [ ] Create side-by-side comparison layout (video left, plots right)
- [ ] Export comparison as video file (MP4) for presentations

### Phase 4: Evaluate and Document
- [ ] Compare tool accuracy, speed, and usability
- [ ] Identify best tool(s) for EMU seizure video analysis
- [ ] Write final recommendation and documentation
