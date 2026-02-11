# Movement Feature Extraction — Method Priority (Simplest to Most Complex)

## Tier 1: Instant Setup (pip install + <10 lines of code, CPU only)

| # | Method | Setup Time | Processing Speed (24h video) | What You Get |
|---|--------|-----------|---------------------------|-------------|
| 1 | **Frame Differencing** | 5 min | ~5 min | Global motion intensity time series |
| 2 | **Motion Energy Analysis (ROI-based)** | 15 min | ~5-10 min | Per-region motion intensity |
| 3 | **Background Subtraction (MOG2/KNN)** | 15 min | ~10-15 min | Foreground masks + motion time series |
| 4 | **DIS Optical Flow** | 15 min | ~15-20 min | Dense motion vectors (direction + magnitude) |
| 5 | **Farneback Optical Flow** | 15 min | ~30-60 min | Dense motion vectors (slightly less accurate) |
| 6 | **Lucas-Kanade Sparse Flow** | 20 min | ~10 min | Tracked point displacements |

All of these use only `pip install opencv-python` — nothing else needed.

## Tier 2: Easy Setup (pip install, CPU, pre-trained models)

| # | Method | Setup Time | Processing Speed (24h video) | What You Get |
|---|--------|-----------|---------------------------|-------------|
| 7 | **MediaPipe Pose** | 10 min | ~24-48h (CPU) | 33 keypoints per frame (single person) |
| 8 | **RTMPose via rtmlib** | 15 min | ~4-8h (CPU) | 17-133 keypoints (multi-person capable) |
| 9 | **MoveNet** | 20 min | ~12-24h (CPU) | 17 keypoints (single person) |

## Tier 3: Moderate Setup (pip install, GPU recommended)

| # | Method | Setup Time | Processing Speed (24h video) | What You Get |
|---|--------|-----------|---------------------------|-------------|
| 10 | **YOLO11-Pose** | 15 min | ~30 min (GPU) / 2-4h (CPU) | 17 keypoints + person detection + tracking |
| 11 | **RAFT Optical Flow** | 30 min | ~20-40 min (GPU) | High-accuracy dense motion vectors |
| 12 | **GMFlow / UniMatch** | 45 min | ~30-50 min (GPU) | Best-accuracy dense motion vectors |

## Tier 4: Complex Setup (multiple dependencies, GPU required)

| # | Method | Setup Time | Processing Speed (24h video) | What You Get |
|---|--------|-----------|---------------------------|-------------|
| 13 | **ViTPose (HuggingFace)** | 30 min | ~2-8h (GPU) | State-of-the-art keypoints (up to 133) |
| 14 | **MMPose (full framework)** | 1-2h | ~1-8h (GPU) | Flexible pose with many model choices |
| 15 | **DeepLabCut** | 1-2h + labeling | ~4-12h (GPU) | Custom keypoints (requires training) |
| 16 | **SLEAP** | 1-2h + labeling | ~2-6h (GPU) | Custom keypoints (requires training) |

## Tier 5: Research Pipeline (labeled data + training required)

| # | Method | Setup Time | Processing Speed | What You Get |
|---|--------|-----------|-----------------|-------------|
| 17 | **B-SOiD** | 2-3h (after pose) | Minutes (CPU) | Unsupervised behavior categories |
| 18 | **MMAction2** | 3-5h + labeled data | Hours (GPU) | Supervised action classification |
| 19 | **SlowFast / X3D** | 3-5h + labeled data | Hours (GPU) | Video-level action recognition |
| 20 | **TimeSformer** | 3-5h + labeled data | Hours (GPU) | Long-range temporal action recognition |

---

## Recommended Order for This Project

1. **Frame differencing** — get a baseline motion signal in minutes
2. **DIS optical flow** — upgrade to directional motion, still CPU-only
3. **MediaPipe Pose** — first skeleton overlay, see if keypoints are usable with blanket occlusion
4. **RTMPose/rtmlib** — upgrade to 133 keypoints if MediaPipe looks promising
5. **RAFT** — high-quality optical flow for detailed motion analysis (needs GPU)
6. **B-SOiD** — feed pose data in, discover movement categories automatically

Each step builds on the previous one, and you can stop at any tier once you have enough signal for your analysis.
