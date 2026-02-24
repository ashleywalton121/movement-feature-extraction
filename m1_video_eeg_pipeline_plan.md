# M1 Video-EEG Movement Correlation Pipeline

## Project Goal

Correlate **movement features extracted from bedside video** with **M1 electrode activity from sEEG** across 28 patients. This requires identifying the right video segments (close framing, daytime, visible limb movement), confirming tight video-EEG synchronization, and then running the actual analysis.

---

## Participants

28 patients with electrodes in precentral/M1 regions (2–29 electrodes per patient). Patient IDs: EM1188 through EM1348.

---

## Phase 0: Understand the Data Layout

**Goal:** Map the directory structure so every downstream script knows where to look.

### What I need to confirm with you:

1. **Where does the data live?** Network drive, mounted share, cloud storage?
   - Path pattern, e.g., `//server/share/EMU/EM1188/` or `/mnt/natus/EM1188/`
2. **Study folder structure** — is it something like:
   ```
   EM1188/
   ├── Study_001/        # ~24hr recording block
   │   ├── *.avi         # video files (how many per study? 2-min chunks?)
   │   ├── *.eeg         # EEG data file
   │   ├── *.vtc         # video timestamp/sync file
   │   └── *.ent         # annotations?
   ├── Study_002/
   └── ...
   ```
3. **Naming conventions** — do AVI filenames encode timestamps, camera IDs, or study numbers?
4. **Multiple cameras?** Some EMUs have 2+ camera angles. Do your patients have multiple AVIs per timepoint (different views), or is it a single camera per study?
5. **How many study folders per patient** on average?
6. **Network access** — can you SSH into the server, or only access via mounted drive? (determines whether we use remote ffprobe vs. local ffprobe on a mount)

### Deliverable:
- `config.yaml` — a configuration file mapping patient IDs to their data paths, listing M1 electrode names, and defining all path patterns. Every script reads from this so paths are never hardcoded.

### Script: `00_build_config.py`
- Reads `precentral_m1_electrodes.csv`
- Walks the data root to discover study folders per patient
- Outputs `config.yaml` with the full inventory

---

## Phase 1: Remote Video Triage (No Copying)

**Goal:** For every AVI file across all 28 patients, extract metadata + thumbnail screenshots so you can visually assess framing, lighting, and content — all without copying multi-GB video files locally.

### Step 1.1: Probe metadata with ffprobe

For each AVI file, extract:
- Duration, resolution, frame rate, codec
- File size, creation/modification timestamp
- Total frame count

This tells you *when* each clip was recorded (daytime vs. night), how long it is, and whether it's a standard recording.

### Step 1.2: Extract thumbnail screenshots with ffmpeg

For each AVI file, pull 3 frames (start, middle, end) as JPEGs **directly from the network path** — ffmpeg can read from a mounted share without copying the whole file. Each thumbnail is ~50-100KB vs. the full AVI which may be hundreds of MB.

This tells you:
- **Framing/crop**: Is the camera zoomed in on the patient in bed, or is it a wide room shot?
- **Lighting**: Is it daytime (lights on) or nighttime (IR/dark)?
- **Patient position**: Are limbs visible? Is the patient under covers?
- **Obstructions**: Are staff, visitors, or equipment blocking the view?

### Step 1.3: Generate HTML contact sheets

Build a browsable HTML report organized by patient → study → video file, showing:
- Metadata table (duration, timestamp, resolution)
- Thumbnail strip (3 frames per video)
- Checkbox or tagging interface for marking files as "good" / "bad" / "maybe"

This is what you'll visually scan to make decisions.

### Step 1.4: Tag and filter

After visual review, you mark each video with tags:
- `framing`: close / medium / wide
- `lighting`: day / night / mixed
- `limbs_visible`: yes / partial / no
- `movement_type`: spontaneous / prompted / none / unknown
- `notes`: free text

These get saved to a `video_triage.csv` that feeds Phase 2.

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `01_probe_videos.py` | Run ffprobe on all AVIs, output `video_metadata.csv` |
| `02_extract_thumbnails.py` | Pull 3 frames per AVI via ffmpeg, save to `thumbs/` folder |
| `03_generate_contact_sheet.py` | Build HTML report from metadata + thumbnails |
| `video_triage.csv` | Your manual annotations (template auto-generated) |

### Time estimate:
- ffprobe: ~1-2 seconds per file (even over network)
- Thumbnails: ~3-5 seconds per file (ffmpeg seeks, doesn't read entire file)
- For 28 patients × ~5-10 studies × ~60 AVIs per study ≈ 8,000-17,000 files
- Total: ~6-24 hours unattended (parallelizable)
- Visual review: probably 2-4 hours of your time, with coffee

---

## Phase 2: EEG Synchronization Assessment

**Goal:** For the studies that passed video triage, run your natus-synchronization pipeline to determine how well the video and EEG clocks are aligned.

### What I need to confirm with you:

1. **What does your natus-sync pipeline take as input?** Study folder path? Specific file types?
2. **What does it output?** A sync quality metric? Offset in ms? Drift rate?
3. **Is it a Python script, MATLAB, or something else?**
4. **How long does it take per study?**
5. **Does it need the full AVI, or just the VTC/timestamp files?**

### Step 2.1: Filter to candidate studies

From Phase 1, extract the list of studies that have at least one "good" video file (close framing, daytime, limbs visible).

### Step 2.2: Run natus-sync pipeline

For each candidate study:
- Run synchronization pipeline
- Capture output metrics (offset, drift, quality score)
- Log any failures or warnings

### Step 2.3: Assess sync quality

Determine thresholds for "acceptable" synchronization:
- What offset is tolerable? (e.g., <50ms for movement-EEG correlation?)
- Is drift consistent or does it accumulate?
- Are there studies where sync fails entirely?

Output: `sync_quality.csv` with per-study metrics.

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `04_filter_candidates.py` | Read `video_triage.csv`, output list of studies to process |
| `05_run_sync_pipeline.py` | Batch wrapper around your natus-sync pipeline |
| `sync_quality.csv` | Output sync metrics per study |

---

## Phase 3: Final File Selection

**Goal:** Cross-reference video quality (Phase 1) with sync quality (Phase 2) to produce a final manifest of AVI + EEG file pairs for analysis.

### Step 3.1: Join video triage with sync results

Merge `video_triage.csv` (video quality tags) with `sync_quality.csv` (sync metrics) on study ID.

### Step 3.2: Apply selection criteria

A file makes the cut if ALL of:
- Framing: `close` or `medium` (patient fills >50% of frame)
- Lighting: `day`
- Limbs visible: `yes` or `partial`
- Sync offset: below threshold (TBD based on your pipeline output)
- Sync drift: below threshold
- Duration: sufficient length (at least 60s? 90s?)

### Step 3.3: Generate final manifest

Output: `analysis_manifest.csv` with columns:
- Patient ID
- Study folder
- AVI file path
- EEG file path
- VTC file path
- M1 electrode names (from the CSV you provided)
- Sync offset (ms)
- Video quality tags
- Total usable duration

### Step 3.4: Copy selected files locally (finally!)

NOW you copy — but only the files that passed both filters. This is a fraction of the total data.

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `06_build_manifest.py` | Join video + sync data, apply filters, output final manifest |
| `07_copy_selected.py` | Copy only selected files to local working directory |

---

## Phase 4: Movement Feature Extraction

**Goal:** Extract quantitative movement features from the selected video files.

### What I need to confirm with you:

1. **What movements are you interested in?**
   - Gross limb movements (arm raise, leg kick)?
   - Fine motor (finger wiggling, hand opening/closing)?
   - Facial movements?
   - Spontaneous vs. task-directed?
2. **Pose estimation tool preference?**
   - **MediaPipe** — fast, runs on CPU, good for upper body/hands, easy setup
   - **OpenPose** — gold standard for research, GPU recommended
   - **DeepLabCut** — if you need custom keypoints (e.g., specific finger joints)
   - **MMPose** — modern, flexible, well-documented
3. **What movement features to extract?**
   - Joint positions over time (x, y trajectories)
   - Velocity / acceleration of specific joints
   - Movement onset/offset detection
   - Movement amplitude
   - Laterality (left vs. right limb)

### Step 4.1: Run pose estimation

For each selected AVI:
- Run pose estimation frame-by-frame
- Extract keypoint positions (e.g., wrists, elbows, shoulders, knees, ankles)
- Output time series of joint coordinates

### Step 4.2: Compute movement features

From raw keypoints, derive:
- **Velocity**: frame-to-frame displacement of each joint
- **Movement episodes**: threshold velocity to detect movement onset/offset
- **Amplitude**: peak displacement during each movement episode
- **Movement rate**: movements per minute
- **Asymmetry index**: L vs. R side movement comparison

### Step 4.3: Align to EEG timeline

Using the sync offsets from Phase 2:
- Convert video frame timestamps to EEG sample indices
- Create aligned time series: movement features ↔ EEG signals

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `08_run_pose_estimation.py` | Batch pose estimation on selected AVIs |
| `09_compute_movement_features.py` | Derive velocity, amplitude, episodes from keypoints |
| `10_align_timelines.py` | Map video frames to EEG samples using sync offsets |

---

## Phase 5: EEG Feature Extraction

**Goal:** Extract M1-relevant neural features from the sEEG data, aligned to the same timeline as movement features.

### Step 5.1: Load and preprocess EEG

For each selected study:
- Load EEG data (what format? .eeg? .edf? .nrd?)
- Extract only the M1 electrode channels (from your CSV)
- Apply standard preprocessing:
  - Notch filter (60 Hz + harmonics)
  - Bandpass filter as needed
  - Artifact rejection (optional, depends on data quality)

### Step 5.2: Extract spectral features

For each M1 electrode, compute time-frequency features:
- **High-gamma power** (70–150 Hz) — your primary band for motor activity, consistent with the seegnificant paper approach
- **Beta power** (13–30 Hz) — motor planning and suppression (beta desynchronization)
- **Alpha/mu power** (8–13 Hz) — sensorimotor rhythm
- **Broadband power** — overall neural activity

Compute using:
- Short-time Fourier transform (STFT) or Morlet wavelets
- Window size to match video frame rate (~33ms windows for 30fps alignment)

### Step 5.3: Compute derived features

- **Beta desynchronization events**: drops in beta power preceding movement
- **High-gamma bursts**: peaks in high-gamma coinciding with movement
- **Event-related spectral perturbation (ERSP)**: spectral changes locked to movement onsets

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `11_extract_eeg.py` | Load EEG, select M1 channels, preprocess |
| `12_compute_spectral_features.py` | Time-frequency decomposition on M1 channels |

---

## Phase 6: Correlation Analysis

**Goal:** Quantify the relationship between video-derived movement features and M1 neural activity.

### Step 6.1: Time-locked analysis

For each detected movement episode:
- Extract EEG features in a window around movement onset (e.g., -1s to +1s)
- Compute trial-averaged spectrograms (similar to the seegnificant paper's approach)
- Look for beta desynchronization before movement, high-gamma during movement

### Step 6.2: Continuous correlation

- Cross-correlation between movement velocity time series and high-gamma power time series
- Identify optimal lag (does neural activity lead movement, and by how much?)
- Granger causality or similar directional analysis

### Step 6.3: Across-electrode analysis

- Which M1 electrodes show strongest movement correlation?
- Somatotopic mapping: do hand-area electrodes correlate with arm movements, leg-area with leg movements?
- Comparison across patients

### Step 6.4: Visualization and reporting

- Per-patient summary figures
- Group-level statistics
- Electrode-movement correlation matrices

### Scripts needed:

| Script | Purpose |
|--------|---------|
| `13_movement_locked_analysis.py` | Trial-averaged EEG around movement onsets |
| `14_continuous_correlation.py` | Cross-correlation, lag analysis |
| `15_group_analysis.py` | Across-patient statistics and figures |

---

## Complete Script Inventory

| # | Script | Phase | Input | Output |
|---|--------|-------|-------|--------|
| 00 | `00_build_config.py` | 0 | electrode CSV + data root | `config.yaml` |
| 01 | `01_probe_videos.py` | 1 | config | `video_metadata.csv` |
| 02 | `02_extract_thumbnails.py` | 1 | config + metadata | `thumbs/*.jpg` |
| 03 | `03_generate_contact_sheet.py` | 1 | metadata + thumbs | `contact_sheet.html` |
| 04 | `04_filter_candidates.py` | 2 | `video_triage.csv` | candidate study list |
| 05 | `05_run_sync_pipeline.py` | 2 | candidate list + config | `sync_quality.csv` |
| 06 | `06_build_manifest.py` | 3 | triage + sync CSVs | `analysis_manifest.csv` |
| 07 | `07_copy_selected.py` | 3 | manifest | local file copies |
| 08 | `08_run_pose_estimation.py` | 4 | selected AVIs | keypoint time series |
| 09 | `09_compute_movement_features.py` | 4 | keypoints | movement features |
| 10 | `10_align_timelines.py` | 4 | movement + sync offsets | aligned time series |
| 11 | `11_extract_eeg.py` | 5 | EEG files + M1 channels | preprocessed EEG |
| 12 | `12_compute_spectral_features.py` | 5 | preprocessed EEG | spectral features |
| 13 | `13_movement_locked_analysis.py` | 6 | aligned movement + EEG | trial-averaged results |
| 14 | `14_continuous_correlation.py` | 6 | aligned time series | correlation metrics |
| 15 | `15_group_analysis.py` | 6 | all patient results | figures + stats |

---

## Decision Points (Need Your Input)

Before I start writing scripts, we should resolve these:

### Data access (Phase 0)
- [ ] Data root path and directory structure
- [ ] File naming conventions
- [ ] Network access method (mount vs. SSH vs. other)
- [ ] EEG file format (.eeg, .edf, .nrd?)

### Natus sync pipeline (Phase 2)
- [ ] What is the pipeline? (Python script, MATLAB?)
- [ ] Input/output format
- [ ] What sync quality metric does it produce?
- [ ] What threshold defines "good" sync?

### Movement analysis (Phase 4)
- [ ] Which body parts / movement types are you targeting?
- [ ] Pose estimation tool preference
- [ ] Movement detection thresholds (can calibrate later)
- [ ] Spontaneous movement only, or also prompted/task movements?

### EEG analysis (Phase 5)
- [ ] Frequency bands of primary interest (high-gamma + beta?)
- [ ] Preprocessing preferences (re-referencing scheme? bipolar vs. common avg?)
- [ ] Time-frequency method preference (wavelets vs. STFT?)

### Analysis design (Phase 6)
- [ ] Primary hypothesis: beta desync before movement? high-gamma during?
- [ ] Statistical approach for group-level analysis
- [ ] Are you planning a publication or is this exploratory?

---

## Suggested Order of Work

1. **Now:** Answer the decision points above so I can tailor the scripts
2. **Day 1:** Build `config.yaml` and run `01_probe_videos.py` + `02_extract_thumbnails.py` (can run overnight on the network)
3. **Day 2:** Generate contact sheets, do visual triage over coffee
4. **Day 3:** Run sync pipeline on candidates
5. **Day 4:** Build final manifest, copy selected files
6. **Week 2:** Movement extraction + EEG processing (these can run in parallel)
7. **Week 3:** Correlation analysis

The bottleneck is Phase 1 (your visual review) and Phase 2 (sync processing time). Everything else can be automated.
