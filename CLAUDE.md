# Movement Feature Extraction Project

## Overview
Research, test, and compare tools for extracting movement features from video. Build visualizations to compare raw video with extracted time-series movement data side-by-side.

## Project Structure
- `research/` — Notes and comparisons of tools
- `scripts/` — Extraction and visualization scripts
- `data/` — Sample videos and extracted features (not committed)
- `visualizations/` — Output figures and videos

## Useful Commands
- **Run extraction:** `python scripts/extract_features.py <video_path>`
- **Generate visualization:** `python scripts/visualize_comparison.py <video_path> <features_path>`

## Notes
- Keep test videos in `data/` but do not commit large files
- Use `.gitignore` for data files and output artifacts
