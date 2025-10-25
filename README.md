# Hackathon challenge: Online Video Stabilization

## Overview
This task is to build an online (preferably real-time) video stabilization method for UAV footage. We provide framed videos from three flight missions and corresponding sensor logs. The focus is on methods that do not rely on future frames at inference time.

## Quick end-to-end workflow

Below is a compact, recommended flow to go from raw data to evaluating and visualizing stabilized videos. Replace `baseline` with `fusion` (or your model name) where appropriate.

1) Split the data (creates `data/data_split.json` and optional per-flight CSVs)

```bash
# temporal split (default): 80% train, 10% val, 10% test
python3 src/split_data.py --strategy temporal --ratios 0.8 0.1 0.1 --output data/data_split.json
```

Output: `data/data_split.json` (and `data/labels_split/{train,val,test}/` if `--split-csv` used).

2) Run inference (generate stabilized frames under `output/<model>/<Flight>/`)

```bash
# Run the baseline on the test split
python3 src/inference.py --split data/data_split.json --split-set test --model baseline
```

Notes:
- `inference.py` writes stabilized frames as JPG under `output/<model>/<Flight>/` and saves per-flight transform JSONs there.
- Use `--model fusion` to run the fusion model (fusion will enable sensor data automatically).

3) Evaluate results (produce `evaluation_results.json`)

```bash
# Evaluate baseline (compares original -> stabilized), writes to output/baseline/evaluation_results.json
python3 src/evaluate.py --model baseline --split data/data_split.json --split-set test
```

Notes:
- For `--model original` the script computes only original-video metrics and saves `data/original_metrics.json` by default.
- Evaluation outputs for models are saved to `output/<model>/evaluation_results.json`.

4) Create viewable MP4 videos from frames (optional but recommended for quick inspection)

```bash
# Create MP4 for a single flight from the original dataset
python3 src/create_video_from_images.py --model original Flight1 --split test

# Create MP4 from a model's output (reads from output/<model>/<Flight>/)
python3 src/create_video_from_images.py --model baseline Flight1 --split test
```

Output: `output/videos/<model>/<Flight>.mp4` (now the canonical place the Gradio app looks for videos).

5) Spin up the Gradio app to visually compare original vs stabilized videos

```bash
# Start the app on the default port
python3 app.py

# Start on a custom port and create a public share link
python3 app.py --port 7860 --share
```

UI tips:
- Use the model selector to pick `baseline`, `fusion`, or `Raw` (original).
- Toggle "Show videos" to display pre-rendered MP4s from `output/videos/<model>/<Flight>.mp4` when available. The app falls back to per-frame image streaming when videos are missing.

Troubleshooting
- If `create_video_from_images.py` creates files with `*_fusion.mp4` names from earlier runs, you can recreate canonical videos by re-running the command above, or create symlinks so the app finds them:

```bash
mkdir -p output/videos/fusion
ln -s ../../fusion/Flight1/Flight1_fusion.mp4 output/videos/fusion/Flight1.mp4  # example
```

If you want, I can add a small helper to auto-migrate existing `*_fusion.mp4` files to the new canonical `output/videos/<model>/<flight>.mp4` names.

