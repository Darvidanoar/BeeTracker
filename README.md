# BeeTracker

A Raspberry Pi (Camera Module 3) computer-vision tool that watches a hive
entrance and counts bees moving in and out.

## How it works

- Captures live frames from a Picamera2-connected Camera Module 3.
- Uses OpenCV color thresholding + contour detection to find bee-sized blobs
  at the hive entrance, then a lightweight centroid tracker assigns each blob
  a persistent ID across frames.
- Classifies each tracked object's vertical movement (up/down) and detects a
  full crossing when an object travels from the "top" edge zone to the
  "bottom" edge zone (or vice versa), incrementing exit/return counters.
- Every 6 minutes, logs the interval's exit/return delta to a local SQLite
  database (`hive_activity.db`) for later analysis (e.g. a dashboard).
- Stops automatically at a configurable cutoff time (default 7:30 PM) or on
  pressing `q` in the preview window.

## Requirements

- Raspberry Pi with Camera Module 3 and `picamera2` / `libcamera` installed
  (these ship with Raspberry Pi OS and are not pip-installable on other
  platforms).
- Python packages: `opencv-python`, `numpy`

```bash
pip install -r requirements.txt
```

## Usage

```bash
python BeeTraker.py
```

Adjust `DB_PATH`, `INTERVAL_SECONDS`, and `CUTOFF` at the top of the script,
and the HSV color range / camera crop settings in `main()`, to match your
setup.
