# Real-Time Waste Detection System  v4.0

Advanced AI-powered environmental monitoring application built on YOLOv8
and OpenCV, with a professional Tkinter dashboard and persistent SQLite
storage.

---

## Project Structure

```
WasteDetectionSystem/
  main.py                    Entry point
  requirements.txt           Python dependencies
  README.md                  This file

  core/
    __init__.py
    database.py              SQLite persistence layer (thread-safe)
    detector.py              YOLOv8 + contour fallback, enriched detections
    renderer.py              OpenCV HUD / bounding-box overlay
    science.py               In-memory scientific waste catalog (17 categories)

  ui/
    __init__.py
    app.py                   Main application window (camera + dashboard)
    report_window.py         Session report popup + save-to-file
    history_window.py        Sortable colour-coded detection history

  utils/
    __init__.py
    logger.py                Console + file logger

  data/                      Created at runtime
    waste_detection.db       SQLite database
    system.log               Application log

  assets/                    Placeholder for icons / sounds
  exports/                   Placeholder for CSV / report exports
```

---

## Setup

```bash
pip install -r requirements.txt
python main.py
```

YOLOv8 (`yolov8n.pt`) is downloaded automatically on the first run
by the `ultralytics` library (~6 MB).

---

## Features

### Detection Engine
- YOLOv8n (nano, fast) for primary inference at 416px input size
- OpenCV contour + shape heuristics as zero-dependency fallback
- Detection throttling (80 ms gate) to sustain UI frame rate
- Per-detection enrichment: bounding box geometry, aspect ratio,
  relative frame coverage, estimated distance (pinhole model)

### Environmental Science Panel
- 17 waste material categories with full scientific metadata:
  biodegradability, toxicity, recycling score, decomposition time,
  carbon footprint, UN hazard class, resin code
- Health risk and environmental impact summaries
- Disposal method recommendations

### Live Charts (Chart tab)
- Toxicity timeline  (60-point rolling window)
- Radiation timeline (normalised, 60-point rolling)
- FPS performance chart
- Category distribution bar chart (colour-coded)

### Dashboard
- Chip metrics: FPS, total detections, hazardous, recyclable
- Status bar: session ID, FPS, radiation level, time
- Mode indicator: LIVE / SIMULATION

### Data Persistence
- All detections logged to SQLite with session ID, bounding box,
  frame size, and area in pixels
- Session stats: start/end time, total frames, avg FPS
- Export to CSV via in-app button

### Report
- Full structured session report (text)
- Saveable to .txt via file dialog
- Includes: summary, category distribution, detection log (last 30),
  hazardous item list, and actionable recommendations

---

## Performance Notes

- Detection throttle: inference runs at most every 80 ms (~12.5/s)
  while the UI refreshes at ~30 fps independently.
- YOLO warm-up: a blank frame is inferred at startup to pre-compile
  the model and avoid the first-frame delay.
- Camera buffer size is set to 1 to minimise latency.
- Charts are redrawn at most every 800 ms to avoid canvas thrashing.

---

## Extending

Add new waste categories by appending entries to `core/science.py`
`WASTE_CATALOG` dict and mapping YOLO class names in `YOLO_TO_WASTE`.
The database seed in `core/database.py` can be extended similarly.
