# Task 4: Object Detection and Tracking

Real-time pipeline: webcam/video input → YOLOv8 detection → SORT tracking →
annotated display with labels and track IDs.

## Files
- `detect_and_track.py` — main script (video capture, detection, drawing, display/save loop)
- `sort.py` — standalone SORT tracker (Kalman filter + Hungarian/IoU matching), no external tracking library required
- `requirements.txt` — dependencies

## Setup
```bash
pip install -r requirements.txt
```
The first run auto-downloads `yolov8n.pt` (~6 MB, pre-trained on COCO's 80 classes).

## Run

Webcam:
```bash
python detect_and_track.py --source 0
```

Video file:
```bash
python detect_and_track.py --source path/to/video.mp4
```

Save the annotated output instead of/while displaying it:
```bash
python detect_and_track.py --source 0 --save output.mp4
```

Track only specific COCO classes (e.g. person=0, car=2, dog=16):
```bash
python detect_and_track.py --source 0 --classes 0 2
```

Press `q` in the display window to stop.

## How each requirement is met
1. **Real-time video input (OpenCV)** — `cv2.VideoCapture` opens either a webcam index or a file path in `open_source()`.
2. **Pre-trained detector (YOLO)** — `ultralytics.YOLO('yolov8n.pt')`, run per-frame with `model.predict(...)`. Swap in `yolov8s/m/l/x.pt` for higher accuracy at the cost of speed.
3. **Per-frame processing + bounding boxes** — each frame's detections are converted to `[x1,y1,x2,y2,conf,class]` and drawn with `cv2.rectangle`.
4. **Tracking (SORT)** — `sort.py` implements SORT: each object is a Kalman filter over `[cx,cy,scale,ratio,vx,vy,vscale]`; frame-to-frame data association uses an IoU cost matrix solved via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). Tracks persist through brief misses (`--max-age`) and only appear after a few confirmed hits (`--min-hits`) to suppress flicker from false detections.
5. **Real-time display with labels + IDs** — each box is annotated with `"<class name> ID:<track id>"`, plus an FPS counter overlay.

## Notes / tuning
- Lower `--conf` to catch more (noisier) detections; raise it to reduce false positives.
- `--max-age` controls how many frames a track survives with no matching detection (handles brief occlusion).
- `--min-hits` controls how many consecutive matched frames are needed before a new track is shown (reduces ID flicker on spurious detections).
- For denser scenes or appearance-based re-identification (e.g. after long occlusions or camera cuts), swap SORT for **Deep SORT**, which adds a CNN re-identification embedding on top of the same Kalman/IoU framework — the detection and drawing code in `detect_and_track.py` would stay the same, only the tracker call changes.
- This was tested end-to-end on a synthetic video in a sandboxed (no-webcam) environment; run locally with `--source 0` to use an actual webcam.
