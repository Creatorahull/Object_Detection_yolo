"""
detect_and_track.py
====================
Real-time object detection + multi-object tracking.

Pipeline:
    1. Video input (webcam or file) via OpenCV.
    2. Per-frame object detection via a pre-trained YOLOv8 model (ultralytics).
    3. Detections are handed to a SORT tracker (Kalman filter + Hungarian
       matching, see sort.py) which assigns a stable ID to each object
       across frames.
    4. Bounding boxes, class labels, confidence, and track IDs are drawn
       and displayed in real time. Optionally the annotated stream is
       saved to a video file.

Usage:
    # Webcam (default camera index 0)
    python detect_and_track.py --source 0

    # Video file
    python detect_and_track.py --source path/to/video.mp4

    # Save annotated output
    python detect_and_track.py --source 0 --save output.mp4

    # Only track specific classes (COCO class ids), e.g. person=0, car=2
    python detect_and_track.py --source 0 --classes 0 2

Press 'q' to quit the display window.
"""

import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO

from sort import Sort


# A distinct BGR color per track ID (cycled), so each tracked object keeps
# a visually stable color across frames.
def color_for_id(track_id: int):
    np.random.seed(int(track_id) * 37 + 1)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def parse_args():
    p = argparse.ArgumentParser(description="Real-time YOLO detection + SORT tracking")
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0) or path to a video file")
    p.add_argument("--model", default="yolov8n.pt",
                   help="Ultralytics YOLO weights (yolov8n/s/m/l/x.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    p.add_argument("--iou-track", type=float, default=0.3, help="SORT IoU match threshold")
    p.add_argument("--max-age", type=int, default=30,
                   help="Frames a track survives without a matching detection")
    p.add_argument("--min-hits", type=int, default=3,
                   help="Consecutive matches required before a track is displayed")
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="Restrict detection to these COCO class ids (default: all)")
    p.add_argument("--save", default=None, help="Optional path to save annotated video")
    p.add_argument("--no-display", action="store_true",
                   help="Don't open a display window (useful for headless runs)")
    return p.parse_args()


def open_source(source: str):
    # Numeric strings are treated as webcam indices, everything else as a file/path.
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()

    model = YOLO(args.model)
    names = model.names  # class id -> class name

    cap = open_source(args.source)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (w, h))

    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_track)

    prev_time = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # --- Detection ---
        results = model.predict(
            frame, conf=args.conf, classes=args.classes, verbose=False
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = results.boxes.cls.cpu().numpy().reshape(-1, 1)
            detections = np.hstack([xyxy, conf, cls])
        else:
            detections = np.empty((0, 6))

        # --- Tracking ---
        tracks = tracker.update(detections)

        # --- Draw ---
        for x1, y1, x2, y2, track_id, cls_id in tracks:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = color_for_id(track_id)
            label = f"{names.get(int(cls_id), str(int(cls_id)))} ID:{int(track_id)}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # --- FPS overlay ---
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}  Tracks: {len(tracks)}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if not args.no_display:
            cv2.imshow("Object Detection & Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
