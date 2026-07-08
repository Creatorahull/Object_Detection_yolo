"""
sort.py — A minimal, dependency-light implementation of SORT
(Simple Online and Realtime Tracking).

Reference: Bewley et al., "Simple Online and Realtime Tracking", 2016.

Each tracked object is modeled with a constant-velocity Kalman filter over
the state [cx, cy, s, r, vx, vy, vs], where:
    cx, cy = center of the bounding box
    s      = scale (area)
    r      = aspect ratio (assumed constant)
    vx,vy,vs = velocities of the above

Frame-to-frame association between existing tracks and new detections is
done via IoU (Intersection-over-Union) cost + the Hungarian algorithm
(scipy.optimize.linear_sum_assignment).
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou_batch(bb_test, bb_gt):
    """Vectorized IoU between two sets of boxes in [x1,y1,x2,y2] format."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

    union = area_test + area_gt - inter
    return inter / np.maximum(union, 1e-6)


def bbox_to_z(bbox):
    """[x1,y1,x2,y2] -> [cx,cy,s,r] (state observation vector)."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([cx, cy, s, r]).reshape((4, 1))


def x_to_bbox(x):
    """[cx,cy,s,r,...] -> [x1,y1,x2,y2]"""
    w = np.sqrt(max(x[2] * x[3], 0))
    h = x[2] / (w + 1e-6)
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))


class KalmanBoxTracker:
    """Represents the internal state of a single tracked object."""

    count = 0

    def __init__(self, bbox, cls_id=None, score=None):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State transition (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on unobservable velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = bbox_to_z(bbox)

        self.time_since_update = 0
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.cls_id = cls_id
        self.score = score

    def update(self, bbox, cls_id=None, score=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox_to_z(bbox))
        if cls_id is not None:
            self.cls_id = cls_id
        if score is not None:
            self.score = score

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(x_to_bbox(self.kf.x.flatten()))
        return self.history[-1]

    def get_state(self):
        return x_to_bbox(self.kf.x.flatten())


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """Returns matches, unmatched_detections, unmatched_trackers."""
    if len(trackers) == 0:
        return (np.empty((0, 2), dtype=int),
                np.arange(len(detections)),
                np.empty((0,), dtype=int))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
    unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort:
    """
    Multi-object tracker.

    Usage:
        tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        tracks = tracker.update(detections)   # detections: Nx6 [x1,y1,x2,y2,score,cls]
        # tracks: Mx6 [x1,y1,x2,y2,track_id,cls]
    """

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections=np.empty((0, 6))):
        """
        detections: array of [x1, y1, x2, y2, score, cls_id], can be empty.
        Must be called once per frame, even with empty detections.
        """
        self.frame_count += 1

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        dets_boxes = detections[:, :4] if len(detections) else np.empty((0, 4))
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets_boxes, trks[:, :4] if len(trks) else np.empty((0, 4)), self.iou_threshold)

        for m in matched:
            det_idx, trk_idx = m[0], m[1]
            self.trackers[trk_idx].update(
                detections[det_idx, :4],
                cls_id=detections[det_idx, 5],
                score=detections[det_idx, 4],
            )

        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                detections[i, :4], cls_id=detections[i, 5], score=detections[i, 4]
            )
            self.trackers.append(trk)

        ret = []
        n = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id], [trk.cls_id])).reshape(1, -1))
            n -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.remove(trk)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
