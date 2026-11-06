"""Depth display and robust person ranging; all distances are metres."""
from __future__ import annotations

import numpy as np


def depth_grayscale(depth: np.ndarray, scale: float, near: float, far: float) -> np.ndarray:
    if not 0 <= near < far:
        raise ValueError("near must be smaller than far")
    metres = depth.astype(np.float32) * scale
    # Reserve zero for missing measurements. Valid near -> 1, far -> 255.
    gray = np.rint(1 + 254 * np.clip((metres - near) / (far - near), 0, 1)).astype(np.uint8)
    gray[depth == 0] = 0
    return gray


def person_depth(depth: np.ndarray, scale: float, box: list[float]) -> dict:
    h, w = depth.shape
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    # Central torso avoids background at the box edge and gaps between legs.
    left, right = max(0, int(x1 + .3 * bw)), min(w, int(x1 + .7 * bw))
    top, bottom = max(0, int(y1 + .2 * bh)), min(h, int(y1 + .65 * bh))
    roi = depth[top:bottom, left:right].astype(np.float32) * scale
    valid = roi[(roi >= .15) & (roi <= 20)]
    ratio = float(valid.size / roi.size) if roi.size else 0.
    result = {"distance_m": None, "depth_valid_ratio": round(ratio, 3),
              "depth_status": "insufficient", "depth_spread_m": None,
              "sample_box": [left, top, right, bottom]}
    if valid.size < 30 or ratio < .35:
        return result
    q25, q75 = np.percentile(valid, [25, 75])
    iqr = float(q75 - q25)
    core = valid[(valid >= q25 - 1.5 * iqr) & (valid <= q75 + 1.5 * iqr)]
    distance = float(np.median(core))
    result.update(distance_m=round(distance, 3), depth_spread_m=round(iqr, 3),
                  depth_status="valid" if iqr < max(.35, distance * .15) else "unstable")
    return result


def iou(a: list, b: list) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - overlap
    return overlap / union if union else 0.


class PersonTracker:
    """Short-lived IoU labels, not identity recognition or persistent identities."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections: list[dict], timestamp: float) -> list[dict]:
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v[1] < 1.}
        candidates = sorted(((iou(d["box"], box), i, tid)
                             for i, d in enumerate(detections)
                             for tid, (box, _) in self.tracks.items()), reverse=True)
        used_d, used_t = set(), set()
        for score, idx, tid in candidates:
            if score < .25:
                break
            if idx not in used_d and tid not in used_t:
                detections[idx]["id"] = tid
                used_d.add(idx)
                used_t.add(tid)
        for d in detections:
            if "id" not in d:
                d["id"] = self.next_id
                self.next_id += 1
            self.tracks[d["id"]] = (d["box"], timestamp)
        return detections
