from __future__ import annotations

import io
import json
import logging
import os
import struct
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .camera import RealSenseCamera
from .geometry import PersonTracker, depth_grayscale, person_depth
from .notifications import FeishuNotifier

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    detection_enabled: bool = True
    confidence: float = .45
    near_m: float = .3
    far_m: float = 6.
    inference_size: int = 640


class VisionEngine:
    def __init__(self):
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.shutdown = threading.Event()
        self.active = True
        self.generation = 0
        self.settings = Settings()
        self.latest = None
        self.packet = None
        self.snapshot = None
        self.seq = 0
        self.output_seq = 0
        self.model_retry = False
        self.started = time.monotonic()
        self.state = {"camera_status": "connecting", "camera_error": None, "camera": None,
                      "model_status": "loading", "model_error": None, "model_name": "YOLO11n",
                      "device": "CPU", "capture_fps": 0., "process_fps": 0.,
                      "inference_ms": None, "pipeline_ms": None, "people_count": 0,
                      "depth_valid_ratio": 0., "frame_age_ms": None, "detections": []}
        self.threads = []
        self.notifications = FeishuNotifier(ROOT / 'config.local.json')

    def start(self):
        self.notifications.start()
        for name, fn in [("realsense-capture", self._capture), ("person-inference", self._process)]:
            thread = threading.Thread(target=fn, name=name, daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.shutdown.set()
        with self.changed:
            self.changed.notify_all()
        for thread in self.threads:
            thread.join(timeout=8)
        self.notifications.close()

    def status(self):
        with self.lock:
            state = dict(self.state)
            state.update(settings=asdict(self.settings), active=self.active, generation=self.generation,
                         uptime_s=round(time.monotonic() - self.started), sequence=self.output_seq)
            state["frame_age_ms"] = round((time.monotonic() - self.snapshot["captured"]) * 1000) if self.snapshot else None
            state['notifications'] = self.notifications.status()
            return state

    def configure(self, values):
        with self.lock:
            proposed = replace(self.settings, **values)
            if proposed.far_m - proposed.near_m < .2 - 1e-9:
                raise ValueError("最远距离至少应比最近距离大 0.2 米")
            self.settings = proposed
            if 'detection_enabled' in values:
                self.notifications.reset_presence()
            return asdict(proposed)

    def set_active(self, enabled):
        with self.changed:
            if enabled != self.active:
                self.generation += 1
                self.active = enabled
                self._clear_frame()
                self.state.update(camera_status="connecting" if enabled else "stopped", camera_error=None)
                self.changed.notify_all()
        return self.status()

    def _clear_frame(self):
        self.notifications.reset_presence()
        self.latest = self.packet = self.snapshot = None
        self.state.update(people_count=0, detections=[], capture_fps=0., process_fps=0.,
                          inference_ms=None, pipeline_ms=None, depth_valid_ratio=0.)

    def get_packet(self):
        with self.lock:
            return self.output_seq, self.packet

    def _capture(self):
        while not self.shutdown.is_set():
            with self.changed:
                self.changed.wait_for(lambda: self.active or self.shutdown.is_set(), timeout=1)
                if not self.active or self.shutdown.is_set():
                    continue
                generation = self.generation
            camera = RealSenseCamera()
            try:
                info = camera.open()
                with self.lock:
                    if generation == self.generation:
                        self.state.update(camera=info, camera_error=None)
                count, window = 0, time.monotonic()
                while not self.shutdown.is_set():
                    with self.lock:
                        if not self.active or generation != self.generation:
                            break
                    result = camera.read()
                    if result is None:
                        continue
                    rgb, depth, timing = result
                    captured = time.monotonic()
                    count += 1
                    elapsed = captured - window
                    with self.changed:
                        if generation != self.generation or not self.active:
                            break
                        self.seq += 1
                        self.latest = (self.seq, generation, rgb, depth, camera.scale, captured, timing)
                        self.state.update(camera_status="streaming", camera_error=None)
                        if elapsed >= 1:
                            self.state["capture_fps"] = round(count / elapsed, 1)
                            count, window = 0, captured
                        self.changed.notify_all()
            except Exception as exc:
                log.exception("RealSense acquisition failed")
                with self.lock:
                    if generation == self.generation and self.active:
                        self.generation += 1
                        self._clear_frame()
                        self.state.update(camera_status="error", camera_error=str(exc))
            finally:
                camera.close()
            self.shutdown.wait(.3 if not self.active else 2.)

    def _load_model(self):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        path = ROOT / "models" / "yolo11n.pt"
        if not path.is_file():
            raise FileNotFoundError("缺少 models/yolo11n.pt，请运行 python setup_model.py")
        device = "0" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(path))
        model.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, device=device, verbose=False, classes=[0])
        with self.lock:
            self.state.update(model_status="ready", model_error=None,
                              device=torch.cuda.get_device_name(0) if device == "0" else "CPU")
        return model, device

    def _process(self):
        model, device, tried = None, "cpu", False
        last_seq = -1
        tracker = PersonTracker()
        track_generation = -1
        count, window = 0, time.monotonic()
        while not self.shutdown.is_set():
            with self.changed:
                self.changed.wait_for(lambda: self.shutdown.is_set() or
                                      (self.latest is not None and self.latest[0] != last_seq), timeout=.5)
                if self.shutdown.is_set():
                    break
                sample = self.latest
                if sample is None or sample[0] == last_seq:
                    continue
                config = self.settings
                retry = self.model_retry
                if config.detection_enabled:
                    self.model_retry = False
            if config.detection_enabled and (not tried or retry):
                self.notifications.reset_presence()
                tried = True
                with self.lock:
                    self.state.update(model_status="loading", model_error=None)
                try:
                    model, device = self._load_model()
                except Exception as exc:
                    model = None
                    log.exception("Detector initialization failed")
                    with self.lock:
                        self.state.update(model_status="error", model_error=str(exc))
                # Model initialization may take seconds: take the newest frame afterward.
                with self.lock:
                    sample = self.latest
                if sample is None:
                    continue
            seq, generation, rgb, depth, scale, captured, timing = sample
            last_seq = seq
            if generation != track_generation:
                tracker = PersonTracker()
                track_generation = generation
            try:
                detections, inference_ms = [], None
                if config.detection_enabled and model is not None:
                    begin = time.perf_counter()
                    prediction = model.predict(rgb, classes=[0], conf=config.confidence,
                                               imgsz=config.inference_size, device=device,
                                               max_det=50, verbose=False)[0]
                    inference_ms = round((time.perf_counter() - begin) * 1000, 1)
                    for row in prediction.boxes.data.cpu().tolist():
                        x1, y1, x2, y2, confidence, _ = row
                        box = [max(0., x1), max(0., y1), min(float(rgb.shape[1]), x2), min(float(rgb.shape[0]), y2)]
                        detections.append({"box": [round(v, 1) for v in box], "confidence": float(confidence),
                                           **person_depth(depth, scale, box)})
                    tracker.update(detections, captured)
                gray = depth_grayscale(depth, scale, config.near_m, config.far_m)
                ok_rgb, jpg = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
                ok_depth, png = cv2.imencode(".png", gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok_rgb or not ok_depth:
                    raise RuntimeError("图像编码失败")
                count += 1
                now = time.monotonic()
                elapsed = now - window
                with self.lock:
                    if generation != self.generation or not self.active or self.latest is None:
                        continue
                    if elapsed >= 1:
                        self.state["process_fps"] = round(count / elapsed, 1)
                        count, window = 0, now
                    self.state.update(inference_ms=inference_ms, people_count=len(detections),
                                      detections=detections, pipeline_ms=round((now - captured) * 1000, 1),
                                      depth_valid_ratio=round(float(np.mean(depth > 0)), 3))
                    metadata = self.status()
                    metadata.update(sequence=seq, width=rgb.shape[1], height=rgb.shape[0],
                                    frame_age_ms=round((now - captured) * 1000),
                                    captured_at=time.time() - (now - captured), rgb_bytes=len(jpg),
                                    depth_bytes=len(png), timing=timing, settings=asdict(config))
                    self.notifications.offer(metadata, observed_at=captured)
                    header = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    self.packet = struct.pack("<I", len(header)) + header + jpg.tobytes() + png.tobytes()
                    self.output_seq = seq
                    self.snapshot = {"rgb": rgb, "depth": depth, "gray": gray,
                                     "metadata": metadata, "captured": captured}
            except Exception as exc:
                log.exception("Frame processing failed")
                with self.lock:
                    self.packet = self.snapshot = None
                    self.state.update(model_status="error", model_error=str(exc), people_count=0, detections=[])
                model, tried = None, True
                self.shutdown.wait(.1)

    def sample_depth(self, x, y):
        with self.lock:
            shot = self.snapshot
            if shot is None or time.monotonic() - shot["captured"] > 3:
                raise ValueError("尚无可用的新鲜深度帧")
            depth = shot["depth"]
            if not 0 <= x < depth.shape[1] or not 0 <= y < depth.shape[0]:
                raise ValueError("像素坐标超出画面")
            raw = int(depth[y, x])
            scale = shot["metadata"]["camera"]["depth_scale"]
            return {"x": x, "y": y, "distance_m": round(raw * scale, 3) if raw else None,
                    "sequence": shot["metadata"]["sequence"]}

    def export_snapshot(self):
        with self.lock:
            shot = self.snapshot
            if shot is None or time.monotonic() - shot["captured"] > 3:
                raise ValueError("尚无可导出的新鲜帧")
        annotated = shot["rgb"].copy()
        for d in shot["metadata"]["detections"]:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (90, 220, 70), 2)
            distance = f'{d["distance_m"]:.2f}m' if d["distance_m"] is not None else "depth N/A"
            cv2.putText(annotated, f'Person #{d["id"]} {d["confidence"]:.0%} {distance}',
                        (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (90, 220, 70), 1, cv2.LINE_AA)
        memory = io.BytesIO()
        metadata = dict(shot["metadata"], raw_depth_units="raw_depth.png value * camera.depth_scale = metres",
                        depth_mapping="invalid=0; valid=1+254*clip((metres-near_m)/(far_m-near_m),0,1)",
                        liveness="Person detection with depth support; NOT certified liveness or anti-spoof detection.")
        with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in [("rgb.jpg", shot["rgb"]), ("rgb_annotated.jpg", annotated),
                               ("depth.png", shot["gray"]), ("raw_depth.png", shot["depth"])]:
                ok, encoded = cv2.imencode(Path(name).suffix, data)
                if not ok:
                    raise RuntimeError("快照编码失败")
                archive.writestr(name, encoded.tobytes())
            archive.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
        return memory.getvalue()
