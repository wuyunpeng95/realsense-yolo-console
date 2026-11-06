from __future__ import annotations

import numpy as np


class RealSenseCamera:
    def __init__(self):
        self.pipeline = None

    def open(self):
        import pyrealsense2 as rs
        self.rs = rs
        context = rs.context()
        devices = list(context.query_devices())
        device = next((d for d in devices if "D435" in d.get_info(rs.camera_info.name)), None)
        if device is None:
            raise RuntimeError("未发现 D435/D435i，请检查 USB 连接并关闭占用相机的软件")
        serial = device.get_info(rs.camera_info.serial_number)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline = rs.pipeline(context)
        try:
            profile = self.pipeline.start(config)
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.info = {"name": device.get_info(rs.camera_info.name), "serial": serial,
                         "usb": device.get_info(rs.camera_info.usb_type_descriptor),
                         "width": 640, "height": 480, "fps": 30, "depth_scale": self.scale,
                         "intrinsics": {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
                                        "ppy": intr.ppy, "coeffs": list(intr.coeffs),
                                        "model": str(intr.model)}}
            return self.info
        except Exception:
            self.close()
            raise

    def read(self):
        frames = self.pipeline.wait_for_frames(2000)
        aligned = self.align.process(frames)
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color or not depth:
            return None
        # Copy out of SDK buffers before the next capture; keep paired frames together.
        return (np.asanyarray(color.get_data()).copy(), np.asanyarray(depth.get_data()).copy(),
                {"color_timestamp_ms": color.get_timestamp(),
                 "depth_timestamp_ms": depth.get_timestamp(),
                 "color_frame_number": color.get_frame_number(),
                 "depth_frame_number": depth.get_frame_number()})

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
