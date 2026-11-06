"""Explicit end-to-end check against the running console and a real D435i.

Temporarily changes runtime settings and restarts capture; restores settings in finally.
Run only when a real camera is connected: python tests/hardware_check.py
"""
import asyncio
import io
import json
import struct
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
import sys

import cv2
import numpy as np
from websockets.asyncio.client import connect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision.geometry import depth_grayscale

BASE = 'http://127.0.0.1:8000'


def api(path, method='GET', payload=None):
    request = urllib.request.Request(BASE + path, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json', 'X-Console-Request': '1'})
    with urllib.request.urlopen(request, timeout=10) as response:
        content = response.read()
        return json.loads(content) if 'application/json' in response.headers.get('Content-Type', '') else content


def wait_ready():
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        state = api('/api/status')
        if state['camera_status'] == 'streaming' and state['model_status'] == 'ready' and state['frame_age_ms'] is not None and state['frame_age_ms'] < 1000:
            return state
        time.sleep(.3)
    raise AssertionError(f'Camera/model not ready: {state}')


def unpack_packet(data):
    size = struct.unpack('<I', data[:4])[0]
    metadata = json.loads(data[4:4 + size])
    start = 4 + size
    assert len(data) == start + metadata['rgb_bytes'] + metadata['depth_bytes']
    rgb = cv2.imdecode(np.frombuffer(data[start:start + metadata['rgb_bytes']], np.uint8), cv2.IMREAD_UNCHANGED)
    depth = cv2.imdecode(np.frombuffer(data[start + metadata['rgb_bytes']:], np.uint8), cv2.IMREAD_UNCHANGED)
    assert rgb.shape == (480, 640, 3)
    assert depth.shape == (480, 640) and depth.dtype == np.uint8
    return metadata


async def next_frame(ws, predicate=lambda _: True):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        data = await asyncio.wait_for(ws.recv(), timeout=10)
        if isinstance(data, bytes):
            frame = unpack_packet(data)
            if predicate(frame):
                return frame
    raise AssertionError('No matching synchronized frame received')


async def main():
    original = api('/api/status')['settings']
    try:
        state = await asyncio.to_thread(wait_ready)
        print('REAL CAMERA READY:', state['camera']['name'], 'USB', state['camera']['usb'], flush=True)
        assert b'DepthSight' in api('/')
        async with connect('ws://127.0.0.1:8000/ws', origin=BASE, max_size=4_000_000) as ws:
            frames = [await next_frame(ws) for _ in range(5)]
            assert all(b['sequence'] > a['sequence'] for a, b in zip(frames, frames[1:]))
            frame = frames[-1]
            print('SYNCHRONIZED STREAM:', json.dumps({k: frame[k] for k in ['sequence', 'capture_fps', 'process_fps', 'inference_ms', 'pipeline_ms', 'people_count', 'depth_valid_ratio']}), flush=True)
            api('/api/settings', 'PATCH', {'confidence': .6, 'inference_size': 320, 'near_m': .5, 'far_m': 4.})
            await next_frame(ws, lambda f: f['settings']['inference_size'] == 320 and f['settings']['confidence'] == .6)
            snapshot = api('/api/snapshot')
            with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
                assert set(archive.namelist()) == {'rgb.jpg', 'rgb_annotated.jpg', 'depth.png', 'raw_depth.png', 'metadata.json'}
                raw = cv2.imdecode(np.frombuffer(archive.read('raw_depth.png'), np.uint8), cv2.IMREAD_UNCHANGED)
                gray = cv2.imdecode(np.frombuffer(archive.read('depth.png'), np.uint8), cv2.IMREAD_UNCHANGED)
                meta = json.loads(archive.read('metadata.json'))
                assert raw.dtype == np.uint16 and raw.shape == gray.shape == (480, 640)
                expected = depth_grayscale(raw, meta['camera']['depth_scale'], meta['settings']['near_m'], meta['settings']['far_m'])
                assert np.array_equal(expected, gray), 'Snapshot grayscale differs from raw depth mapping'
            print('SNAPSHOT: all 5 files, raw uint16 and exact grayscale verified', flush=True)
            sample = api('/api/depth?x=320&y=240')
            assert sample['sequence'] > 0
            api('/api/settings', 'PATCH', {'detection_enabled': False})
            disabled = await next_frame(ws, lambda f: not f['settings']['detection_enabled'])
            assert disabled['detections'] == [] and disabled['inference_ms'] is None
            print('DETECTION TOGGLE / PIXEL DEPTH: verified', flush=True)
            stopped = api('/api/camera/stop', 'POST')
            assert stopped['camera_status'] == 'stopped' and stopped['people_count'] == 0
            try:
                api('/api/snapshot')
                raise AssertionError('Stopped capture exposed a stale snapshot')
            except urllib.error.HTTPError as error:
                assert error.code == 409
            api('/api/settings', 'PATCH', {'detection_enabled': True})
            api('/api/camera/start', 'POST')
            resumed = await next_frame(ws, lambda f: f['generation'] > frame['generation'] and f['camera_status'] == 'streaming')
            assert resumed['sequence'] > frame['sequence']
            print('CAMERA STOP / START: verified; old snapshot cleared, new generation received', flush=True)
        print('HARDWARE CHECK PASSED', flush=True)
    finally:
        api('/api/settings', 'PATCH', original)
        api('/api/camera/start', 'POST')


if __name__ == '__main__':
    asyncio.run(main())
