import unittest
import json
import struct
import threading
import time
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app import app, engine
from vision.geometry import PersonTracker, depth_grayscale, person_depth
from vision.engine import VisionEngine
from vision.notifications import FeishuNotifier


class DepthTests(unittest.TestCase):
    def test_mapping_preserves_order_clips_and_reserves_invalid(self):
        raw = np.array([[0, 100, 300, 1500, 3000, 6000, 9000]], np.uint16)
        gray = depth_grayscale(raw, .001, .3, 6.)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(gray[0, 0], 0)
        self.assertEqual(gray[0, 1], 1)
        self.assertEqual(gray[0, 2], 1)
        self.assertEqual(gray[0, 5], 255)
        self.assertEqual(gray[0, 6], 255)
        self.assertTrue(np.all(np.diff(gray[0].astype(int)) >= 0))

    def test_ranging_ignores_missing_pixels_and_sparse_outliers(self):
        depth = np.full((100, 100), 2000, np.uint16)
        depth[20:25, 30:70] = 0
        depth[30:32, 30:70] = 9000
        result = person_depth(depth, .001, [0, 0, 100, 100])
        self.assertEqual(result['distance_m'], 2.)
        self.assertEqual(result['depth_status'], 'valid')

    def test_missing_depth_never_returns_distance(self):
        result = person_depth(np.zeros((100, 100), np.uint16), .001, [0, 0, 100, 100])
        self.assertIsNone(result['distance_m'])
        self.assertEqual(result['depth_status'], 'insufficient')

    def test_variable_depth_marks_unstable(self):
        depth = np.tile(np.linspace(1000, 6000, 100).astype(np.uint16), (100, 1))
        self.assertEqual(person_depth(depth, .001, [0, 0, 100, 100])['depth_status'], 'unstable')

    def test_tracking_ids_are_unique_and_expire(self):
        tracker = PersonTracker()
        first = tracker.update([{'box': [0, 0, 50, 100]}, {'box': [60, 0, 110, 100]}], 1.)
        following = tracker.update([{'box': [2, 0, 52, 100]}, {'box': [62, 0, 112, 100]}], 1.1)
        self.assertEqual([d['id'] for d in first], [d['id'] for d in following])
        after_expiry = tracker.update([{'box': [2, 0, 52, 100]}], 3.)
        self.assertNotEqual(first[0]['id'], after_expiry[0]['id'])


class APITests(unittest.TestCase):
    def setUp(self):
        # Do not enter lifespan here: these checks intentionally do not open hardware.
        self.client = TestClient(app)
        self.headers = {'X-Console-Request': '1'}
        self.original = engine.settings

    def tearDown(self):
        engine.settings = self.original
        self.client.close()

    def test_page_and_static_assets(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)
        self.assertEqual(self.client.get('/static/style.css').status_code, 200)

    def test_rejects_invalid_ranges_sizes_and_nulls(self):
        for payload in [{'near_m': 6., 'far_m': 1.}, {'inference_size': 641},
                        {'confidence': None}, {'confidence': .99}, {'surprise': True}]:
            self.assertEqual(self.client.patch('/api/settings', json=payload, headers=self.headers).status_code, 422)
        response = self.client.patch('/api/settings', json={'near_m': .1, 'far_m': .3}, headers=self.headers)
        self.assertEqual(response.status_code, 200)

    def test_rejects_foreign_origin_mutations(self):
        self.assertEqual(self.client.patch('/api/settings', json={'confidence': .5}).status_code, 403)
        self.assertEqual(self.client.patch('/api/settings', json={'confidence': .5},
                                          headers={**self.headers, 'Origin': 'https://example.com'}).status_code, 403)

    def test_missing_snapshot_is_conflict(self):
        self.assertEqual(self.client.get('/api/snapshot').status_code, 409)

    def test_notification_address_can_be_saved_without_echoing_credentials(self):
        notifier = FeishuNotifier(config={'webhook': ''})
        address = 'https://open.feishu.cn/open-apis/bot/v2/hook/api-test-only'
        with patch.object(engine, 'notifications', notifier):
            response = self.client.patch('/api/notifications', json={'webhook': address}, headers=self.headers)
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()['configured'])
            self.assertNotIn(address, response.text)
            self.assertEqual(notifier.webhook, address)
            invalid = self.client.patch('/api/notifications', json={'webhook': 'https://example.com/secret'}, headers=self.headers)
            self.assertEqual(invalid.status_code, 422)
            self.assertEqual(notifier.webhook, address)
            self.assertEqual(self.client.patch('/api/notifications', json={'webhook': None}, headers=self.headers).status_code, 422)


class EngineFailureTests(unittest.TestCase):
    def test_model_failure_keeps_rgb_and_depth_streaming(self):
        console = VisionEngine()
        captured = time.monotonic()
        console.state.update(camera_status='streaming', camera={'depth_scale': .001})
        console.latest = (1, 0, np.zeros((480, 640, 3), np.uint8),
                          np.full((480, 640), 1800, np.uint16), .001, captured, {})
        with patch.object(console, '_load_model', side_effect=RuntimeError('test model failure')):
            worker = threading.Thread(target=console._process)
            worker.start()
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and console.get_packet()[1] is None:
                    time.sleep(.01)
                _, packet = console.get_packet()
                self.assertIsNotNone(packet, 'RGB/depth output must survive model load failure')
                length = struct.unpack('<I', packet[:4])[0]
                metadata = json.loads(packet[4:4 + length])
                self.assertEqual(metadata['model_status'], 'error')
                self.assertEqual(metadata['detections'], [])
                self.assertGreater(metadata['rgb_bytes'], 0)
                self.assertGreater(metadata['depth_bytes'], 0)
            finally:
                console.shutdown.set()
                with console.changed:
                    console.changed.notify_all()
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
