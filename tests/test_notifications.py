import json
import queue
import os
import threading
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from vision.notifications import FeishuNotifier, build_alert

HOOK = 'https://open.feishu.cn/open-apis/bot/v2/hook/test-only'


def frame(*confidences):
    return {'captured_at': 0, 'sequence': 42, 'camera': {'name': 'RealSense D435I'},
            'detections': [{'id': i + 1, 'confidence': confidence, 'distance_m': 2.1,
                            'visible_seconds': 6., 'depth_status': 'valid'} for i, confidence in enumerate(confidences)]}


class NotificationTests(unittest.TestCase):
    def make_notifier(self):
        return FeishuNotifier(config={'webhook': HOOK, 'enabled': True, 'cooldown_seconds': 60})

    def wait_for_state(self, notifier, predicate):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            state = notifier.status()
            if not notifier.pending and predicate(state):
                return state
            time.sleep(.01)
        self.fail(f'Notification did not complete: {notifier.status()}')

    def submit_initial_person(self, notifier):
        for i in range(11):
            self.assertFalse(notifier.offer(frame(.95), observed_at=i * .5))
        self.assertTrue(notifier.offer(frame(.95), observed_at=5.5))
        self.wait_for_state(notifier, lambda state: state['sent_count'] == 1)

    def test_strict_threshold_uses_unrounded_confidence(self):
        self.assertIsNone(build_alert(frame(.84, .85)))
        event = build_alert(frame(.85, .85001, .93))
        text = event['payload']['content']['text']
        self.assertIn('2 名', text)
        self.assertIn('93.00%', text)
        self.assertIn('2.10 m', text)
        self.assertIn('1970-01-01 08:00:00', text)
        self.assertNotIn('人员 #1｜', text)
        self.assertIsNone(build_alert(frame(float('nan'))))

    def test_empty_and_low_confidence_do_not_enqueue(self):
        notifier = self.make_notifier()
        self.assertFalse(notifier.offer(frame()))
        self.assertFalse(notifier.offer(frame(.85)))
        self.assertTrue(notifier.tasks.empty())

    def test_cooldown_prevents_repeated_frames_and_allows_later_alert(self):
        notifier = self.make_notifier()
        for i in range(11):
            with patch('vision.notifications.time.monotonic', return_value=100. + i * .5):
                self.assertFalse(notifier.offer(frame(.9)))
        with patch('vision.notifications.time.monotonic', return_value=105.5):
            self.assertTrue(notifier.offer(frame(.9)))
        notifier.tasks.get_nowait()
        notifier.pending = False
        for i in range(1, 120):
            with patch('vision.notifications.time.monotonic', return_value=105.5 + i * .5):
                self.assertFalse(notifier.offer(frame(.95)))
        with patch('vision.notifications.time.monotonic', return_value=165.5):
            self.assertTrue(notifier.offer(frame(.95)))

    def test_exact_five_seconds_does_not_alert_but_more_than_five_does(self):
        notifier = self.make_notifier()
        for i in range(11):
            self.assertFalse(notifier.offer(frame(.95), observed_at=i * .5))
        self.assertTrue(notifier.offer(frame(.95), observed_at=5.001))
        message = notifier.tasks.get_nowait()['payload']['content']['text']
        self.assertIn('连续出现 > 5 秒', message)

    def test_missing_person_resets_timer_even_when_id_is_reused(self):
        notifier = self.make_notifier()
        for i in range(9):
            self.assertFalse(notifier.offer(frame(.95), observed_at=i * .5))
        self.assertFalse(notifier.offer(frame(), observed_at=4.5))
        for i in range(11):
            self.assertFalse(notifier.offer(frame(.95), observed_at=5 + i * .5))
        self.assertTrue(notifier.offer(frame(.95), observed_at=10.01))

    def test_different_people_cannot_accumulate_time(self):
        notifier = self.make_notifier()
        for i in range(15):
            sample = frame(.99)
            sample['detections'][0]['id'] = i // 5 + 1
            self.assertFalse(notifier.offer(sample, observed_at=i * .5))

    def test_duration_is_appearance_time_and_current_confidence_must_pass(self):
        notifier = self.make_notifier()
        for i in range(13):
            self.assertFalse(notifier.offer(frame(.7), observed_at=i * .5))
        self.assertTrue(notifier.offer(frame(.9), observed_at=6.5))

    def test_capture_gap_generation_change_and_detection_off_reset(self):
        for mode in ('gap', 'generation', 'disabled'):
            notifier = self.make_notifier()
            for i in range(9):
                self.assertFalse(notifier.offer(frame(.95), observed_at=i * .5))
            sample = frame(.95)
            observed = 4.5
            if mode == 'gap':
                observed = 10.
            elif mode == 'generation':
                sample['generation'] = 1
            else:
                sample['settings'] = {'detection_enabled': False}
            self.assertFalse(notifier.offer(sample, observed_at=observed), mode)
            self.assertEqual(notifier.status()['longest_visible_seconds'], 0., mode)

    def test_rule_changes_reset_presence_and_are_applied(self):
        notifier = self.make_notifier()
        for i in range(9):
            self.assertFalse(notifier.offer(frame(.9), observed_at=i * .5))
        notifier.configure({'threshold': .9, 'min_visible_seconds': 2.})
        for i in range(5):
            self.assertFalse(notifier.offer(frame(.95), observed_at=5 + i * .5))
        self.assertTrue(notifier.offer(frame(.95), observed_at=7.1))
        text = notifier.tasks.get_nowait()['payload']['content']['text']
        self.assertIn('> 2 秒', text)
        self.assertIn('> 90%', text)

    def test_invalid_rule_values_are_rejected(self):
        notifier = self.make_notifier()
        for values in ({'threshold': float('nan')}, {'threshold': 1.},
                       {'min_visible_seconds': 0}, {'min_visible_seconds': float('inf')}):
            with self.assertRaises(ValueError):
                notifier.configure(values)

    def test_disabled_notifier_never_enqueues_detection(self):
        notifier = self.make_notifier()
        notifier.configure({'enabled': False})
        self.assertFalse(notifier.offer(frame(.99)))

    def test_credential_never_appears_in_status_and_settings_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            path.write_text(json.dumps({'feishu': {'webhook': HOOK}}), encoding='utf-8')
            notifier = FeishuNotifier(path)
            notifier.configure({'enabled': False, 'cooldown_seconds': 120,
                                'threshold': .9, 'min_visible_seconds': 7.5})
            self.assertNotIn(HOOK, json.dumps(notifier.status()))
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['feishu']['webhook'], HOOK)
            self.assertEqual(saved['feishu']['cooldown_seconds'], 120)
            self.assertFalse(FeishuNotifier(path).enabled)
            self.assertEqual(FeishuNotifier(path).threshold, .9)
            self.assertEqual(FeishuNotifier(path).min_visible_seconds, 7.5)

    def test_feishu_business_error_is_not_success(self):
        notifier = self.make_notifier()
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, *_): return b'{"code":19024,"msg":"keyword missing"}'
        with patch('vision.notifications.urllib.request.urlopen', return_value=Response()):
            with self.assertRaisesRegex(RuntimeError, '19024'):
                notifier._post({'msg_type': 'text', 'content': {'text': 'test'}})

    def test_worker_sends_without_blocking_capture_and_tracks_result(self):
        notifier = self.make_notifier()
        delivered = []
        with patch.object(notifier, '_post', side_effect=lambda payload, **_: delivered.append(payload)):
            notifier.start()
            try:
                for i in range(11):
                    self.assertFalse(notifier.offer(frame(.91), observed_at=i * .5))
                self.assertTrue(notifier.offer(frame(.91), observed_at=5.5))
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline and notifier.status()['sent_count'] == 0:
                    time.sleep(.01)
                self.assertEqual(notifier.status()['sent_count'], 1)
                self.assertEqual(notifier.status()['last_kind'], 'detection')
                self.assertEqual(len(delivered), 1)
            finally:
                notifier.close()

    def test_same_id_does_not_repeat_after_multiple_cooldown_periods(self):
        notifier = self.make_notifier()
        with patch.object(notifier, '_post') as deliver:
            notifier.start()
            try:
                self.submit_initial_person(notifier)
                baseline = notifier.last_queued
                for i in range(1, 361):
                    with patch('vision.notifications.time.monotonic', return_value=baseline + i * .5):
                        self.assertFalse(notifier.offer(frame(.99), observed_at=5.5 + i * .5))
                self.assertEqual(deliver.call_count, 1)
                self.assertEqual(notifier.status()['current_notified_people'], 1)
            finally:
                notifier.close()

    def test_next_alert_contains_only_new_ids_and_keeps_full_scene_count(self):
        notifier = self.make_notifier()
        with patch.object(notifier, '_post') as deliver:
            notifier.start()
            try:
                self.submit_initial_person(notifier)
                baseline = notifier.last_queued
                for i in range(1, 121):
                    with patch('vision.notifications.time.monotonic', return_value=baseline + i * .5):
                        offered = notifier.offer(frame(.95, .97), observed_at=5.5 + i * .5)
                    self.assertEqual(offered, i == 120)
                self.wait_for_state(notifier, lambda state: state['sent_count'] == 2)
                text = deliver.call_args_list[1].args[0]['content']['text']
                self.assertIn('人员 #2｜', text)
                self.assertNotIn('人员 #1｜', text)
                self.assertIn('当前帧共检测到 2 人', text)
                self.assertEqual(notifier.status()['current_notified_people'], 2)
            finally:
                notifier.close()

    def test_missed_frame_rule_edits_and_toggle_do_not_rearm_notified_id(self):
        notifier = self.make_notifier()
        with patch.object(notifier, '_post') as deliver:
            notifier.start()
            try:
                self.submit_initial_person(notifier)
                self.assertFalse(notifier.offer(frame(), observed_at=6.))
                notifier.configure({'threshold': .9})
                notifier.configure({'enabled': False})
                notifier.configure({'enabled': True})
                baseline = notifier.last_queued
                for i in range(14):
                    with patch('vision.notifications.time.monotonic', return_value=baseline + 100 + i * .5):
                        self.assertFalse(notifier.offer(frame(.99), observed_at=6.5 + i * .5))
                self.assertEqual(deliver.call_count, 1)
                self.assertEqual(notifier.status()['notified_people'], 1)
            finally:
                notifier.close()

    def test_restarted_capture_can_notify_same_number_in_new_generation(self):
        notifier = self.make_notifier()
        with patch.object(notifier, '_post') as deliver:
            notifier.start()
            try:
                self.submit_initial_person(notifier)
                baseline = notifier.last_queued
                for i in range(12):
                    sample = frame(.99)
                    sample['generation'] = 1
                    with patch('vision.notifications.time.monotonic', return_value=baseline + 100 + i * .5):
                        offered = notifier.offer(sample, observed_at=6 + i * .5)
                    self.assertEqual(offered, i == 11)
                self.wait_for_state(notifier, lambda state: state['sent_count'] == 2)
                self.assertEqual(deliver.call_count, 2)
                self.assertEqual(notifier.status()['current_notified_people'], 1)
            finally:
                notifier.close()

    def test_failed_delivery_does_not_mark_person_as_notified(self):
        notifier = self.make_notifier()
        with patch.object(notifier, '_post', side_effect=RuntimeError('temporary failure')) as deliver:
            with patch.object(notifier.stop, 'wait', return_value=False):
                notifier.start()
                try:
                    for i in range(11):
                        self.assertFalse(notifier.offer(frame(.95), observed_at=i * .5))
                    self.assertTrue(notifier.offer(frame(.95), observed_at=5.5))
                    self.wait_for_state(notifier, lambda state: state['failure_count'] == 1)
                    self.assertEqual(deliver.call_count, 3)
                    self.assertEqual(notifier.status()['notified_people'], 0)
                    baseline = notifier.last_queued
                    with patch('vision.notifications.time.monotonic', return_value=baseline + 61):
                        self.assertTrue(notifier.offer(frame(.95), observed_at=6.))
                finally:
                    notifier.close()

    def test_web_saved_address_is_persisted_without_changing_rules(self):
        new_hook = HOOK + '-new'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.local.json'
            path.write_text(json.dumps({'feishu': {'webhook': HOOK, 'threshold': .91,
                            'min_visible_seconds': 8, 'cooldown_seconds': 15}}), encoding='utf-8')
            notifier = FeishuNotifier(path)
            state = notifier.configure({'webhook': f'  {new_hook}  '})
            self.assertTrue(state['configured'])
            self.assertEqual((state['threshold'], state['min_visible_seconds'], state['cooldown_seconds']), (.91, 8, 15))
            self.assertNotIn(new_hook, json.dumps(state))
            with patch.dict(os.environ, {'FEISHU_WEBHOOK': HOOK}):
                self.assertEqual(FeishuNotifier(path).webhook, new_hook)

    def test_invalid_destinations_do_not_modify_saved_address(self):
        notifier = self.make_notifier()
        invalid = ['http://open.feishu.cn/open-apis/bot/v2/hook/test',
                   'https://example.com/open-apis/bot/v2/hook/test',
                   'https://open.feishu.cn.evil.test/open-apis/bot/v2/hook/test',
                   'https://user@open.feishu.cn/open-apis/bot/v2/hook/test',
                   'https://open.feishu.cn/open-apis/bot/v2/hook/',
                   HOOK + '?token=secret', HOOK + '#secret', HOOK + '/extra',
                   HOOK + '\nsecret', 42]
        for address in invalid:
            with self.assertRaises(ValueError):
                notifier.configure({'webhook': address})
            self.assertEqual(notifier.webhook, HOOK)

    def test_same_address_save_keeps_notified_ids(self):
        notifier = self.make_notifier()
        notifier.notified_keys.add((0, 1))
        notifier.notification_generation = 0
        notifier.configure({'webhook': HOOK})
        self.assertEqual(notifier.status()['notified_people'], 1)
        self.assertEqual(notifier.destination_revision, 0)

    def test_destination_change_cancels_old_queued_test(self):
        notifier = self.make_notifier()
        notifier.send_test()
        notifier.configure({'webhook': HOOK + '-new'})
        with patch.object(notifier, '_post') as deliver:
            notifier.start()
            try:
                self.wait_for_state(notifier, lambda state: state['status'] == 'idle')
                deliver.assert_not_called()
                notifier.send_test()
                self.wait_for_state(notifier, lambda state: state['sent_count'] == 1)
                self.assertEqual(deliver.call_args.kwargs['webhook'], HOOK + '-new')
            finally:
                notifier.close()

    def test_in_flight_delivery_is_not_redirected_or_acknowledged_for_new_address(self):
        notifier = self.make_notifier()
        started, finish = threading.Event(), threading.Event()
        destinations = []
        def deliver(payload, *, webhook, secret):
            destinations.append(webhook)
            started.set()
            finish.wait(timeout=2)
        with patch.object(notifier, '_post', side_effect=deliver):
            notifier.start()
            try:
                for i in range(11):
                    notifier.offer(frame(.95), observed_at=i * .5)
                self.assertTrue(notifier.offer(frame(.95), observed_at=5.5))
                self.assertTrue(started.wait(timeout=1))
                notifier.configure({'webhook': HOOK + '-new'})
                finish.set()
                state = self.wait_for_state(notifier, lambda state: state['status'] == 'idle')
                self.assertEqual(destinations, [HOOK])
                self.assertEqual(state['notified_people'], 0)
                self.assertEqual(state['sent_count'], 0)
            finally:
                finish.set()
                notifier.close()


if __name__ == '__main__':
    unittest.main()
