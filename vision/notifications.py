"""Non-blocking Feishu alerts. Credentials stay in local config/environment only."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger(__name__)
SHANGHAI = timezone(timedelta(hours=8), 'Asia/Shanghai')
DEFAULT_THRESHOLD = .85
DEFAULT_VISIBLE_SECONDS = 5.


def local_time(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, SHANGHAI).strftime('%Y-%m-%d %H:%M:%S')


def build_alert(frame: dict, threshold: float = DEFAULT_THRESHOLD,
                min_visible_seconds: float = DEFAULT_VISIBLE_SECONDS, *,
                excluded_ids: set | frozenset = frozenset()) -> dict | None:
    people = [d for d in frame.get('detections', [])
              if d.get('id') not in excluded_ids
              and isinstance(d.get('confidence'), (float, int))
              and math.isfinite(d['confidence']) and d['confidence'] > threshold
              and math.isfinite(d.get('visible_seconds', 0))
              and d.get('visible_seconds', 0) > min_visible_seconds]
    if not people:
        return None
    people.sort(key=lambda d: d['confidence'], reverse=True)
    detected_at = local_time(frame['captured_at'])
    name = (frame.get('camera') or {}).get('name', 'D435i')
    lines = ['【DepthSight 人员检测告警】', f'检测时间：{detected_at}（北京时间 UTC+8）',
             f'设备：{name}', f'信息摘要：{len(people)} 名人员连续出现超过 {min_visible_seconds:g} 秒且置信度大于 {threshold * 100:g}%；当前帧共检测到 {len(frame["detections"])} 人。',
             f'最高置信度：{people[0]["confidence"]:.2%}', '目标详情：']
    labels = {'valid': '深度有效', 'unstable': '深度波动', 'insufficient': '深度不足'}
    for d in people[:10]:
        distance = f'{d["distance_m"]:.2f} m' if d.get('distance_m') is not None else '不可用'
        first_seen = d.get('first_seen_at', local_time(frame['captured_at'] - d['visible_seconds']))
        lines.append(f'• 人员 #{d.get("id", "—")}｜连续出现 {d["visible_seconds"]:.1f} 秒｜首次出现 {first_seen}｜置信度 {d["confidence"]:.2%}｜轴向距离 {distance}｜{labels.get(d.get("depth_status"), "深度未知")}')
    if len(people) > 10:
        lines.append(f'另有 {len(people) - 10} 个目标未展开。')
    lines.extend([f'帧编号：{frame.get("sequence", "—")}',
                  f'触发条件：同一人员连续出现 > {min_visible_seconds:g} 秒，且当前置信度严格 > {threshold * 100:g}%'])
    return {'kind': 'detection', 'detected_at': detected_at,
            'person_ids': [d['id'] for d in people],
            'summary': f'{len(people)} 人连续出现超过 {min_visible_seconds:g} 秒且置信度 > {threshold * 100:g}%，最高 {people[0]["confidence"]:.2%}',
            'payload': {'msg_type': 'text', 'content': {'text': '\n'.join(lines)}}}


class FeishuNotifier:
    def __init__(self, config_path: Path | None = None, *, config: dict | None = None):
        self.path = config_path
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.tasks = queue.Queue(maxsize=1)
        self.worker = None
        self.pending = False
        self.last_queued = float('-inf')
        self.last_test = float('-inf')
        self.document = {}
        self.presence = {}
        self.presence_generation = None
        self.last_observed = None
        self.rule_revision = 0
        self.destination_revision = 0
        # Independent of appearance timers: misses/config edits must not re-alert the same ID.
        self.notified_keys = set()
        self.notification_generation = None
        error = None
        try:
            if config is None and config_path and config_path.exists():
                self.document = json.loads(config_path.read_text(encoding='utf-8'))
            options = config if config is not None else self.document.get('feishu', {})
            # A web-saved local value takes precedence over an optional deployment default.
            self.webhook = options.get('webhook', os.environ.get('FEISHU_WEBHOOK', ''))
            self.webhook = self._validate_webhook(self.webhook)
            self.secret = os.environ.get('FEISHU_SIGN_SECRET', options.get('sign_secret', ''))
            self.enabled = options.get('enabled', True)
            self.cooldown = int(options.get('cooldown_seconds', 60))
            self.threshold = float(options.get('threshold', DEFAULT_THRESHOLD))
            self.min_visible_seconds = float(options.get('min_visible_seconds', DEFAULT_VISIBLE_SECONDS))
            if not isinstance(self.enabled, bool) or not 15 <= self.cooldown <= 3600:
                raise ValueError('飞书配置的 enabled 应为布尔值，cooldown_seconds 应为 15～3600')
            self._validate_rule(self.threshold, self.min_visible_seconds)
        except Exception:
            # Never echo malformed config/URLs or exception text containing credentials.
            self.webhook, self.secret, self.enabled, self.cooldown = '', '', False, 60
            self.threshold, self.min_visible_seconds = DEFAULT_THRESHOLD, DEFAULT_VISIBLE_SECONDS
            error = '无法读取飞书本地配置，请检查 config.local.json 或环境变量'
        self.state = {'status': 'error' if error else 'idle', 'last_error': error,
                      'sent_count': 0, 'failure_count': 0, 'last_sent_at': None,
                      'last_detection_at': None, 'last_summary': None, 'last_kind': None}

    def start(self):
        self.worker = threading.Thread(target=self._run, name='feishu-alerts', daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=7)

    def status(self):
        with self.lock:
            return {**self.state, 'enabled': self.enabled, 'configured': bool(self.webhook),
                    'threshold': self.threshold, 'min_visible_seconds': self.min_visible_seconds,
                    'tracked_people': len(self.presence),
                    'deduplicate_by_track': True,
                    'notified_people': sum(generation == self.notification_generation for generation, _ in self.notified_keys),
                    'current_notified_people': sum((self.notification_generation, person_id) in self.notified_keys for person_id in self.presence),
                    'longest_visible_seconds': round(max((p['last'] - p['first'] for p in self.presence.values()), default=0), 1),
                    'cooldown_seconds': self.cooldown,
                    'cooldown_remaining_s': math.ceil(max(0, self.cooldown - (time.monotonic() - self.last_queued)))}

    def configure(self, values):
        with self.lock:
            enabled = values.get('enabled', self.enabled)
            cooldown = values.get('cooldown_seconds', self.cooldown)
            threshold = values.get('threshold', self.threshold)
            visible_seconds = values.get('min_visible_seconds', self.min_visible_seconds)
            webhook = self._validate_webhook(values.get('webhook', self.webhook))
            if not isinstance(enabled, bool) or not isinstance(cooldown, int) or not 15 <= cooldown <= 3600:
                raise ValueError('冷却时间应为 15～3600 秒')
            self._validate_rule(threshold, visible_seconds)
            if self.path:
                document = {**self.document, 'feishu': {**self.document.get('feishu', {}),
                            'enabled': enabled, 'cooldown_seconds': cooldown,
                            'threshold': threshold, 'min_visible_seconds': visible_seconds,
                            'webhook': webhook}}
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(self.path)
                self.document = document
            destination_changed = webhook != self.webhook
            if destination_changed:
                self.destination_revision += 1
                self.notified_keys.clear()
                self.last_queued = self.last_test = float('-inf')
                self.state.update(status='idle', last_error=None, last_sent_at=None,
                                  last_kind=None, last_summary=None, last_detection_at=None)
            if destination_changed or (enabled, threshold, visible_seconds) != (self.enabled, self.threshold, self.min_visible_seconds):
                self.rule_revision += 1
                self.reset_presence()
            self.enabled, self.cooldown = enabled, cooldown
            self.threshold, self.min_visible_seconds = float(threshold), float(visible_seconds)
            self.webhook = webhook
            return self.status()

    @staticmethod
    def _validate_webhook(value):
        if not isinstance(value, str):
            raise ValueError('请输入完整的飞书 Webhook 地址')
        value = value.strip()
        if not value:
            return ''
        if len(value) > 2048 or any(ord(char) < 33 for char in value):
            raise ValueError('飞书 Webhook 地址不能包含空白或控制字符')
        parsed = urlsplit(value)
        if (parsed.scheme != 'https' or parsed.netloc != 'open.feishu.cn'
                or not re.fullmatch(r'/open-apis/bot/v2/hook/[A-Za-z0-9_-]+', parsed.path)
                or parsed.query or parsed.fragment):
            raise ValueError('请输入 https://open.feishu.cn/open-apis/bot/v2/hook/ 开头的完整机器人地址')
        return value

    @staticmethod
    def _validate_rule(threshold, seconds):
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not .1 <= threshold <= .99):
            raise ValueError('推送置信度应为 10%～99%')
        if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds) or not .1 <= seconds <= 3600):
            raise ValueError('连续出现时长应为 0.1～3600 秒')

    def reset_presence(self):
        with self.lock:
            self.presence.clear()
            self.last_observed = None
            self.presence_generation = None

    def _observe(self, frame, observed_at):
        generation = frame.get('generation', 0)
        if generation != self.notification_generation:
            self.notified_keys.clear()
            self.notification_generation = generation
        if self.last_observed is not None:
            if generation == self.presence_generation and observed_at <= self.last_observed:
                return None  # Duplicate/out-of-order timestamps cannot advance dwell time.
            if generation != self.presence_generation or observed_at - self.last_observed > 1.:
                self.presence.clear()
        self.presence_generation, self.last_observed = generation, observed_at
        current, detections = {}, []
        for d in frame.get('detections', []):
            person_id = d.get('id')
            if not isinstance(person_id, int) or person_id in current:
                continue
            previous = self.presence.get(person_id)
            first = previous['first'] if previous else observed_at
            current[person_id] = {'first': first, 'last': observed_at}
            elapsed = observed_at - first
            detections.append({**d, 'visible_seconds': elapsed,
                               'first_seen_at': local_time(frame['captured_at'] - elapsed)})
        # A missed detection resets that person's timer, even if the tracker reuses its ID later.
        self.presence = current
        return {**frame, 'detections': detections}

    def offer(self, frame, *, observed_at=None):
        with self.lock:
            if (not self.enabled or not self.webhook or self.stop.is_set()
                    or not frame.get('settings', {}).get('detection_enabled', True)):
                self.reset_presence()
                return False
            now = time.monotonic()
            observed = self._observe(frame, now if observed_at is None else observed_at)
            if observed is None or self.pending or now - self.last_queued < self.cooldown:
                return False
            generation = frame.get('generation', 0)
            excluded = {person_id for session, person_id in self.notified_keys if session == generation}
            event = build_alert(observed, self.threshold, self.min_visible_seconds, excluded_ids=excluded)
            if event is None:
                return False
            event['rule_revision'] = self.rule_revision
            event['target_keys'] = [(generation, person_id) for person_id in event['person_ids']]
            event.update(destination_revision=self.destination_revision,
                         webhook=self.webhook, secret=self.secret)
            self.tasks.put_nowait(event)
            self.pending = True
            self.last_queued = now
            self.state.update(status='queued', last_detection_at=event['detected_at'])
            return True

    def send_test(self):
        with self.lock:
            if not self.webhook:
                raise ValueError('尚未配置飞书 Webhook')
            if self.pending or time.monotonic() - self.last_test < 15:
                raise ValueError('推送正在进行或测试过于频繁，请稍后再试')
            event = {'kind': 'test', 'summary': '飞书推送连接测试（非人员告警）',
                     'destination_revision': self.destination_revision,
                     'webhook': self.webhook, 'secret': self.secret,
                     'payload': {'msg_type': 'text', 'content': {'text': '\n'.join([
                         '【DepthSight 飞书推送测试】', f'时间：{local_time()}（北京时间 UTC+8）',
                         '本消息为连接测试，不表示检测到人员。',
                         f'人员告警规则：同一人员连续出现 > {self.min_visible_seconds:g} 秒，且当前置信度严格 > {self.threshold * 100:g}%；同一采集会话同编号仅成功告警一次；冷却间隔 {self.cooldown} 秒。',
                         '推送内容：检测时间、连续出现时长、人数摘要、置信度、轴向距离与深度质量。'])}}}
            self.tasks.put_nowait(event)
            self.pending = True
            self.last_test = time.monotonic()
            self.state['status'] = 'queued'
            return self.status()

    def _post(self, payload, *, webhook=None, secret=None):
        body = dict(payload)
        webhook = self.webhook if webhook is None else webhook
        secret = self.secret if secret is None else secret
        if secret:
            timestamp = str(int(time.time()))
            signature = hmac.new(f'{timestamp}\n{secret}'.encode(), digestmod=hashlib.sha256).digest()
            body.update(timestamp=timestamp, sign=base64.b64encode(signature).decode())
        request = urllib.request.Request(webhook, data=json.dumps(body, ensure_ascii=False).encode('utf-8'),
                                         headers={'Content-Type': 'application/json; charset=utf-8'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                result = json.loads(response.read(65536))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'飞书 HTTP {exc.code}') from None
        except urllib.error.URLError:
            raise RuntimeError('飞书网络连接失败，请检查网络或代理') from None
        except (TimeoutError, OSError):
            raise RuntimeError('飞书连接超时或网络不可用') from None
        except (ValueError, TypeError):
            raise RuntimeError('飞书返回了无法识别的响应') from None
        if not isinstance(result, dict) or result.get('code', result.get('StatusCode')) != 0:
            code = result.get('code', result.get('StatusCode', 'unknown')) if isinstance(result, dict) else 'unknown'
            raise RuntimeError(f'飞书拒绝消息（code={code}），请检查机器人关键词、签名及 IP 限制')

    def _run(self):
        while not self.stop.is_set():
            try:
                event = self.tasks.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    if event['destination_revision'] != self.destination_revision:
                        self.state['status'] = 'idle'
                        continue
                    if event['kind'] == 'detection' and (not self.enabled or event['rule_revision'] != self.rule_revision):
                        self.state['status'] = 'idle'
                        continue
                    self.state['status'] = 'sending'
                # Timeout or transient failure: bounded retry in this worker, never the camera thread.
                for attempt in range(3):
                    if self.stop.is_set():
                        break
                    with self.lock:
                        if event['destination_revision'] != self.destination_revision:
                            self.state['status'] = 'idle'
                            break
                        if event['kind'] == 'detection' and (not self.enabled or event['rule_revision'] != self.rule_revision):
                            self.state['status'] = 'idle'
                            break
                    try:
                        self._post(event['payload'], webhook=event['webhook'], secret=event['secret'])
                    except Exception as exc:
                        if attempt < 2 and not self.stop.wait(2 * (attempt + 1)):
                            continue
                        with self.lock:
                            if event['destination_revision'] == self.destination_revision:
                                self.state.update(status='error', last_error=str(exc).replace(event['webhook'], '[redacted]'),
                                                  failure_count=self.state['failure_count'] + 1)
                        log.warning('Feishu push failed after retries; see local console status')
                    else:
                        with self.lock:
                            if event['destination_revision'] != self.destination_revision:
                                break
                            if event['kind'] == 'detection':
                                # Commit only acknowledged deliveries. Failed sends remain retryable.
                                self.notified_keys.update(key for key in event['target_keys']
                                                          if key[0] == self.notification_generation)
                            self.state.update(status='sent', last_error=None, last_sent_at=local_time(),
                                              last_summary=event['summary'], last_kind=event['kind'],
                                              sent_count=self.state['sent_count'] + 1)
                    break
            finally:
                with self.lock:
                    self.pending = False
                self.tasks.task_done()
