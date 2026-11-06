'use strict';

const $ = id => document.getElementById(id);
const ctxRGB = $('rgb-canvas').getContext('2d');
const ctxDepth = $('depth-canvas').getContext('2d');
const viewNames = { console: ['实时视觉控制台', '同步观察真实世界的色彩、深度与人员信息。', '实时控制台'], events: ['运行记录', '查看当前浏览器会话中的设备、检测与配置事件。', '运行记录'], device: ['设备信息', '来自 RealSense SDK 的真实设备信息与处理架构。', '设备信息'], guide: ['使用说明', '了解采集、人员检测、深度测距与快照导出。', '使用说明'] };
let socket, reconnectTimer, lastStatus = null, initialized = false, online = false;
let pendingFrame = null, decoding = false, renderGeneration = 0, lastFrameAt = 0, lastMessageAt = 0;
let displayedSequence = 0, lastStatsAt = 0, lastHistoryAt = 0, previousPeople = null;
let previousCamera = '', previousModel = '', eventEntries = [], fpsHistory = [], toastTimer;
let lastImageRGB = null, lastImageDepth = null, lastFrameMeta = null, probe = null;
let cameraBusy = false, snapshotBusy = false;
let knownGeneration = -1;
let notificationsInitialized = false, previousNotificationCount = null, previousNotificationError = null;

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(message) {
  $('toast').textContent = message;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4000);
}

function addEvent(kind, message) {
  eventEntries.unshift({ time: new Date().toLocaleTimeString('zh-CN', { hour12: false }), kind, message });
  eventEntries = eventEntries.slice(0, 100);
  renderEvents();
}

function renderEvents() {
  $('events-body').innerHTML = eventEntries.length ? eventEntries.map(e => `<tr><td class="event-time">${escapeHTML(e.time)}</td><td><span class="event-kind">${escapeHTML(e.kind)}</span></td><td>${escapeHTML(e.message)}</td></tr>`).join('') : '<tr><td colspan="3" class="empty-cell">本次会话暂无运行记录</td></tr>';
}

function switchView(view) {
  if (!viewNames[view]) return;
  document.querySelectorAll('.view').forEach(el => { el.hidden = el.id !== `view-${view}`; });
  document.querySelectorAll('.nav-item').forEach(el => el.classList.toggle('active', el.dataset.view === view));
  const [title, description, breadcrumb] = viewNames[view];
  $('page-title').textContent = title;
  $('page-description').textContent = description;
  $('breadcrumb-name').textContent = breadcrumb;
  location.hash = view === 'console' ? '' : view;
}

document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => switchView(button.dataset.view)));
window.addEventListener('hashchange', () => switchView(location.hash.slice(1) || 'console'));
$('clear-events').addEventListener('click', () => { eventEntries = []; renderEvents(); });
$('date-label').textContent = new Date().toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit', weekday: 'short' });

async function request(path, method = 'GET', payload) {
  const response = await fetch(path, { method, headers: { 'Content-Type': 'application/json', 'X-Console-Request': '1' }, ...(payload === undefined ? {} : { body: JSON.stringify(payload) }) });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(typeof data.detail === 'string' ? data.detail : `请求失败（${response.status}），请检查参数或服务状态`);
  }
  return response.json();
}

function syncControls(settings, keys = Object.keys(settings)) {
  if (keys.includes('detection_enabled')) $('detection-enabled').checked = settings.detection_enabled;
  if (keys.includes('confidence')) {
    $('confidence').value = Math.round(settings.confidence * 100);
    $('confidence-value').textContent = `${Math.round(settings.confidence * 100)}%`;
  }
  if (keys.includes('inference_size')) $('inference-size').value = settings.inference_size;
  if (keys.includes('near_m')) $('near-distance').value = settings.near_m;
  if (keys.includes('far_m')) $('far-distance').value = settings.far_m;
}

async function updateSettings(values, control) {
  if (control) control.disabled = true;
  try {
    const settings = await request('/api/settings', 'PATCH', values);
    const keys = Object.keys(values);
    if (lastStatus) for (const key of keys) lastStatus.settings[key] = settings[key];
    syncControls(settings, keys);
    const names = { detection_enabled: '人员检测', confidence: '置信度阈值', inference_size: '推理尺寸', near_m: '最近显示距离', far_m: '最远显示距离' };
    const details = Object.entries(values).map(([k, v]) => `${names[k]}：${k === 'detection_enabled' ? (v ? '开启' : '关闭') : v}`).join('；');
    addEvent('配置', details);
    toast('配置已更新，将用于后续采集帧');
  } catch (error) {
    toast(error.message);
    if (lastStatus) syncControls(lastStatus.settings, Object.keys(values));
  } finally {
    if (control) control.disabled = !online;
  }
}

$('confidence').addEventListener('input', event => { $('confidence-value').textContent = `${event.target.value}%`; });
$('confidence').addEventListener('change', event => updateSettings({ confidence: Number(event.target.value) / 100 }, event.target));
$('inference-size').addEventListener('change', event => updateSettings({ inference_size: Number(event.target.value) }, event.target));
$('detection-enabled').addEventListener('change', event => updateSettings({ detection_enabled: event.target.checked }, event.target));
$('depth-form').addEventListener('submit', event => {
  event.preventDefault();
  const near = Number($('near-distance').value), far = Number($('far-distance').value);
  if (!Number.isFinite(near) || !Number.isFinite(far) || far - near < .2 - 1e-9) return toast('最远距离至少应比最近距离大 0.2 米');
  updateSettings({ near_m: near, far_m: far }, event.submitter);
});

async function updateNotifications(values, control) {
  control.disabled = true;
  try {
    const state = await request('/api/notifications', 'PATCH', values);
    if (values.enabled !== undefined) $('notifications-enabled').checked = state.enabled;
    if (values.cooldown_seconds !== undefined) $('notification-cooldown').value = state.cooldown_seconds;
    if (values.threshold !== undefined) $('notification-threshold').value = Number((state.threshold * 100).toFixed(1));
    if (values.min_visible_seconds !== undefined) $('notification-duration').value = state.min_visible_seconds;
    if (values.webhook !== undefined) $('notification-webhook').value = '';
    renderNotifications(state);
    addEvent('推送配置', values.webhook !== undefined ? '飞书推送地址已保存（完整地址不写入运行记录）' : values.enabled === undefined ? `飞书规则：连续出现 > ${state.min_visible_seconds}s、置信度 > ${Number((state.threshold * 100).toFixed(1))}%，冷却 ${state.cooldown_seconds}s` : `飞书告警已${state.enabled ? '开启' : '关闭'}`);
    toast(values.webhook !== undefined ? '推送地址已保存并生效，可发送连接测试' : '飞书推送配置已保存');
  } catch (error) { toast(error.message); notificationsInitialized = false; }
  finally { control.disabled = !online; }
}
$('notifications-enabled').addEventListener('change', event => updateNotifications({ enabled: event.target.checked }, event.target));
$('notification-webhook-form').addEventListener('submit', event => {
  event.preventDefault();
  const webhook = $('notification-webhook').value.trim();
  try {
    const address = new URL(webhook);
    if (address.protocol !== 'https:' || address.host !== 'open.feishu.cn' || address.username || address.password || address.search || address.hash || !/^\/open-apis\/bot\/v2\/hook\/[A-Za-z0-9_-]+$/.test(address.pathname) || /\s/.test(webhook)) throw new Error();
  } catch { return toast('请输入完整的飞书自定义机器人 HTTPS Webhook 地址'); }
  updateNotifications({ webhook }, event.submitter);
});
$('notification-form').addEventListener('submit', event => {
  event.preventDefault();
  const seconds = Number($('notification-cooldown').value);
  const threshold = Number($('notification-threshold').value) / 100;
  const duration = Number($('notification-duration').value);
  if (!Number.isInteger(seconds) || seconds < 15 || seconds > 3600) return toast('推送冷却间隔应为 15～3600 秒的整数');
  if (!Number.isFinite(threshold) || threshold < .1 || threshold > .99) return toast('推送置信度阈值应为 10%～99%');
  if (!Number.isFinite(duration) || duration < .1 || duration > 3600) return toast('连续出现时长应为 0.1～3600 秒');
  updateNotifications({ cooldown_seconds: seconds, threshold, min_visible_seconds: duration }, event.submitter);
});
$('notification-test').addEventListener('click', async event => {
  event.target.disabled = true;
  try {
    const state = await request('/api/notifications/test', 'POST');
    renderNotifications(state);
    toast('连接测试已进入发送队列，请查看推送状态');
    addEvent('飞书', '已请求发送连接测试（非人员检测告警）');
  } catch (error) { toast(error.message); }
});
$('camera-button').addEventListener('click', async () => {
  if (!lastStatus || cameraBusy) return;
  cameraBusy = true;
  $('camera-button').disabled = true;
  try {
    const action = lastStatus.active ? 'stop' : 'start';
    const state = await request(`/api/camera/${action}`, 'POST');
    renderStatus(state);
    addEvent('操作', action === 'stop' ? '已停止相机采集，释放 USB 设备' : '开始连接相机并恢复采集');
    if (action === 'stop') clearFrames('采集已停止', '点击右上角“开始采集”恢复');
  } catch (error) { toast(error.message); }
  finally { cameraBusy = false; $('camera-button').disabled = !online; }
});
$('reload-model').addEventListener('click', async () => {
  try { await request('/api/model/reload', 'POST'); toast('已请求重新加载模型；请保持采集和人员检测开启'); addEvent('模型', '请求重新加载检测模型'); }
  catch (error) { toast(error.message); }
});
$('snapshot-button').addEventListener('click', async () => {
  if (snapshotBusy) return;
  snapshotBusy = true;
  $('snapshot-button').disabled = true;
  try {
    const response = await fetch('/api/snapshot');
    if (!response.ok) {
      const info = await response.json();
      throw new Error(info.detail || '快照导出失败');
    }
    const url = URL.createObjectURL(await response.blob());
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = response.headers.get('Content-Disposition')?.match(/filename="([^"]+)"/)?.[1] || 'depthsight-snapshot.zip';
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
    addEvent('快照', '已导出同步 RGB、标注图、灰度及原始深度与元数据');
    toast('同步快照已导出，原始深度保留为 16 位 PNG');
  } catch (error) { toast(error.message); }
  finally { snapshotBusy = false; $('snapshot-button').disabled = !online || !lastFrameAt || performance.now() - lastFrameAt > 3000; }
});

document.querySelectorAll('[data-fullscreen]').forEach(button => button.addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $(button.dataset.fullscreen).requestFullscreen();
  } catch { toast('当前浏览器未允许全屏显示'); }
}));

$('show-boxes').addEventListener('change', drawCurrentFrame);
$('depth-canvas').addEventListener('click', async event => {
  if (!lastFrameMeta || performance.now() - lastFrameAt > 3000) return;
  const rect = event.currentTarget.getBoundingClientRect();
  // Fullscreen uses object-fit: contain, so account for horizontal/vertical letterboxing.
  const factor = Math.min(rect.width / 640, rect.height / 480);
  const x = Math.floor((event.clientX - rect.left - (rect.width - 640 * factor) / 2) / factor);
  const y = Math.floor((event.clientY - rect.top - (rect.height - 480 * factor) / 2) / factor);
  if (x < 0 || y < 0 || x >= 640 || y >= 480) return;
  try {
    const data = await request(`/api/depth?x=${x}&y=${y}`);
    probe = { x, y, until: performance.now() + 5000 };
    $('depth-probe').textContent = `(${x}, ${y}) ${data.distance_m === null ? '无有效深度' : `${data.distance_m.toFixed(3)} m`} · #${data.sequence}`;
    $('depth-probe').hidden = false;
    drawCurrentFrame();
  } catch (error) { toast(error.message); }
});

function setConnection(connected) {
  online = connected;
  $('connection').innerHTML = `<i class="dot ${connected ? 'good' : 'bad'}"></i>${connected ? '本地服务已连接' : '服务重连中'}`;
  $('camera-button').disabled = !connected || cameraBusy;
  document.querySelectorAll('#detection-enabled, #confidence, #inference-size, #depth-form button, #near-distance, #far-distance, #reload-model, #notifications-enabled, #notification-form input, #notification-form button, #notification-webhook-form input, #notification-webhook-form button').forEach(el => { el.disabled = !connected; });
  if (!connected) $('notification-test').disabled = true;
}

function clearFrames(title, description) {
  renderGeneration++;
  pendingFrame = null;
  lastFrameAt = 0;
  displayedSequence = 0;
  lastImageRGB?.close(); lastImageDepth?.close();
  lastImageRGB = lastImageDepth = lastFrameMeta = null;
  ctxRGB.clearRect(0, 0, 640, 480);
  ctxDepth.clearRect(0, 0, 640, 480);
  for (const id of ['rgb-empty', 'depth-empty']) {
    $(id).querySelector('strong').textContent = title;
    $(id).querySelector('span').textContent = description;
    $(id).hidden = false;
  }
  document.querySelectorAll('.live-badge').forEach(el => { el.hidden = true; });
  $('depth-probe').hidden = true;
  $('snapshot-button').disabled = true;
  $('stream-seq').textContent = '';
  $('metric-people').textContent = '—';
  $('metric-fps').textContent = '—';
  $('metric-latency').textContent = '—';
  $('capture-fps').textContent = '—';
  $('rgb-caption').textContent = 'RGB · 640 × 480';
  $('depth-caption').textContent = 'DEPTH · 等待新帧';
  $('depth-valid').textContent = '有效深度 —';
  $('result-count').textContent = '0';
  $('detections-body').innerHTML = `<tr><td colspan="5" class="empty-cell">${escapeHTML(title)}</td></tr>`;
}

function renderStatus(state, fromFrame = false) {
  if (state.generation < knownGeneration) return;
  knownGeneration = state.generation;
  lastStatus = state;
  renderNotifications(state.notifications);
  if (!initialized) { syncControls(state.settings); initialized = true; }
  const streaming = state.camera_status === 'streaming';
  const labels = { connecting: '连接中', streaming: '在线', stopped: '已停止', error: '设备异常' };
  const label = labels[state.camera_status] || '未知';
  const color = streaming ? 'good' : state.camera_status === 'error' ? 'bad' : 'waiting';
  $('metric-camera').innerHTML = `${label}<span class="status-pulse"></span>`;
  $('metric-camera').classList.toggle('good', streaming);
  $('side-dot').className = `dot ${color}`;
  $('side-state').textContent = streaming ? `USB ${state.camera?.usb || '—'} 已连接` : label;
  $('metric-device').textContent = state.camera ? `D435i · USB ${state.camera.usb} · 640 × 480` : '等待 RealSense D435i';
  $('camera-button').innerHTML = `<svg><use href="#i-${state.active ? 'pause' : 'play'}"/></svg><span>${state.active ? '停止采集' : '开始采集'}</span>`;
  $('camera-button').disabled = !online || cameraBusy;
  $('capture-fps').textContent = streaming && state.capture_fps ? state.capture_fps.toFixed(1) : '—';
  const enabled = state.settings.detection_enabled;
  const modelLabels = { loading: '加载中', ready: '已就绪', error: '异常' };
  $('metric-model').textContent = `${state.model_name} · ${enabled ? (modelLabels[state.model_status] || '等待') : '检测关闭'}`;
  $('model-backend').textContent = `轻量人员检测 · ${state.device}`;
  $('model-dot').className = `model-dot ${state.model_status === 'ready' ? 'good' : state.model_status === 'error' ? 'bad' : ''}`;
  $('reload-model').hidden = state.model_status !== 'error';
  $('people-foot').textContent = !enabled ? '人员检测已关闭' : state.model_status === 'ready' ? 'RGB 识别 · 深度辅助测距' : '等待检测模型就绪';
  const issues = [];
  if (state.camera_error) issues.push(`相机：${state.camera_error}；服务会自动重试连接。`);
  if (enabled && state.model_error) issues.push(`检测模型：${state.model_error}；可尝试重新加载。`);
  if (!issues.length && streaming && enabled && state.model_status === 'loading') issues.push('正在加载并预热人员检测模型，首次启动需要数秒。');
  $('alert').hidden = !issues.length;
  $('alert').textContent = issues.join(' ');
  if (previousCamera !== state.camera_status) {
    addEvent('相机', `${label}${state.camera_error ? `：${state.camera_error}` : ''}`);
    previousCamera = state.camera_status;
  }
  if (previousModel !== state.model_status) {
    addEvent('模型', `${state.model_name} ${modelLabels[state.model_status] || state.model_status}${state.model_error ? `：${state.model_error}` : ''}`);
    previousModel = state.model_status;
  }
  if (!streaming) {
    clearFrames(state.camera_status === 'stopped' ? '采集已停止' : state.camera_status === 'error' ? '相机连接异常' : '正在连接相机', state.camera_status === 'stopped' ? '点击右上角“开始采集”恢复' : '请检查 USB 连接及设备占用');
    $('stream-dot').className = `dot ${color}`;
    $('stream-text').textContent = state.camera_status === 'stopped' ? '采集已停止 · 设备已释放' : '等待设备数据流';
  } else if (fromFrame) {
    $('metric-people').textContent = enabled && state.model_status === 'ready' ? state.people_count : '—';
    $('metric-fps').textContent = state.process_fps ? state.process_fps.toFixed(1) : '—';
    $('metric-latency').textContent = enabled && state.inference_ms !== null ? Math.round(state.inference_ms) : '—';
    $('stream-dot').className = 'dot good';
    $('stream-text').textContent = `双流已同步 · 处理延迟 ${Math.round(state.pipeline_ms)} ms`;
    $('stream-seq').textContent = `FRAME #${state.sequence}`;
    $('depth-valid').textContent = `有效深度 ${(state.depth_valid_ratio * 100).toFixed(1)}%`;
    $('legend-near').textContent = `${state.settings.near_m.toFixed(1)} m 近`;
    $('legend-far').textContent = `远 ${state.settings.far_m.toFixed(1)} m`;
    $('rgb-caption').textContent = `RGB · ${state.width} × ${state.height} · #${state.sequence}`;
    $('depth-caption').textContent = `DEPTH · 对齐 RGB · #${state.sequence}`;
    renderDetections(state);
    if (performance.now() - lastHistoryAt > 1000) {
      fpsHistory.push(state.process_fps || 0);
      fpsHistory = fpsHistory.slice(-32);
      const max = Math.max(30, ...fpsHistory);
      $('fps-spark').querySelector('polyline').setAttribute('points', fpsHistory.map((fps, i) => `${i / Math.max(1, fpsHistory.length - 1) * 100},${29 - fps / max * 26}`).join(' '));
      if (enabled && state.model_status === 'ready' && previousPeople !== state.people_count) {
        addEvent('检测', `当前检测 ${state.people_count} 人${previousPeople === null ? '' : `（上次 ${previousPeople} 人）`}`);
        previousPeople = state.people_count;
      }
      lastHistoryAt = performance.now();
    }
  }
  $('footer-node').textContent = `D435i / ${state.model_name} / ${state.device}`;
  renderDevice(state);
}

function renderNotifications(state) {
  if (!state) return;
  if (!notificationsInitialized) {
    $('notifications-enabled').checked = state.enabled;
    $('notification-cooldown').value = state.cooldown_seconds;
    $('notification-threshold').value = Number((state.threshold * 100).toFixed(1));
    $('notification-duration').value = state.min_visible_seconds;
    notificationsInitialized = true;
  }
  const labels = { idle: '已就绪，等待触发', queued: '等待发送', sending: '正在推送', sent: '推送成功', error: '推送异常' };
  const pending = ['queued', 'sending'].includes(state.status);
  $('notification-webhook').placeholder = state.configured ? '已配置，粘贴完整新地址可替换' : '粘贴完整 https://open.feishu.cn/… 地址';
  $('notification-webhook-help').textContent = state.configured ? '地址已配置。留空保持原地址；完整地址不自动回显。' : '尚未配置推送地址，粘贴完整机器人地址后点击保存。';
  $('notification-rule').textContent = `同一人员连续出现 > ${state.min_visible_seconds} 秒，且当前置信度 > ${Number((state.threshold * 100).toFixed(1))}% 时推送。`;
  $('notification-presence').textContent = state.enabled ? `当前跟随 ${state.tracked_people} 人 · 其中 ${state.current_notified_people ?? 0} 人已通知 · 最长连续出现 ${state.longest_visible_seconds.toFixed(1)} / ${state.min_visible_seconds} 秒` : '自动告警已关闭，连续出现计时已清空';
  $('notification-status').textContent = !state.configured ? 'Webhook 未配置' : !state.enabled ? '自动告警已关闭' : (labels[state.status] || '等待触发');
  $('notification-dot').className = `dot ${state.status === 'error' ? 'bad' : state.enabled && state.configured ? 'good' : 'waiting'}`;
  $('notification-count').textContent = `${state.sent_count} 条`;
  $('notification-detail').textContent = state.last_sent_at ? `最近成功：${state.last_sent_at}（北京时间）${state.last_kind === 'test' ? ' · 连接测试' : ''}${state.cooldown_remaining_s ? ` · 冷却剩余 ${state.cooldown_remaining_s}s` : ''}` : `独立推送线程 · 冷却 ${state.cooldown_seconds}s · 不阻塞画面`;
  $('notification-error').hidden = !state.last_error;
  $('notification-error').textContent = state.last_error || '';
  $('notification-test').disabled = !online || !state.configured || pending;
  if (previousNotificationCount !== null && state.sent_count > previousNotificationCount) addEvent('飞书', `${state.last_kind === 'test' ? '连接测试' : '人员告警'}发送成功：${state.last_summary}`);
  if (state.last_error && previousNotificationError !== state.last_error) addEvent('飞书异常', state.last_error);
  previousNotificationCount = state.sent_count;
  previousNotificationError = state.last_error;
}

function renderDetections(state) {
  const detections = state.detections || [];
  $('result-count').textContent = detections.length;
  if (!detections.length) {
    const reason = !state.settings.detection_enabled ? '人员检测已关闭，双画面正常采集' : state.model_status !== 'ready' ? '等待检测模型就绪' : '当前画面未检测到人员';
    $('detections-body').innerHTML = `<tr><td colspan="5" class="empty-cell">${reason}</td></tr>`;
    return;
  }
  const depthLabels = { valid: '深度有效', unstable: '深度波动', insufficient: '深度不足' };
  $('detections-body').innerHTML = detections.map(d => `<tr><td class="person-id">#${String(d.id).padStart(2, '0')}</td><td>人员 <span class="muted mono">person</span></td><td><div class="confidence-cell"><span>${Math.round(d.confidence * 100)}%</span><span class="confidence-bar"><span style="width:${Math.round(d.confidence * 100)}%"></span></span></div></td><td class="person-id">${d.distance_m === null ? '—' : `${d.depth_status === 'unstable' ? '≈ ' : ''}${d.distance_m.toFixed(2)} m`}</td><td><span class="depth-state ${d.depth_status}">${depthLabels[d.depth_status]}</span></td></tr>`).join('');
}

function renderDevice(state) {
  const camera = state.camera;
  const fields = camera ? [['设备名称', camera.name], ['序列号', camera.serial], ['连接状态', state.camera_status], ['USB 协议', camera.usb], ['采集分辨率', `${camera.width} × ${camera.height} / RGB + Z16`], ['设定采集帧率', `${camera.fps} FPS`], ['深度单位', `${camera.depth_scale.toPrecision(7)} m / 原始值`], ['深度对齐', 'Depth → Color / rs.align'], ['检测模型', `${state.model_name} · COCO person (class 0)`], ['推理设备', state.device], ['RGB 内参', JSON.stringify(camera.intrinsics, null, 2)]] : [['设备状态', '尚未读取到 RealSense 设备信息']];
  $('device-details').innerHTML = fields.map(([key, value]) => `<dt>${escapeHTML(key)}</dt><dd>${escapeHTML(value)}</dd>`).join('');
}

function drawBoxes(context, meta, isDepth = false) {
  if (!$('show-boxes').checked) return;
  context.lineWidth = 2;
  context.font = '14px "Segoe UI", "Microsoft YaHei", sans-serif';
  context.textBaseline = 'top';
  for (const d of meta.detections || []) {
    const [x1, y1, x2, y2] = d.box;
    const color = d.depth_status === 'valid' ? '#65e3a3' : '#f0cf78';
    context.strokeStyle = color;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    if (isDepth) {
      const [sx, sy, ex, ey] = d.sample_box;
      context.save(); context.setLineDash([4, 4]); context.lineWidth = 1;
      context.strokeRect(sx, sy, ex - sx, ey - sy); context.restore();
    }
    const label = `人员 #${d.id}  ${Math.round(d.confidence * 100)}%  ${d.distance_m === null ? '深度不足' : `${d.depth_status === 'unstable' ? '≈' : ''}${d.distance_m.toFixed(2)}m`}`;
    const width = context.measureText(label).width + 14;
    const x = Math.max(0, Math.min(x1, 640 - width));
    const y = y1 >= 27 ? y1 - 27 : y1 + 2;
    context.fillStyle = '#10271fe6'; context.fillRect(x, y, width, 25);
    context.fillStyle = color; context.fillText(label, x + 7, y + 5);
  }
}

function drawCurrentFrame() {
  if (!lastImageRGB || !lastImageDepth || !lastFrameMeta) return;
  ctxRGB.drawImage(lastImageRGB, 0, 0);
  ctxDepth.drawImage(lastImageDepth, 0, 0);
  drawBoxes(ctxRGB, lastFrameMeta);
  drawBoxes(ctxDepth, lastFrameMeta, true);
  if (probe && probe.until > performance.now()) {
    ctxDepth.strokeStyle = '#a0ffb9'; ctxDepth.lineWidth = 1;
    ctxDepth.beginPath(); ctxDepth.moveTo(probe.x - 8, probe.y); ctxDepth.lineTo(probe.x + 8, probe.y);
    ctxDepth.moveTo(probe.x, probe.y - 8); ctxDepth.lineTo(probe.x, probe.y + 8); ctxDepth.stroke();
  } else { $('depth-probe').hidden = true; probe = null; }
}

async function consumeFrames() {
  if (decoding) return;
  decoding = true;
  try {
    while (pendingFrame) {
      const buffer = pendingFrame;
      pendingFrame = null;
      const generation = renderGeneration;
      const headerLength = new DataView(buffer).getUint32(0, true);
      const start = 4 + headerLength;
      const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
      if (start + meta.rgb_bytes + meta.depth_bytes !== buffer.byteLength) throw new Error('实时帧长度不匹配');
      const [rgb, depth] = await Promise.all([
        createImageBitmap(new Blob([buffer.slice(start, start + meta.rgb_bytes)], { type: 'image/jpeg' })),
        createImageBitmap(new Blob([buffer.slice(start + meta.rgb_bytes)], { type: 'image/png' }))
      ]);
      if (generation !== renderGeneration || meta.generation < knownGeneration || !online) { rgb.close(); depth.close(); continue; }
      lastImageRGB?.close(); lastImageDepth?.close();
      lastImageRGB = rgb; lastImageDepth = depth; lastFrameMeta = meta;
      displayedSequence = meta.sequence;
      lastFrameAt = performance.now();
      drawCurrentFrame();
      $('rgb-empty').hidden = $('depth-empty').hidden = true;
      document.querySelectorAll('.live-badge').forEach(el => { el.hidden = false; });
      $('snapshot-button').disabled = snapshotBusy;
      if (performance.now() - lastStatsAt > 250) { renderStatus(meta, true); lastStatsAt = performance.now(); }
    }
  } catch (error) {
    console.error('Frame decode failed', error);
    toast('画面解码异常，正在等待下一帧');
  } finally { decoding = false; }
}

function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`);
  socket.binaryType = 'arraybuffer';
  socket.onopen = () => {
    // A restarted service begins a new generation counter and default settings.
    knownGeneration = -1;
    initialized = false;
    notificationsInitialized = false;
    previousNotificationCount = previousNotificationError = null;
    previousCamera = previousModel = '';
    previousPeople = null;
    setConnection(true);
    lastMessageAt = performance.now();
    addEvent('连接', '已连接本地实时服务');
  };
  socket.onmessage = event => {
    lastMessageAt = performance.now();
    if (typeof event.data === 'string') {
      try { renderStatus(JSON.parse(event.data)); } catch (error) { console.error(error); }
    } else { pendingFrame = event.data; consumeFrames(); }
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    if (online) addEvent('连接', '实时服务连接断开，自动重连中');
    setConnection(false);
    clearFrames('实时连接已断开', '请保持本机服务运行，正在自动重连');
    $('metric-camera').innerHTML = '连接断开<span class="status-pulse"></span>';
    $('metric-camera').classList.remove('good');
    $('side-dot').className = $('stream-dot').className = 'dot bad';
    $('side-state').textContent = $('stream-text').textContent = '本地服务重连中';
    $('alert').hidden = false;
    $('alert').textContent = '本地服务未连接。请运行 start.bat 或 python app.py；网页将自动重连。';
    reconnectTimer = setTimeout(connect, 2000);
  };
}

setInterval(() => {
  const now = performance.now();
  if (online && now - lastMessageAt > 5000) socket.close();
  if (lastFrameAt && now - lastFrameAt > 3000) {
    clearFrames('画面已超时', '等待新的处理帧，请检查设备与模型状态');
    $('stream-dot').className = 'dot waiting';
    $('stream-text').textContent = '处理帧超时 · 等待恢复';
  }
  if (probe && now > probe.until) { probe = null; $('depth-probe').hidden = true; drawCurrentFrame(); }
}, 1000);

setConnection(false);
renderEvents();
switchView(location.hash.slice(1) || 'console');
connect();
