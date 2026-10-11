const $ = id => document.getElementById(id);
const state = {ready: false, phase: 'idle', mode: 'filtered', groups: [], group: null,
  selected: new Set(), device: null, records: [], metrics: new Map(), sources: new Set(),
  playbackNext: 0, gapMs: 0, limiterEvents: 0, generation: 0, expectedBlocks: null,
  receivedBlocks: 0, demoFinished: false, pendingEvents: new Map(), volume: .5};

function notice(message, error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
}
function selectedNames() {
  return state.group?.profiles.filter(p => state.selected.has(p.id)).map(p => p.name) || [];
}
function controls() {
  const idle = state.phase === 'idle', ready = state.ready && idle;
  $('calibrateBtn').disabled = !ready || $('source').value !== 'microphone';
  $('exampleBtn').disabled = !ready || !state.exampleAvailable;
  $('uploadInput').disabled = !ready;
  $('startBtn').disabled = !ready || (state.mode === 'filtered' && !state.selected.size);
  $('stopBtn').disabled = idle;
  $('permissionBtn').disabled = !idle;
  $('profileGroup').disabled = !idle;
  $('deleteBtn').disabled = !ready || !state.group;
  $('source').disabled = !idle;
  $('inputDevice').disabled = !idle;
  $('hopMs').disabled = !idle;
  $('exportBtn').disabled = !state.records.length;
  $('micControls').hidden = $('source').value !== 'microphone';
  $('filteredMode').classList.toggle('active', state.mode === 'filtered');
  $('bypassMode').classList.toggle('active', state.mode === 'bypass');
  $('selectionText').textContent = state.mode === 'bypass' ? 'Suara asli diteruskan tanpa pemisahan model.' :
    state.selected.size ? `${selectedNames().join(' + ')} dipilih untuk didengar.` : 'Pilih speaker, lalu mulai filtering.';
  document.querySelectorAll('.step').forEach((el, i) => el.classList.toggle('active',
    i === (state.phase === 'listening' ? 2 : state.group ? 1 : 0)));
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
async function pollHealth() {
  try {
    const info = await api('/api/health');
    state.ready = info.status === 'ready' && !info.profile_busy;
    state.device = info.device;
    state.exampleAvailable = info.example_available;
    const label = info.profile_busy ? 'Membuat profil…' : info.status === 'ready' ? `Model siap · ${info.device.toUpperCase()}` : info.status === 'error' ? 'Model belum tersedia' : 'Memuat & memanaskan model…';
    $('modelBadge').innerHTML = '<i></i>';
    $('modelBadge').append(document.createTextNode(label));
    $('modelBadge').className = `pill ${info.status}`;
    $('computeLabel').textContent = info.device === 'mps' ? 'SepFormer GPU Metal + ECAPA hybrid' : 'CPU · mode debugging';
    if (info.status === 'error') notice(info.error, true);
    else if (!state.healthSeen && state.ready) {
      state.healthSeen = true;
      notice('Model siap. Mulai dengan Bypass untuk memeriksa rute, atau gunakan profil uji + demo Mix.wav.');
      $('calibrationState').textContent = 'Pastikan semua peserta berbicara';
    }
    controls();
  } catch {
    state.ready = false;
    $('modelBadge').textContent = 'Server tidak terhubung';
    controls();
  }
}
async function loadGroups(preferredId) {
  const {groups} = await api('/api/profiles');
  state.groups = groups;
  $('profileGroup').replaceChildren();
  if (!groups.length) $('profileGroup').add(new Option('Belum ada profil', ''));
  for (const g of groups) $('profileGroup').add(new Option(
    `${g.id === 'example10' ? 'Uji Mix.wav' : 'Kalibrasi'} · ${g.calibration_seconds} detik · ${g.profiles.map(p => p.name).join(', ')}`, g.id));
  const group = groups.find(g => g.id === preferredId) || groups[0] || null;
  state.group = group;
  state.selected = new Set(group ? [group.profiles[0].id] : []);
  if (group) $('profileGroup').value = group.id;
  renderSpeakers();
}
function renderSpeakers() {
  const target = $('speakerCards');
  target.replaceChildren();
  if (!state.group) {
    target.innerHTML = '<div class="empty-state"><span>♫</span><p>Profil suara akan muncul di sini.</p><small>Rekam kalibrasi atau gunakan profil uji sebelumnya.</small></div>';
    $('profileInfo').textContent = '';
    controls(); return;
  }
  for (const [index, p] of state.group.profiles.entries()) {
    const card = document.createElement('article');
    card.className = `speaker-card ${state.selected.has(p.id) ? 'selected' : ''}`;
    const top = document.createElement('div'); top.className = 'speaker-top';
    const avatar = document.createElement('span'); avatar.className = 'speaker-avatar'; avatar.textContent = `0${index + 1}`;
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.className = 'speaker-check';
    checkbox.checked = state.selected.has(p.id); checkbox.setAttribute('aria-label', `Dengarkan ${p.name}`);
    checkbox.addEventListener('change', async () => {
      if (checkbox.checked) state.selected.add(p.id); else state.selected.delete(p.id);
      if (state.phase !== 'idle') await stop('Pilihan berubah. Mulai lagi untuk menggunakan speaker yang baru dipilih.');
      renderSpeakers();
    });
    top.append(avatar, checkbox);
    const name = document.createElement('input'); name.className = 'speaker-name'; name.value = p.name; name.maxLength = 50;
    name.setAttribute('aria-label', `Nama profil ${index + 1}`);
    name.addEventListener('change', async () => {
      try {
        const group = await api(`/api/profiles/${state.group.id}`, {method: 'PATCH',
          headers: {'Content-Type': 'application/json'}, body: JSON.stringify({speaker_id: p.id, name: name.value})});
        state.group = group; renderSpeakers();
      } catch (error) {name.value = p.name; notice(error.message, true);}
    });
    const info = document.createElement('small'); info.textContent = `${p.id} · dengarkan untuk mengenali orangnya`;
    const audio = document.createElement('audio'); audio.controls = true; audio.preload = 'none'; audio.src = p.preview_url; audio.volume = state.volume;
    audio.addEventListener('play', () => document.querySelectorAll('audio').forEach(other => {if (other !== audio) other.pause();}));
    if ($('outputDevice').value && audio.setSinkId) audio.setSinkId($('outputDevice').value).catch(error => notice(error.message, true));
    card.append(top, name, info, audio); target.append(card);
  }
  $('profileInfo').textContent = `Kalibrasi ${state.group.calibration_seconds} detik · pembuatan profil ${state.group.creation_seconds.toFixed(2)} detik. ` +
    (state.group.possible_duplicate_pairs.length ? 'Ada kandidat yang mirip; periksa suara duplikat.' : 'Tiga kandidat belum membuktikan jumlah orang. Periksa suara bercampur dan identitas bertukar.');
  controls();
}
async function enumerateDevices() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const currentInput = $('inputDevice').value, currentOutput = $('outputDevice').value;
  $('inputDevice').replaceChildren(new Option('Mikrofon default', ''));
  $('outputDevice').replaceChildren(new Option('Output default macOS', ''));
  for (const d of devices) {
    if (d.kind === 'audioinput') $('inputDevice').add(new Option(d.label || `Mikrofon ${$('inputDevice').length}`, d.deviceId));
    if (d.kind === 'audiooutput') $('outputDevice').add(new Option(d.label || `Output ${$('outputDevice').length}`, d.deviceId));
  }
  if ([...$('inputDevice').options].some(o => o.value === currentInput)) $('inputDevice').value = currentInput;
  if ([...$('outputDevice').options].some(o => o.value === currentOutput)) $('outputDevice').value = currentOutput;
  $('outputDevice').disabled = !('setSinkId' in AudioContext.prototype);
  $('outputHint').textContent = $('outputDevice').disabled ?
    'Browser ini memakai output default. Pilih AirPods Max di System Settings → Sound → Output.' :
    'Output bisa dipilih di browser. Untuk uji awal, gunakan AirPods Max melalui USB-C.';
}
const constraints = () => ({audio: {channelCount: 1, echoCancellation: false, noiseSuppression: false,
  autoGainControl: false, ...($('inputDevice').value ? {deviceId: {exact: $('inputDevice').value}} : {})}});
function routeSettings(track) {
  const settings = track.getSettings();
  $('activeInput').textContent = track.label || 'Mikrofon default';
  $('inputSettings').textContent = `Track ${settings.sampleRate || '?'} Hz · echo ${String(settings.echoCancellation ?? '?')} · noise suppression ${String(settings.noiseSuppression ?? '?')} · auto gain ${String(settings.autoGainControl ?? '?')}`;
  state.inputSettings = settings;
  state.inputLabel = track.label;
}
async function connectSocket() {
  state.ws = new WebSocket(`ws://${location.host}/ws`);
  state.ws.binaryType = 'arraybuffer';
  const ws = state.ws;
  ws.onmessage = ({data}) => {
    if (typeof data !== 'string') {scheduleAudio(data); return;}
    const event = JSON.parse(data);
    const waiter = state.pendingEvents.get(event.event);
    if (waiter) {state.pendingEvents.delete(event.event); waiter.resolve(event);}
    if (event.event === 'error') {
      for (const pending of state.pendingEvents.values()) pending.reject(new Error(event.message));
      state.pendingEvents.clear(); stop(event.message, true);
    } else if (event.event === 'calibration_progress') {
      $('countdown').innerHTML = `${Math.max(0, Math.ceil(30 - event.seconds))}<span>detik</span>`;
      $('calibrationProgress').style.width = `${event.seconds / 30 * 100}%`;
      $('calibrationState').textContent = `${event.seconds.toFixed(1)} / 30 detik terekam`;
    } else if (event.event === 'processing_profiles') {
      releaseInput(); state.phase = 'processing'; controls();
      $('calibrationState').textContent = 'Membuat tiga kandidat profil…';
      notice('30 detik selesai. Mikrofon sudah dihentikan; model sedang membuat profil.');
    } else if (event.event === 'profiles_ready') {
      const id = event.group.id;
      stop('Profil siap. Dengarkan cuplikan, beri nama, dan pilih speaker.').then(() => loadGroups(id));
      $('calibrationState').textContent = `Profil dibuat dalam ${event.group.creation_seconds.toFixed(2)} detik`;
    } else if (event.event === 'metrics') {
      state.metrics.set(event.sequence, event);
      $('processMetric').innerHTML = `${Math.round(event.processing_p95_ms)}<small>ms</small>`;
      $('queueMetric').innerHTML = `${event.pending}<small>blok</small>`;
      if (event.limiter_gain < 1) state.limiterEvents++;
    }
  };
  ws.onclose = () => {if (state.ws === ws && state.phase !== 'idle') stop('Koneksi audio terputus. Mulai ulang sesi.', true);};
  await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = () => reject(new Error('Tidak dapat membuka koneksi audio lokal'));});
}
function waitEvent(event) {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {state.pendingEvents.delete(event); reject(new Error('Server tidak merespons; coba ulang.'));}, 15000);
    state.pendingEvents.set(event, {resolve: value => {clearTimeout(timeout); resolve(value);}, reject: error => {clearTimeout(timeout); reject(error);}});
  });
}
async function command(action, data = {}) {
  const event = action === 'start' ? 'started' : action === 'calibrate' ? 'calibrating' : 'stopped';
  const ready = waitEvent(event);
  state.ws.send(JSON.stringify({action, ...data}));
  return ready;
}
async function prepareAudio(useDemo) {
  state.ctx = new AudioContext({latencyHint: 'interactive', sampleRate: 48000});
  const ctx = state.ctx;
  await ctx.resume();
  if ($('outputDevice').value && ctx.setSinkId) await ctx.setSinkId($('outputDevice').value);
  await ctx.audioWorklet.addModule('/static/capture-worklet.js');
  state.capture = new AudioWorkletNode(ctx, 'tse-capture');
  state.master = ctx.createGain(); state.master.gain.value = state.volume; state.master.connect(ctx.destination);
  const silent = ctx.createGain(); silent.gain.value = 0; state.capture.connect(silent).connect(ctx.destination);
  if (useDemo) {
    const response = await fetch('/api/demo.wav');
    if (!response.ok) throw new Error('Mix.wav tidak tersedia');
    const buffer = await ctx.decodeAudioData(await response.arrayBuffer());
    const hopSeconds = state.runHopMs / 1000;
    const duration = Math.floor((buffer.duration - 10) / hopSeconds) * hopSeconds;
    if (duration < .2) throw new Error('Demo tidak punya audio lanjutan yang cukup');
    state.demoSource = ctx.createBufferSource(); state.demoSource.buffer = buffer; state.demoSource.connect(state.capture);
    state.demoDuration = duration;
    $('activeInput').textContent = 'Mix.wav · bagian setelah kalibrasi 10 detik';
    $('inputSettings').textContent = `Demo ${duration.toFixed(1)} detik (hop penuh) · PCM melalui AudioWorklet · ${ctx.sampleRate} Hz`;
    state.inputLabel = 'Mix.wav demo'; state.inputSettings = {sampleRate: ctx.sampleRate, demoOffsetSeconds: 10, demoSeconds: duration};
  } else {
    state.stream = await navigator.mediaDevices.getUserMedia(constraints());
    const track = state.stream.getAudioTracks()[0]; routeSettings(track);
    track.onended = () => {if (state.phase !== 'idle') stop('Mikrofon berakhir atau rute berubah. Pilih perangkat dan mulai lagi.', true);};
    state.micSource = ctx.createMediaStreamSource(state.stream); state.micSource.connect(state.capture);
  }
  ctx.onstatechange = () => {
    if (state.ctx === ctx && state.phase === 'listening' && ctx.state !== 'running') stop('Audio browser berhenti. Kembali ke tab dan mulai lagi.', true);
  };
  state.capture.port.onmessage = ({data}) => {
    if (data.done) {
      if (state.phase === 'calibrating') releaseInput();
      else {state.demoFinished = true; state.expectedBlocks = data.blocks; finishDemoIfDrained();}
      return;
    }
    if (!['calibrating', 'listening'].includes(state.phase) || state.ws?.readyState !== WebSocket.OPEN) return;
    if (state.ws.bufferedAmount > data.samples.byteLength * 3) {stop('Transport audio tertinggal; sesi dihentikan.', true); return;}
    let squares = 0; for (const x of data.samples) squares += x * x;
    const rms = Math.sqrt(squares / data.samples.length), db = 20 * Math.log10(Math.max(rms, 1e-6));
    $('inputLevel').style.width = `${Math.max(0, Math.min(100, (db + 60) / 60 * 100))}%`;
    $('inputDb').textContent = `${Math.round(db)} dB`;
    const packet = new ArrayBuffer(16 + data.samples.byteLength), header = new DataView(packet);
    header.setUint32(0, data.sequence, true); header.setUint32(4, data.samples.length, true);
    header.setFloat64(8, performance.now() - (ctx.currentTime - data.startTime) * 1000, true);
    new Float32Array(packet, 16).set(data.samples); state.ws.send(packet);
  };
}
function scheduleAudio(packet) {
  if (state.phase !== 'listening' || !state.ctx) return;
  const header = new DataView(packet), seq = header.getUint32(0, true), size = header.getUint32(4, true), capturedMs = header.getFloat64(8, true);
  if (size !== state.runHopMs * 16 || packet.byteLength !== 16 + size * 4 || seq !== state.receivedBlocks) {stop('Urutan hasil audio tidak valid.', true); return;}
  const ctx = state.ctx, earliest = ctx.currentTime + .02;
  const gap = state.playbackNext ? Math.max(0, earliest - state.playbackNext) * 1000 : 0;
  const startAt = Math.max(earliest, state.playbackNext);
  const estimatedDelay = performance.now() + (startAt - ctx.currentTime) * 1000 - capturedMs;
  if (startAt - ctx.currentTime > .6 || estimatedDelay > 1500) {stop('Playback tertinggal. Sesi dihentikan agar audio lama tidak menumpuk.', true); return;}
  const buffer = ctx.createBuffer(1, size, 16000); buffer.copyToChannel(new Float32Array(packet, 16), 0);
  const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(state.master);
  state.sources.add(source); source.onended = () => state.sources.delete(source); source.start(startAt);
  state.playbackNext = startAt + size / 16000; state.gapMs += gap; state.receivedBlocks++;
  const metrics = state.metrics.get(seq) || {}; state.metrics.delete(seq);
  state.records.push({sequence: seq, estimated_application_delay_ms: estimatedDelay, playback_gap_ms: gap,
    browser_playback_queue_ms: (startAt - ctx.currentTime) * 1000, ...metrics});
  if (state.records.length > 9000) state.records.shift();
  const delays = state.records.slice(-300).map(r => r.estimated_application_delay_ms).sort((a,b) => a-b);
  $('delayMetric').innerHTML = `${Math.round(delays[Math.floor((delays.length - 1) * .95)])}<small>ms</small>`;
  $('gapMetric').innerHTML = `${Math.round(state.gapMs)}<small>ms</small>`;
  $('playbackBuffer').textContent = `Buffer browser: ${Math.round((state.playbackNext - ctx.currentTime) * 1000)} ms`;
  $('exportBtn').disabled = false;
  finishDemoIfDrained();
}
function finishDemoIfDrained() {
  if (!state.demoFinished || state.receivedBlocks !== state.expectedBlocks) return;
  const generation = state.generation;
  setTimeout(() => {if (generation === state.generation && state.phase === 'listening') stop('Demo selesai. Unduh laporan atau coba mikrofon langsung.');},
    Math.max(0, (state.playbackNext - state.ctx.currentTime) * 1000 + 40));
}
function releaseInput() {
  if (state.stream) {for (const track of state.stream.getTracks()) {track.onended = null; track.stop();} state.stream = null;}
  if (state.micSource) {state.micSource.disconnect(); state.micSource = null;}
}
async function stop(message = 'Sesi berhenti. Mikrofon dilepas dan playback dihentikan.', error = false) {
  state.stopReason = message;
  state.generation++; state.phase = 'idle';
  const ws = state.ws; state.ws = null;
  if (ws?.readyState === WebSocket.OPEN) {ws.send(JSON.stringify({action:'stop'})); ws.close();}
  for (const pending of state.pendingEvents.values()) pending.reject(new Error('Sesi dihentikan'));
  state.pendingEvents.clear();
  state.capture?.port.postMessage({action: 'active', value: false});
  releaseInput();
  if (state.demoSource) {try {state.demoSource.stop();} catch {} state.demoSource = null;}
  for (const source of state.sources) {try {source.stop();} catch {}}
  state.sources.clear();
  const ctx = state.ctx; state.ctx = null;
  if (ctx) {ctx.onstatechange = null; await ctx.close();}
  $('listenTitle').textContent = 'Siap mendengarkan'; $('inputLevel').style.width = '0%';
  $('queueMetric').innerHTML = '0<small>blok</small>';
  $('playbackBuffer').textContent = 'Buffer browser: 0 ms · sesi berhenti';
  controls(); notice(message, error);
}
async function begin(calibration = false) {
  if (state.phase !== 'idle' || !state.ready) return;
  if (!calibration && state.mode === 'filtered' && !state.selected.size) {notice('Pilih minimal satu speaker.', true); return;}
  state.phase = 'preparing'; controls();
  state.runHopMs = calibration ? 200 : state.mode === 'bypass' ? 20 : Number($('hopMs').value);
  document.querySelectorAll('audio').forEach(audio => audio.pause());
  try {
    const useDemo = !calibration && $('source').value === 'demo';
    await prepareAudio(useDemo); await connectSocket();
    state.records = []; state.metrics.clear(); state.playbackNext = 0; state.gapMs = 0;
    state.receivedBlocks = 0; state.demoFinished = false; state.expectedBlocks = null; state.limiterEvents = 0;
    state.stopReason = null;
    state.reportInfo = {mode: calibration ? 'calibration' : state.mode, selected: [...state.selected], profile_group: state.group?.id,
      started_at: new Date().toISOString(), browser: navigator.userAgent, source: useDemo ? 'demo' : 'microphone',
      device: state.device, context_ms: 1500, hop_ms: state.runHopMs, input_settings: state.inputSettings,
      input_label: state.inputLabel, audio_context_rate_hz: state.ctx.sampleRate,
      output_selection: $('outputDevice').selectedOptions[0]?.textContent,
      audio_context_base_latency_seconds: state.ctx.baseLatency ?? null,
      audio_context_output_latency_seconds: state.ctx.outputLatency ?? null,
      metric_scope: 'browser capture graph to scheduled playback; not acoustic microphone-to-ear latency'};
    await command(calibration ? 'calibrate' : 'start', {rate: state.ctx.sampleRate, mode: state.mode,
      group_id: state.group?.id, selected: [...state.selected], hop_ms: Number($('hopMs').value)});
    state.phase = calibration ? 'calibrating' : 'listening';
    const startAt = state.ctx.currentTime + .05;
    state.capture.port.postMessage({action: 'active', value: true, startAt, hopMs: state.runHopMs,
      limitFrames: calibration ? state.ctx.sampleRate * 30 : useDemo ? Math.round(state.demoDuration * state.ctx.sampleRate) : 0});
    if (useDemo) state.demoSource.start(startAt, 10, state.demoDuration);
    $('configFooter').textContent = `SepFormer + ECAPA · Konteks 1.500 ms · Hop ${state.runHopMs} ms`;
    $('listenTitle').textContent = calibration ? 'Kalibrasi berlangsung' : useDemo ? 'Demo sedang diproses' : 'Mendengarkan langsung';
    notice(calibration ? 'Rekam 30 detik percakapan. Pastikan setiap speaker berbicara.' :
      useDemo ? 'Demo diproses per hop melalui pipeline live; tidak memutar hasil WAV yang sudah dibuat.' : 'Mikrofon aktif. Hasil diproses bertahap dan diputar ke output yang dipilih.');
    controls();
  } catch (error) {await stop(error.name === 'NotAllowedError' ? 'Izin mikrofon ditolak. Izinkan localhost di pengaturan browser, lalu coba lagi.' : error.message, true);}
}

$('permissionBtn').addEventListener('click', async () => {
  let stream;
  try {stream = await navigator.mediaDevices.getUserMedia(constraints()); routeSettings(stream.getAudioTracks()[0]); await enumerateDevices(); notice('Perangkat ditemukan. Mic belum merekam; pilih rute lalu mulai.');}
  catch (error) {notice(`Tidak dapat mengakses mikrofon: ${error.message}`, true);}
  finally {stream?.getTracks().forEach(track => track.stop());}
});
$('source').addEventListener('change', controls);
$('outputDevice').addEventListener('change', async () => {
  try {
    if (state.ctx?.setSinkId) await state.ctx.setSinkId($('outputDevice').value);
    for (const audio of document.querySelectorAll('audio')) if (audio.setSinkId) await audio.setSinkId($('outputDevice').value);
  } catch (error) {await stop(`Output tidak tersedia: ${error.message}`, true);}
});
$('calibrateBtn').addEventListener('click', () => begin(true));
$('startBtn').addEventListener('click', () => begin());
$('stopBtn').addEventListener('click', () => stop());
for (const id of ['filteredMode', 'bypassMode']) $(id).addEventListener('click', async () => {
  if (state.phase !== 'idle') await stop('Mode berubah. Tekan Mulai untuk memakai mode baru.');
  state.mode = $(id).dataset.mode; controls();
});
$('volume').addEventListener('input', () => {
  state.volume = Number($('volume').value); $('volumeText').textContent = `${Math.round(state.volume * 100)}%`;
  if (state.master && state.ctx) state.master.gain.setTargetAtTime(state.volume, state.ctx.currentTime, .02);
  document.querySelectorAll('audio').forEach(audio => audio.volume = state.volume);
});
$('exampleBtn').addEventListener('click', async () => {
  try {const group = await api('/api/example-profile', {method:'POST'}); await loadGroups(group.id); notice('Profil uji berasal dari 10 detik pertama Mix.wav. Pilih Demo Mix.wav untuk menguji bagian lanjutannya.');}
  catch (error) {notice(error.message, true);}
});
$('profileGroup').addEventListener('change', () => {
  state.group = state.groups.find(g => g.id === $('profileGroup').value) || null;
  state.selected = new Set(state.group ? [state.group.profiles[0].id] : []); renderSpeakers();
});
$('deleteBtn').addEventListener('click', async () => {
  try {await api(`/api/profiles/${state.group.id}`, {method:'DELETE'}); await loadGroups(); notice('Profil website dihapus. Rekaman asli dan hasil eksperimen sebelumnya tetap utuh.');}
  catch (error) {notice(error.message, true);}
});
$('uploadInput').addEventListener('change', async () => {
  const file = $('uploadInput').files[0]; if (!file) return;
  if (file.size > 30 * 1024 * 1024) {notice('WAV maksimal 30 MiB.', true); return;}
  state.phase = 'processing'; controls(); notice('Membuat profil dari 30 detik pertama WAV…');
  try {const group = await api('/api/calibration-upload', {method:'POST', body:file}); await loadGroups(group.id); notice('Profil siap. Dengarkan dan pilih suara.');}
  catch (error) {notice(error.message, true);}
  finally {state.phase = 'idle'; $('uploadInput').value = ''; controls();}
});
$('exportBtn').addEventListener('click', () => {
  const report = {...state.reportInfo, ended_or_exported_at: new Date().toISOString(), samples_kept: state.records.length,
    stop_reason: state.stopReason,
    playback_gap_sum_ms: state.gapMs, limiter_events: state.limiterEvents,
    live_audio_saved: false, records: state.records};
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `suara-session-${Date.now()}.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
window.addEventListener('pagehide', () => {releaseInput(); state.ws?.close();});
navigator.mediaDevices?.addEventListener('devicechange', async () => {
  if (state.phase !== 'idle') await stop('Perangkat audio berubah. Periksa rute lalu mulai lagi.', true);
  await enumerateDevices();
});
await loadGroups().catch(error => notice(error.message, true));
await pollHealth();
await enumerateDevices().catch(() => {});
setInterval(pollHealth, 2500);
