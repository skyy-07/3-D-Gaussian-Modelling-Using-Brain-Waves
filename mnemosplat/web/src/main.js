import './style.css';
import { SplatStage } from './stage.js';

/* ------------------------------------------------------------------ helpers */
const $ = (sel) => document.querySelector(sel);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = (value) =>
  String(value ?? '').replace(/[&<>"']/g, (ch) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
const int = new Intl.NumberFormat('en-US');
const asset = (rel) => (String(rel).startsWith('/') ? String(rel) : `/assets/${String(rel).replace(/^\/+/, '')}`);
const resolve = (baseDir, rel) => {
  if (!rel) return null;
  if (/^(https?:)?\/\//.test(rel) || rel.startsWith('/')) return rel;
  return `${baseDir}${rel}`;
};

async function fetchJSON(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${url}`);
  return res.json();
}

/* -------------------------------------------------------------------- state */
const state = {
  scenarios: [],
  key: null,
  manifest: null,
  manifestUrl: null,
  baseDir: null,
  frame: 0,
  playing: false,
  firstReady: false,
  booted: false,
  xr: false,
  seenFrames: new Set(),
  uploadFile: null,
  uploading: false,
  eegMode: 'condition',
  health: null,
};

let stage = null;
let playTimer = null;
let toastTimer = null;
let narrativeTimer = null;
let prefetching = false;

const dom = {
  app: $('#app'),
  stage: $('#stage'),
  boot: $('#boot'),
  bootFill: $('#boot-fill'),
  bootStatus: $('#boot-status'),
  busy: $('#busy'),
  toast: $('#toast'),
  mode: $('#mode'),
  aboutBtn: $('#about-btn'),
  modal: $('#modal'),
  modalProv: $('#modal-provenance'),
  eegBtn: $('#eeg-upload-btn'),
  eegModal: $('#eeg-modal'),
  eegDrop: $('#eeg-drop'),
  eegFile: $('#eeg-file'),
  eegDropTitle: $('#eeg-drop-title'),
  eegDropHint: $('#eeg-drop-hint'),
  eegSfreq: $('#eeg-sfreq'),
  eegSubmit: $('#eeg-submit'),
  eegStatus: $('#eeg-status'),
  eegProgress: $('#eeg-progress'),
  eegProgressFill: $('#eeg-progress-fill'),
  eegModes: $('#eeg-modes'),
  eegModeHelp: $('#eeg-mode-help'),
  reconModal: $('#recon-modal'),
  reconGrid: $('#recon-grid'),
  reconMeta: $('#recon-meta'),
  lightbox: $('#lightbox'),
  lightboxImg: $('#lightbox-img'),
  lightboxCap: $('#lightbox-cap'),
  salience: $('#salience'),
  salienceDial: $('#salience-dial'),
  spark: $('#spark'),
  bands: $('#bands'),
  metrics: $('#metrics'),
  objects: $('#objects'),
  keyframeImg: $('#keyframe-img'),
  attentionImg: $('#attention-img'),
  evKeyframe: $('#ev-keyframe'),
  evAttention: $('#ev-attention'),
  beat: $('#beat'),
  conditionChip: $('#condition-chip'),
  narrative: $('#narrative'),
  provenance: $('#provenance'),
  play: $('#play'),
  scrub: $('#scrub'),
  sliderFill: $('#slider-fill'),
  ticks: $('#ticks'),
  epochCount: $('#epoch-count'),
  splatCount: $('#splat-count'),
  xrOverlay: $('#xr-overlay'),
  xrPrev: $('#xr-prev'),
  xrNext: $('#xr-next'),
  xrEpoch: $('#xr-epoch'),
  xrSalience: $('#xr-salience'),
  xrCondition: $('#xr-condition'),
};

/* -------------------------------------------------------------------- toasts */
function toast(message, ms = 5000) {
  dom.toast.textContent = message;
  dom.toast.classList.add('is-visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => dom.toast.classList.remove('is-visible'), ms);
}

function setBusy(on) {
  dom.busy.classList.toggle('is-busy', Boolean(on && state.firstReady));
}

function failBoot(message) {
  dom.bootStatus.classList.add('is-error');
  dom.bootStatus.textContent = message;
  dom.bootFill.style.width = '0%';
}

function finishBoot() {
  dom.bootFill.style.width = '100%';
  dom.bootStatus.textContent = 'memory volume ready';
  setTimeout(() => {
    dom.boot.classList.add('is-hidden');
    state.booted = true;
    setBusy(false);
  }, 420);
}

/* --------------------------------------------------------------- boot / data */
async function loadIndex() {
  try {
    return await fetchJSON('/assets/index.json');
  } catch (error) {
    // A single-scenario run (`mnemosplat demo --scenario focus`) has no index.json.
    const scenarios = [];
    for (const key of ['focus', 'distracted', 'street']) {
      try {
        const manifest = await fetchJSON(asset(`${key}/manifest.json`));
        scenarios.push({
          key,
          label: manifest.scenario === 'distracted' ? 'Diffuse recall'
            : manifest.scenario === 'street' ? 'Blue hour' : 'Vivid recall',
          caption: manifest.scene_graph?.title || '',
          manifest: `${key}/manifest.json`,
          salience_mean: manifest.stats?.salience_mean,
        });
      } catch {
        /* not generated */
      }
    }
    if (!scenarios.length) throw error;
    return { scenarios, note: 'index rebuilt from manifests' };
  }
}

function pickInitialKey(index) {
  const fromQuery = new URLSearchParams(location.search).get('scene');
  if (fromQuery && index.scenarios.some((s) => s.key === fromQuery)) return fromQuery;
  const preferred = index.scenarios.find((s) => s.key === 'focus');
  return (preferred || index.scenarios[0]).key;
}

async function loadScenario(key, { keepFrame = true } = {}) {
  const entry = state.scenarios.find((s) => s.key === key);
  if (!entry) throw new Error(`Unknown condition: ${key}`);
  const url = asset(entry.manifest);
  const manifest = await fetchJSON(url);
  if (!manifest.frames?.length) throw new Error(`Episode ${key} has no frames`);

  state.key = key;
  state.manifest = manifest;
  state.manifestUrl = url;
  state.baseDir = url.slice(0, url.lastIndexOf('/') + 1);
  state.frame = clamp(keepFrame ? state.frame : 0, 0, manifest.frames.length - 1);

  // Frame 0 carries the lift camera. On the first scenario only, map that camera to
  // the viewer origin; condition switches must not steal the user's viewpoint.
  const camera = manifest.frames[0].camera?.position;
  if (!stage.framedOnce && camera) {
    stage.setFraming(camera);
    stage.framedOnce = true;
  }

  renderModeToggle();
  renderAll();
  await showFrame();
  schedulePrefetch();
}

async function boot() {
  try {
    stage = new SplatStage({
      container: dom.stage,
      xrOverlay: dom.xrOverlay,
      onReady: handleReady,
      onProgress: handleProgress,
      onError: handleStageError,
      onXRChange: handleXR,
      onTrigger: handleTrigger,
    });
    stage.init();
  } catch (error) {
    failBoot(`WebGL / WebXR could not start in this browser. ${error?.message || error}`);
    return;
  }

  let index;
  try {
    index = await loadIndex();
    state.scenarios = index.scenarios || [];
    if (!state.scenarios.length) throw new Error('No scenarios in index.json');
  } catch (error) {
    failBoot(
      'No episode found under /assets. Generate one first:\n'
      + 'mnemosplat demo --scenario both --out out/demo --force\n'
      + 'mnemosplat serve\n\n'
      + `(${error?.message || error})`,
    );
    return;
  }

  const key = pickInitialKey(index);
  const wantedFrame = Number(new URLSearchParams(location.search).get('frame'));
  if (Number.isFinite(wantedFrame) && wantedFrame > 0) state.frame = wantedFrame;
  refreshHealth();

  dom.bootStatus.textContent = 'reading narration and neural telemetry…';
  try {
    await loadScenario(key);
  } catch (error) {
    failBoot(`Could not load ${key}/manifest.json. ${error?.message || error}`);
  }
}

/* ------------------------------------------------------------------ rendering */
function currentFrame() {
  return state.manifest?.frames?.[state.frame] || null;
}

function scenarioLabel(key = state.key) {
  return state.scenarios.find((s) => s.key === key)?.label || state.manifest?.scene_graph?.title || key;
}

function renderModeToggle() {
  dom.mode.replaceChildren();
  for (const scenario of state.scenarios) {
    const active = scenario.key === state.key;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `mode-btn${active ? ' is-active' : ''}`;
    button.setAttribute('aria-pressed', String(active));
    button.title = scenario.caption || scenario.label;

    const dot = document.createElement('span');
    dot.className = 'mode-dot';
    const copy = document.createElement('span');
    copy.className = 'mode-copy';
    const name = document.createElement('strong');
    name.textContent = scenario.label || scenario.key;
    const meta = document.createElement('span');
    meta.textContent = Number.isFinite(Number(scenario.salience_mean))
      ? `salience ${Number(scenario.salience_mean).toFixed(2)}`
      : (scenario.caption || 'same room · different encoding');
    copy.append(name, meta);
    button.append(dot, copy);
    button.addEventListener('click', () => selectCondition(scenario.key));
    dom.mode.append(button);
  }
  if (dom.mode.scrollWidth > dom.mode.clientWidth) {
    dom.mode.querySelector('.mode-btn.is-active')?.scrollIntoView({
      block: 'nearest',
      inline: 'nearest',
    });
  }
}

function renderSpark(frames) {
  const values = frames.map((f) => Number(f.epoch?.salience ?? 0.5));
  const min = Math.min(0.1, ...values);
  const max = Math.max(0.9, ...values);
  const span = Math.max(1e-3, max - min);
  const n = Math.max(1, values.length - 1);
  const points = values.map((v, i) => [4 + (i * 112) / n, 39 - ((v - min) / span) * 31]);
  const line = points.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(' ');
  const area = `${line} L${points.at(-1)[0].toFixed(1)} 44 L${points[0][0].toFixed(1)} 44 Z`;
  dom.spark.innerHTML = `
    <path d="${area}" fill="rgba(233,178,95,0.08)"></path>
    <path d="${line}" fill="none" stroke="rgba(233,178,95,0.5)" stroke-width="1.1"></path>
    ${points.map((p, i) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${i === state.frame ? 2.5 : 1.3}" fill="${i === state.frame ? '#f6d69c' : 'rgba(244,239,230,0.32)'}"></circle>`).join('')}
  `;
}

function renderBands(epoch) {
  const rows = [
    ['theta', 'θ', Number(epoch.theta || 0)],
    ['alpha', 'α', Number(epoch.alpha || 0)],
    ['beta', 'β', Number(epoch.beta || 0)],
    ['gamma', 'γ', Number(epoch.gamma || 0)],
  ];
  const total = rows.reduce((sum, row) => sum + row[2], 0) || 1;
  dom.bands.replaceChildren();
  for (const [key, glyph, value] of rows) {
    const pct = (value / total) * 100;
    const row = document.createElement('div');
    row.className = 'band';
    row.innerHTML = `
      <span class="band-name">${glyph} ${key}</span>
      <span class="band-track"><i style="width:${clamp(pct, 2, 100).toFixed(1)}%"></i></span>
      <span class="band-val">${pct.toFixed(0)}%</span>
    `;
    dom.bands.append(row);
  }
}

function renderMetrics(epoch) {
  const lat = Number(epoch.lateralization || 0);
  const side = Math.abs(lat) < 0.07 ? 'balanced' : lat > 0 ? 'right' : 'left';
  const items = [
    ['Focus index', `${(Number(epoch.focus_index || 0) * 100).toFixed(0)}`, '%'],
    ['γ / θ', `${Number(epoch.gamma_theta || 0).toFixed(2)}`, ''],
    ['Certainty', `${(Number(epoch.certainty || 0) * 100).toFixed(0)}`, '%'],
    ['Hemifield', `${lat >= 0 ? '+' : ''}${lat.toFixed(2)}`, side],
  ];
  dom.metrics.innerHTML = items.map(([label, value, suffix]) => `
    <div>
      <span class="metric-label">${esc(label)}</span>
      <span class="metric-value">${esc(value)}<small>${esc(suffix)}</small></span>
    </div>
  `).join('');
}

function renderObjects(frame) {
  const entries = Object.entries(frame.object_salience || {}).sort((a, b) => b[1] - a[1]);
  const meta = new Map((frame.objects || []).map((o) => [o.id, o]));
  dom.objects.replaceChildren();
  if (!entries.length) {
    dom.objects.innerHTML = '<span class="prov">no object salience in this frame</span>';
    return;
  }
  const shown = entries.slice(0, 6);
  for (const [id, value] of shown) {
    const object = meta.get(id) || {};
    const row = document.createElement('div');
    row.className = 'obj';
    row.innerHTML = `
      <span class="obj-dot" style="background:${esc(object.color || '#e9b25f')}"></span>
      <span class="obj-name">${esc(object.label || id)}</span>
      <span class="obj-val">${Number(value).toFixed(2)}</span>
      <span class="obj-bar"><b style="width:${clamp(Number(value) * 100, 2, 100).toFixed(1)}%"></b></span>
    `;
    dom.objects.append(row);
  }
  if (entries.length > shown.length) {
    const more = document.createElement('span');
    more.className = 'objects-more';
    more.textContent = `+ ${entries.length - shown.length} lower-salience objects`;
    dom.objects.append(more);
  }
}

function renderEvidence(frame) {
  const keyframe = resolve(state.baseDir, frame.keyframe);
  const attention = resolve(state.baseDir, frame.attention);
  if (keyframe && dom.keyframeImg.getAttribute('src') !== keyframe) dom.keyframeImg.src = keyframe;
  if (attention && dom.attentionImg.getAttribute('src') !== attention) dom.attentionImg.src = attention;
  dom.evKeyframe.hidden = !keyframe;
  dom.evAttention.hidden = !attention;
}

function provenanceChips(manifest) {
  const up = manifest?.upstream || {};
  const chips = [];
  const eeg = String(up.eeg || '');
  chips.push({
    text: eeg.startsWith('synthetic') ? 'synthetic EEG' : 'recorded EEG',
    tone: eeg.startsWith('synthetic') ? 'is-warn' : 'is-real',
  });
  if (up.eeg_file) {
    chips.push({ text: String(up.eeg_file), tone: '' });
  }
  const keyframe = String(up.keyframe || '');
  chips.push({
    text: keyframe.includes('muse') ? 'Muse Image keyframe' : 'procedural keyframe',
    tone: keyframe.includes('muse') ? 'is-real' : '',
  });
  const graph = String(up.scene_graph || '');
  chips.push({
    text: graph.startsWith('meta') ? `scene graph · ${graph.split(':').slice(1).join(':') || 'Meta'}` : 'offline scene graph',
    tone: graph.startsWith('meta') ? 'is-real' : '',
  });
  const depth = String(up.depth || 'depth');
  chips.push({
    text: `depth · ${depth.replace('oracle-keyframe', 'keyframe oracle').replace('shared', 'shared')}`,
    tone: '',
  });
  return chips;
}

function renderProvenance(manifest) {
  dom.provenance.replaceChildren();
  for (const chip of provenanceChips(manifest)) {
    const span = document.createElement('span');
    span.className = `prov ${chip.tone}`;
    span.textContent = chip.text;
    dom.provenance.append(span);
  }
}

function renderStory(frame) {
  const beat = String(frame.label || '').split(' · ')[0] || `epoch ${state.frame + 1}`;
  dom.beat.textContent = beat;
  dom.conditionChip.textContent = scenarioLabel();
  const quote = frame.narrative
    || state.manifest?.scene_graph?.beats?.[state.frame]?.narrative
    || state.manifest?.scene_graph?.narrative
    || '';
  clearTimeout(narrativeTimer);
  dom.narrative.style.opacity = '0';
  narrativeTimer = setTimeout(() => {
    dom.narrative.textContent = quote;
    dom.narrative.style.opacity = '1';
  }, 110);
}

function renderTimeline(frames) {
  dom.scrub.max = String(Math.max(0, frames.length - 1));
  dom.scrub.value = String(state.frame);
  const progress = frames.length > 1 ? (state.frame / (frames.length - 1)) * 100 : 100;
  dom.sliderFill.style.setProperty('--progress', `${progress}%`);
  dom.sliderFill.parentElement.style.setProperty('--progress', `${progress}%`);
  dom.epochCount.textContent = `epoch ${state.frame + 1} / ${frames.length}`;

  dom.ticks.replaceChildren();
  frames.forEach((frame, index) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `tick${index === state.frame ? ' is-active' : ''}`;
    const label = String(frame.label || `epoch ${index + 1}`).split(' · ')[0];
    button.textContent = label;
    button.title = frame.label || label;
    button.addEventListener('click', () => setFrame(index));
    dom.ticks.append(button);
  });
}

function renderXR() {
  const frame = currentFrame();
  if (!frame) return;
  dom.xrEpoch.textContent = `epoch ${state.frame + 1} / ${state.manifest.frames.length}`;
  dom.xrSalience.textContent = Number(frame.epoch?.salience ?? 0).toFixed(2);
  dom.xrCondition.textContent = scenarioLabel();
}

function renderAll() {
  const manifest = state.manifest;
  if (!manifest) return;
  const frame = currentFrame();
  if (!frame) return;

  renderSpark(manifest.frames);
  dom.salience.textContent = Number(frame.epoch?.salience ?? 0).toFixed(2);
  dom.salienceDial.style.setProperty('--p', `${Math.round(clamp(Number(frame.epoch?.salience ?? 0), 0, 1) * 100)}%`);
  renderBands(frame.epoch || {});
  renderMetrics(frame.epoch || {});
  renderObjects(frame);
  renderEvidence(frame);
  renderStory(frame);
  renderProvenance(manifest);
  renderTimeline(manifest.frames);
  dom.splatCount.textContent = int.format(Number(frame.n_splats || 0));
  renderXR();
  state.seenFrames.add(`${state.key}:${state.frame}`);
}

/* ------------------------------------------------------------- frame loading */
function frameUrl(frame) {
  return resolve(state.baseDir, frame?.splat);
}

function showFrame() {
  const frame = currentFrame();
  const url = frameUrl(frame);
  if (!url) {
    toast('This epoch has no .splat asset.');
    return Promise.resolve();
  }
  setBusy(true);
  return stage.show(url).finally(() => setBusy(false));
}

function setFrame(index, { pause = true } = {}) {
  const frames = state.manifest?.frames;
  if (!frames?.length) return;
  const next = ((index % frames.length) + frames.length) % frames.length;
  if (next === state.frame) {
    renderAll();
    return;
  }
  state.frame = next;
  renderAll();
  showFrame();
  if (pause) setPlaying(false);
}

function setPlaying(on) {
  state.playing = Boolean(on) && Boolean(state.manifest);
  dom.play.classList.toggle('is-playing', state.playing);
  dom.play.setAttribute('aria-label', state.playing ? 'Pause consolidation' : 'Play consolidation');
  clearInterval(playTimer);
  if (state.playing) {
    playTimer = setInterval(() => {
      const frames = state.manifest?.frames || [];
      if (!frames.length) return;
      setFrame((state.frame + 1) % frames.length, { pause: false });
    }, 1750);
  }
}

async function selectCondition(key) {
  if (!key || key === state.key) return;
  setPlaying(false);
  setBusy(true);
  try {
    await loadScenario(key, { keepFrame: true });
  } catch (error) {
    toast(`Could not switch condition: ${error?.message || error}`);
  } finally {
    setBusy(false);
  }
}

function toggleCondition() {
  if (state.scenarios.length < 2) return;
  const index = state.scenarios.findIndex((s) => s.key === state.key);
  const next = state.scenarios[(index + 1) % state.scenarios.length];
  selectCondition(next.key);
}

function schedulePrefetch() {
  const frames = state.manifest?.frames || [];
  const urls = frames.map((f) => frameUrl(f)).filter((u) => u && u !== frameUrl(currentFrame()));
  if (!urls.length) return;
  const run = () => prefetchSequential(urls);
  if ('requestIdleCallback' in window) requestIdleCallback(run, { timeout: 5000 });
  else setTimeout(run, 2800);
}

async function prefetchSequential(urls) {
  if (prefetching) return;
  prefetching = true;
  for (const url of urls) {
    try {
      await fetch(url, { cache: 'force-cache' });
    } catch {
      /* best effort */
    }
  }
  prefetching = false;
}

/* ---------------------------------------------------------------- callbacks */
function handleProgress(pct) {
  if (state.firstReady) return;
  const value = clamp(Number(pct) || 0, 0, 100);
  dom.bootFill.style.width = `${value}%`;
  dom.bootStatus.textContent = value >= 100
    ? 'processing gaussians…'
    : `lifting gaussians · ${Math.round(value)}%`;
}

function handleReady() {
  if (!state.firstReady) {
    state.firstReady = true;
    finishBoot();
  }
  setBusy(false);
}

function handleStageError(error) {
  const message = error?.message || String(error);
  toast(`Could not load the volume: ${message}`);
  if (!state.firstReady) failBoot(`Could not load the volume. ${message}`);
  setBusy(false);
  console.error(error);
}

function handleXR(active) {
  state.xr = active;
  document.body.classList.toggle('xr-live', active);
  if (active) {
    setPlaying(false);
    renderXR();
  }
}

function handleTrigger(direction) {
  if (!state.manifest) return;
  setFrame(direction === 'prev' ? state.frame - 1 : state.frame + 1);
}

/* --------------------------------------------------------------------- modal */
function openModal() {
  const manifest = state.manifest;
  if (manifest) {
    const up = manifest.upstream || {};
    dom.modalProv.innerHTML = [
      ...provenanceChips(manifest).map((c) => `<span class="prov ${c.tone}">${esc(c.text)}</span>`),
      `<span class="prov">${esc(int.format(Number(manifest.stats?.n_splats_peak || 0)))} splats peak</span>`,
      `<span class="prov">${esc(manifest.frames?.length || 0)} epochs</span>`,
      `<span class="prov">${esc(String(up.eeg_calibration || 'population calibration'))}</span>`,
    ].join('');
  }
  dom.modal.hidden = false;
}

function closeModal() {
  dom.modal.hidden = true;
}

function openLightbox(src, caption) {
  if (!src) {
    toast('That image is not available for this frame.');
    return;
  }
  dom.lightboxImg.src = src;
  dom.lightboxImg.alt = caption;
  dom.lightboxCap.textContent = caption;
  dom.lightbox.hidden = false;
}

function closeLightbox() {
  dom.lightbox.hidden = true;
  dom.lightboxImg.removeAttribute('src');
}

/* --------------------------------------------------- reconstruction status */
function reconInfo() {
  return state.health?.reconstruction || null;
}

function reconUsable() {
  const info = reconInfo();
  return Boolean(info?.configured && info?.ready && info?.enabled);
}

async function refreshHealth() {
  try {
    state.health = await fetchJSON('/api/health');
  } catch {
    state.health = null;
  }
  updateEEGModes();
}

function updateEEGModes() {
  const info = reconInfo();
  const usable = reconUsable();
  for (const button of dom.eegModes?.querySelectorAll('[data-eeg-mode]') || []) {
    const mode = button.dataset.eegMode;
    button.classList.toggle('is-active', mode === state.eegMode);
    if (mode === 'reconstruct') {
      button.title = usable
        ? 'Experimental BrainVis static-frame reconstruction on an external GPU'
        : (info?.reason || 'GPU reconstruction is not configured');
    }
  }

  if (state.eegMode === 'reconstruct') {
    let message = '';
    let warn = false;
    if (!info?.configured) {
      message = 'No BrainVis GPU worker is configured. The CPU demo cannot run Stable Diffusion.';
      warn = true;
    } else if (!info?.enabled) {
      message = 'GPU reconstruction is disabled on this public deployment. Set an admin token or run locally.';
      warn = true;
    } else if (!info?.ready) {
      message = `GPU worker is not ready: ${info?.reason || 'checkpoint or CUDA missing'}`;
      warn = true;
    } else {
      message = 'Ready. Uses 128-channel EEG in BrainVis training order; returns independent static frames, not video.';
    }
    dom.eegModeHelp.innerHTML = esc(message);
    dom.eegModeHelp.classList.toggle('is-warn', warn);
  } else {
    dom.eegModeHelp.innerHTML = '<strong>Condition</strong> re-lifts the same baked scene. EEG changes only density, opacity and spread — never RGB pixels. This is the controlled demo.';
    dom.eegModeHelp.classList.remove('is-warn');
  }
  updateEEGSubmit();
}

function setEEGMode(mode) {
  if (mode !== 'condition' && mode !== 'reconstruct') return;
  state.eegMode = mode;
  updateEEGModes();
  if (state.eegMode === 'reconstruct') {
    setEEGStatus(
      reconUsable()
        ? 'Ready. Upload 128 channels; the GPU worker will generate static frames.'
        : 'Reconstruction is unavailable. See the status above.',
      reconUsable() ? 'is-ok' : 'is-error',
    );
  } else {
    setEEGStatus('Choose a recording to begin.');
  }
}

function updateEEGSubmit() {
  if (!dom.eegSubmit) return;
  const canRun = state.eegMode === 'condition' || reconUsable();
  dom.eegSubmit.disabled = Boolean(state.uploading || !state.uploadFile || !canRun);
  if (state.uploading) return;
  dom.eegSubmit.textContent = state.eegMode === 'reconstruct'
    ? 'Reconstruct frames'
    : 'Condition volume';
}

/* --------------------------------------------------------------- EEG upload */
const EEG_MAX_BYTES = 12 * 1024 * 1024;
const EEG_EXTENSIONS = new Set(['csv', 'tsv', 'txt', 'npy']);

function formatBytes(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function setEEGStatus(message, tone = '') {
  dom.eegStatus.textContent = message;
  dom.eegStatus.className = `eeg-status${tone ? ` ${tone}` : ''}`;
}

function setEEGProgress(percent) {
  dom.eegProgressFill.style.width = `${clamp(Number(percent) || 0, 0, 100).toFixed(1)}%`;
}

function clearEEGSelection() {
  state.uploadFile = null;
  dom.eegFile.value = '';
  dom.eegDrop.classList.remove('is-set', 'is-over');
  dom.eegDropTitle.textContent = 'Drop an EEG file here';
  dom.eegDropHint.textContent = 'CSV with channel columns, or NPY · max 12 MB';
  dom.eegSubmit.disabled = true;
}

function resetEEGForm() {
  clearEEGSelection();
  dom.eegSfreq.value = '250';
  setEEGProgress(0);
  setEEGStatus('Choose a recording to begin.');
  updateEEGModes();
}

function openEEGModal() {
  closeModal();
  dom.eegModal.hidden = false;
  refreshHealth();
  updateEEGModes();
  setTimeout(() => dom.eegDrop.focus(), 0);
}

function closeEEGModal() {
  dom.eegModal.hidden = true;
  if (!state.uploading) resetEEGForm();
}

function setEEGFile(file) {
  if (!file) return;
  const extension = String(file.name || '').split('.').pop().toLowerCase();
  if (!EEG_EXTENSIONS.has(extension)) {
    clearEEGSelection();
    setEEGStatus(
      'That file type is not supported yet. Upload a CSV, TSV, TXT or NPY file.',
      'is-error',
    );
    return;
  }
  if (!file.size) {
    clearEEGSelection();
    setEEGStatus('That file is empty.', 'is-error');
    return;
  }
  if (file.size > EEG_MAX_BYTES) {
    clearEEGSelection();
    setEEGStatus(
      `That file is ${formatBytes(file.size)}; the limit is ${formatBytes(EEG_MAX_BYTES)}.`,
      'is-error',
    );
    return;
  }

  state.uploadFile = file;
  dom.eegDrop.classList.add('is-set');
  dom.eegDropTitle.textContent = file.name;
  dom.eegDropHint.textContent = `${formatBytes(file.size)} · ${extension.toUpperCase()} · click to replace`;
  setEEGProgress(0);
  if (state.eegMode === 'reconstruct') {
    const info = reconInfo();
    setEEGStatus(
      reconUsable()
        ? `Ready. ${info?.channels_required || 128} channels required; frames are independent, not video.`
        : 'Reconstruction is unavailable. Switch to condition mode or configure the GPU worker.',
      reconUsable() ? 'is-ok' : 'is-error',
    );
  } else {
    setEEGStatus('Ready. The baked keyframe and depth map will be reused.', 'is-ok');
  }
  updateEEGSubmit();
}

function uploadEEGFile(file, sfreq, mode, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/eeg');
    xhr.responseType = 'text';
    xhr.timeout = 120_000;

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress?.((event.loaded / event.total) * 100);
    });
    xhr.addEventListener('load', () => {
      let body = null;
      try {
        body = JSON.parse(xhr.responseText || '{}');
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300 && body?.job) {
        resolve(body);
        return;
      }
      reject(new Error(body?.detail || body?.error || `Upload failed (${xhr.status}).`));
    });
    xhr.addEventListener('error', () => {
      reject(new Error('Network error while uploading the EEG file.'));
    });
    xhr.addEventListener('timeout', () => {
      reject(new Error('The upload timed out before the server received the file.'));
    });

    const form = new FormData();
    form.append('file', file, file.name);
    form.append('sfreq', String(sfreq));
    form.append('mode', mode);
    xhr.send(form);
  });
}

async function pollEEGJob(jobId) {
  const started = Date.now();
  const timeout = 20 * 60 * 1000;
  let failures = 0;
  while (Date.now() - started < timeout) {
    let job;
    try {
      job = await fetchJSON(`/api/eeg/jobs/${encodeURIComponent(jobId)}`);
      failures = 0;
    } catch (error) {
      // A Render cold start or a transient network blip should not discard a job that
      // is still running server-side. Give up only after a short run of failures.
      failures += 1;
      if (failures >= 6) throw error;
      setEEGStatus(`Connection interrupted; retrying (${failures}/6)…`, 'is-busy');
      await new Promise((resolve) => setTimeout(resolve, 2000));
      continue;
    }
    if (job.message) {
      setEEGStatus(job.message, job.state === 'error' ? 'is-error' : 'is-busy');
    }
    const progress = Number(job.progress);
    if (Number.isFinite(progress)) {
      setEEGProgress(25 + clamp(progress, 0, 1) * 75);
    }
    if (job.state === 'done') return job;
    if (job.state === 'error') {
      throw new Error(job.error || 'The uploaded EEG could not be processed.');
    }
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
  throw new Error(
    'The re-lift is taking longer than expected. The free instance may be busy; '
    + 'try a shorter recording or check the service logs.',
  );
}

function completeReconstruction(job) {
  const images = Array.isArray(job.images) ? job.images : [];
  if (!images.length) throw new Error('The GPU worker returned no frames.');

  dom.reconGrid.innerHTML = images.map((image, index) => {
    const caption = [
      `frame ${String(index + 1).padStart(2, '0')}`,
      image.label ? esc(image.label) : 'unclassified',
      image.predicted_class != null ? `class ${esc(image.predicted_class)}` : '',
    ].filter(Boolean).join(' · ');
    return `
      <figure>
        <button type="button" data-recon-src="${esc(asset(image.url))}" data-recon-cap="${caption}">
          <img src="${esc(asset(image.url))}" alt="${caption}" loading="lazy" />
        </button>
        <figcaption>${caption}</figcaption>
      </figure>`;
  }).join('');

  const meta = job.meta || {};
  dom.reconMeta.textContent = [
    `${images.length} static frames`,
    meta.window_s ? `${meta.window_s}s windows` : '',
    job.filename ? String(job.filename) : '',
    job.elapsed_s ? `${Number(job.elapsed_s).toFixed(1)}s GPU` : '',
  ].filter(Boolean).join(' · ');

  dom.reconModal.hidden = false;
  state.uploading = false;
  resetEEGForm();
  dom.eegModal.hidden = true;
  toast(`Experimental reconstruction ready · ${images.length} static frames`, 7000);
}

async function completeEEGUpload(job) {
  const entry = job.scenario;
  if (!entry?.manifest) throw new Error('The upload finished without a manifest.');

  const existing = state.scenarios.findIndex((scenario) => scenario.key === entry.key);
  if (existing >= 0) state.scenarios[existing] = entry;
  else state.scenarios.push(entry);

  // Keep the condition switcher compact; the server also keeps only a few jobs.
  const uploads = state.scenarios.filter((scenario) => scenario.recorded);
  if (uploads.length > 2) {
    const retired = new Set(uploads.slice(0, uploads.length - 2).map((scenario) => scenario.key));
    state.scenarios = state.scenarios.filter((scenario) => !retired.has(scenario.key));
  }
  // Renumber after pruning so two surviving uploads cannot share a label.
  state.scenarios
    .filter((scenario) => scenario.recorded)
    .forEach((scenario, index) => {
      scenario.label = index ? `Recorded EEG ${index + 1}` : 'Recorded EEG';
    });

  renderModeToggle();
  await loadScenario(entry.key, { keepFrame: false });
  state.uploading = false;
  resetEEGForm();
  setEEGStatus('Recorded EEG loaded. Only density, opacity and spread changed.', 'is-ok');
  dom.eegModal.hidden = true;
  toast(
    `Recorded EEG loaded · ${job.n_epochs}→${job.n_beats} beats · `
    + `salience ${Number(job.scenario.salience_mean || 0).toFixed(2)}`,
    7000,
  );
}

async function submitEEG() {
  if (state.uploading || !state.uploadFile) return;
  const mode = state.eegMode;
  if (mode === 'reconstruct' && !reconUsable()) {
    setEEGStatus('Experimental reconstruction is not available on this deployment.', 'is-error');
    return;
  }
  const sfreq = Number(dom.eegSfreq.value);
  if (!Number.isFinite(sfreq) || sfreq < 100 || sfreq > 2000) {
    setEEGStatus('Enter a sampling rate between 100 and 2000 Hz.', 'is-error');
    return;
  }

  state.uploading = true;
  updateEEGSubmit();
  dom.eegSubmit.textContent = 'Uploading…';
  setEEGProgress(2);
  setEEGStatus(
    mode === 'reconstruct'
      ? 'Uploading the recording for GPU reconstruction…'
      : 'Uploading the recording…',
    'is-busy',
  );
  setBusy(true);

  try {
    const queued = await uploadEEGFile(state.uploadFile, sfreq, mode, (percent) => {
      setEEGProgress(2 + percent * 0.2);
      setEEGStatus(`Uploading the recording · ${Math.round(percent)}%`, 'is-busy');
    });
    setEEGProgress(25);
    setEEGStatus(
      mode === 'reconstruct'
        ? 'Queued. The external GPU worker is loading the BrainVis cascade…'
        : 'Queued. Analysing posterior EEG bands…',
      'is-busy',
    );
    const done = await pollEEGJob(queued.job);
    if (mode === 'reconstruct') completeReconstruction(done);
    else await completeEEGUpload(done);
  } catch (error) {
    state.uploading = false;
    updateEEGSubmit();
    setEEGProgress(0);
    setEEGStatus(error?.message || String(error), 'is-error');
  } finally {
    setBusy(false);
    if (!state.uploading) updateEEGSubmit();
  }
}

/* -------------------------------------------------------------------- events */
function bind() {
  dom.play.addEventListener('click', () => setPlaying(!state.playing));
  dom.scrub.addEventListener('input', (event) => {
    setFrame(Number(event.target.value));
  });
  dom.scrub.addEventListener('pointerdown', () => setPlaying(false));

  dom.aboutBtn.addEventListener('click', openModal);
  dom.modal.addEventListener('click', (event) => {
    if (event.target.closest('[data-close]')) closeModal();
  });

  dom.eegBtn.addEventListener('click', openEEGModal);
  dom.eegModal.addEventListener('click', (event) => {
    if (event.target.closest('[data-eeg-close]')) closeEEGModal();
  });
  dom.eegModes?.addEventListener('click', (event) => {
    const button = event.target.closest('[data-eeg-mode]');
    if (button && !button.disabled) setEEGMode(button.dataset.eegMode);
  });
  dom.reconModal?.addEventListener('click', (event) => {
    if (event.target.closest('[data-recon-close]')) dom.reconModal.hidden = true;
    const image = event.target.closest('[data-recon-src]');
    if (image) openLightbox(image.dataset.reconSrc, image.dataset.reconCap);
  });
  dom.eegDrop.addEventListener('click', () => dom.eegFile.click());
  dom.eegDrop.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      event.stopPropagation();
      dom.eegFile.click();
    }
  });
  dom.eegFile.addEventListener('change', (event) => {
    setEEGFile(event.target.files?.[0]);
  });
  for (const type of ['dragenter', 'dragover']) {
    dom.eegDrop.addEventListener(type, (event) => {
      event.preventDefault();
      dom.eegDrop.classList.add('is-over');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    dom.eegDrop.addEventListener(type, (event) => {
      event.preventDefault();
      dom.eegDrop.classList.remove('is-over');
    });
  }
  dom.eegDrop.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) setEEGFile(file);
  });
  dom.eegSubmit.addEventListener('click', submitEEG);

  dom.lightbox.addEventListener('click', (event) => {
    if (event.target === dom.lightbox || event.target.closest('.lightbox-close')) closeLightbox();
  });
  dom.evKeyframe.addEventListener('click', () => {
    openLightbox(dom.keyframeImg.getAttribute('src'), 'Synthesised keyframe · the only image the lifter sees');
  });
  dom.evAttention.addEventListener('click', () => {
    openLightbox(
      dom.attentionImg.getAttribute('src'),
      `EEG salience field · epoch ${state.frame + 1} · warm = more densely consolidated`,
    );
  });

  dom.xrPrev.addEventListener('click', () => handleTrigger('prev'));
  dom.xrNext.addEventListener('click', () => handleTrigger('next'));
  dom.xrCondition.addEventListener('click', toggleCondition);

  window.addEventListener('keydown', (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const tag = event.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    if (!dom.lightbox.hidden) {
      if (event.key === 'Escape') closeLightbox();
      return;
    }
    if (!dom.reconModal.hidden) {
      if (event.key === 'Escape') dom.reconModal.hidden = true;
      return;
    }
    if (!dom.eegModal.hidden) {
      if (event.key === 'Escape') closeEEGModal();
      return;
    }
    if (!dom.modal.hidden) {
      if (event.key === 'Escape') closeModal();
      return;
    }

    switch (event.key) {
      case 'ArrowRight':
        event.preventDefault();
        setFrame(state.frame + 1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        setFrame(state.frame - 1);
        break;
      case ' ':
        event.preventDefault();
        setPlaying(!state.playing);
        break;
      case 'h':
      case 'H':
        dom.app.classList.toggle('hud-hidden');
        break;
      case 'r':
      case 'R':
        stage.resetView();
        break;
      case 'f':
      case 'F':
        selectCondition('focus');
        break;
      case 'd':
      case 'D':
        selectCondition('distracted');
        break;
      case '1':
      case '2': {
        const scenario = state.scenarios[Number(event.key) - 1];
        if (scenario) selectCondition(scenario.key);
        break;
      }
      case 'a':
      case 'A':
        openLightbox(dom.attentionImg.getAttribute('src'), 'EEG salience field');
        break;
      default:
        break;
    }
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) setPlaying(false);
  });
}

/* ---------------------------------------------------------------------- go */
bind();
boot();
