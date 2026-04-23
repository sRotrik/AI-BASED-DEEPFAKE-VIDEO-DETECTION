/* ===================================================
   script.js — DeepGuard frontend logic
   =================================================== */

'use strict';

// ─────────────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────────────
const dropZone        = document.getElementById('drop-zone');
const fileInput       = document.getElementById('file-input');
const videoPreviewWrap = document.getElementById('video-preview-wrap');
const videoPreview    = document.getElementById('video-preview');
const removeBtn       = document.getElementById('remove-btn');
const videoFilename   = document.getElementById('video-filename');
const videoSize       = document.getElementById('video-size');
const analyzeBtn      = document.getElementById('analyze-btn');
const btnLabel        = document.getElementById('btn-label');
const resultPlaceholder = document.getElementById('result-placeholder');
const resultCard      = document.getElementById('result-card');
const verdictBadge    = document.getElementById('verdict-badge');
const verdictEmoji    = document.getElementById('verdict-emoji');
const verdictLabel    = document.getElementById('verdict-label');
const verdictSub      = document.getElementById('verdict-sub');
const realPct         = document.getElementById('real-pct');
const fakePct         = document.getElementById('fake-pct');
const realBar         = document.getElementById('real-bar');
const fakeBar         = document.getElementById('fake-bar');
const statAuc         = document.getElementById('stat-auc');
const statThresh      = document.getElementById('stat-thresh');
const statDevice      = document.getElementById('stat-device');
const toast           = document.getElementById('toast');
const toastMsg        = document.getElementById('toast-msg');
const modelStatusText = document.getElementById('model-status-text');

let selectedFile = null;

// ─────────────────────────────────────────────────────
// Ping model status on load
// ─────────────────────────────────────────────────────
async function checkModelStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      modelStatusText.textContent = `Model ready · AUC ${data.val_auc}`;
    } else {
      modelStatusText.textContent = 'Model status unknown';
    }
  } catch {
    modelStatusText.textContent = 'Server not reachable';
  }
}
checkModelStatus();

// ─────────────────────────────────────────────────────
// Utility: format file size
// ─────────────────────────────────────────────────────
function formatSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ─────────────────────────────────────────────────────
// Utility: show / hide toast
// ─────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg) {
  toastMsg.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 4000);
}

// ─────────────────────────────────────────────────────
// Load a file into the preview section
// ─────────────────────────────────────────────────────
function loadFile(file) {
  if (!file || !file.type.startsWith('video/')) {
    showToast('Please select a valid video file (MP4, AVI, MOV, MKV…)');
    return;
  }
  if (file.size > 200 * 1024 * 1024) {
    showToast('File too large. Please use a video under 200 MB.');
    return;
  }

  selectedFile = file;

  // Show preview
  const url = URL.createObjectURL(file);
  videoPreview.src = url;
  videoFilename.textContent = file.name;
  videoSize.textContent     = formatSize(file.size);
  videoPreviewWrap.classList.add('visible');
  dropZone.classList.add('has-video');

  // Enable analyze button
  analyzeBtn.disabled = false;

  // Reset any previous result
  resetResult();
}

// ─────────────────────────────────────────────────────
// Reset result panel
// ─────────────────────────────────────────────────────
function resetResult() {
  resultCard.classList.remove('visible');
  resultPlaceholder.style.display = '';
  // Reset bars to 0
  realBar.style.width = '0%';
  fakeBar.style.width = '0%';
}

// ─────────────────────────────────────────────────────
// Remove selected video
// ─────────────────────────────────────────────────────
function clearVideo() {
  selectedFile = null;
  videoPreview.src = '';
  fileInput.value  = '';
  videoPreviewWrap.classList.remove('visible');
  dropZone.classList.remove('has-video');
  analyzeBtn.disabled = true;
  resetResult();
}

removeBtn.addEventListener('click', clearVideo);

// ─────────────────────────────────────────────────────
// Drag & Drop events
// ─────────────────────────────────────────────────────
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
});

['dragleave', 'dragend'].forEach(evt => {
  dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

// Click to open file picker (but not if the label inside was clicked)
dropZone.addEventListener('click', e => {
  if (e.target.id === 'browse-btn' || e.target.closest('#browse-btn')) return;
  fileInput.click();
});

// Keyboard accessibility
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

// ─────────────────────────────────────────────────────
// Set analyze button loading state
// ─────────────────────────────────────────────────────
function setLoading(is) {
  analyzeBtn.classList.toggle('loading', is);
  analyzeBtn.disabled = is;
  btnLabel.textContent = is ? 'Analyzing…' : 'Analyze Video';
}

// ─────────────────────────────────────────────────────
// Render results
// ─────────────────────────────────────────────────────
function renderResult(data) {
  const isReal = data.label === 'REAL';

  // Verdict badge
  verdictBadge.className = `verdict-badge ${isReal ? 'real' : 'fake'}`;
  verdictEmoji.textContent = isReal ? '✅' : '🚨';
  verdictLabel.textContent = isReal ? 'REAL' : 'FAKE';
  verdictSub.textContent   = isReal
    ? `${data.p_real}% confidence it is authentic`
    : `${data.p_fake}% confidence it is manipulated`;

  // Stats chips
  statAuc.textContent    = data.val_auc;
  statThresh.textContent = data.threshold + '%';
  statDevice.textContent = data.device.toUpperCase();

  // Show card, hide placeholder
  resultPlaceholder.style.display = 'none';
  resultCard.classList.add('visible');

  // Animate bars (short delay so CSS transition fires)
  requestAnimationFrame(() => {
    setTimeout(() => {
      realPct.textContent  = data.p_real + '%';
      fakePct.textContent  = data.p_fake + '%';
      realBar.style.width  = data.p_real + '%';
      fakeBar.style.width  = data.p_fake + '%';
    }, 80);
  });
}

// ─────────────────────────────────────────────────────
// Submit to API
// ─────────────────────────────────────────────────────
async function analyzeVideo() {
  if (!selectedFile) return;

  setLoading(true);
  resetResult();

  const formData = new FormData();
  formData.append('video', selectedFile);

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      body:   formData,
    });

    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || 'Server error. Please try again.');
      return;
    }

    renderResult(data);

  } catch (err) {
    console.error(err);
    showToast('Could not reach the server. Is app.py running?');
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener('click', analyzeVideo);
