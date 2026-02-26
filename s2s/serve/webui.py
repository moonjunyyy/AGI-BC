"""
S2S Web UI — pure HTML/JS, no Python UI framework required.

Registers a single GET / route on the FastAPI app that returns a
self-contained single-page application.  The SPA communicates with the
same server via its existing WebSocket and REST endpoints:

    WS  /ws/chat    — bidirectional PCM streaming
    POST /api/eval  — dual-agent dialogue evaluation
    GET  /health    — status check
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# Single-page application HTML
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S2S Interface</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#0f0f0f;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
header{padding:10px 18px;border-bottom:1px solid #222;display:flex;align-items:center;gap:12px;flex-shrink:0}
h1{font-size:.95rem;font-weight:600;color:#fff;margin-right:auto}
.badge{font-size:.72rem;padding:2px 8px;border-radius:99px;background:#222}
.badge.ok{background:#1b3b1b;color:#5c5}
.badge.err{background:#3b1b1b;color:#e55}
input.url{background:#1a1a1a;border:1px solid #333;color:#ccc;padding:4px 8px;border-radius:4px;font-size:.78rem;width:220px}
button.sm{padding:4px 12px;background:#1a3a5c;border:none;border-radius:4px;color:#fff;cursor:pointer;font-size:.78rem}
button.sm:hover{background:#234e7a}
nav{display:flex;border-bottom:1px solid #222;flex-shrink:0}
nav button{background:none;border:none;border-bottom:2px solid transparent;color:#777;padding:7px 18px;cursor:pointer;font-size:.82rem}
nav button.on{color:#fff;border-bottom-color:#4a9eff}
.tab{display:none;flex:1;overflow:hidden;flex-direction:column}
.tab.on{display:flex}

/* Chat */
#transcript{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:7px}
.msg{max-width:68%;padding:8px 11px;border-radius:8px;font-size:.82rem;line-height:1.45;word-break:break-word}
.msg.u{align-self:flex-end;background:#1a3a5c;color:#cde}
.msg.m{align-self:flex-start;background:#1e1e1e;border:1px solid #2a2a2a}
.bar{padding:10px 14px;border-top:1px solid #222;display:flex;align-items:center;gap:10px;flex-shrink:0}
#rec{width:40px;height:40px;border-radius:50%;border:none;cursor:pointer;font-size:1rem;background:#1a3a5c;color:#fff;transition:background .15s;flex-shrink:0}
#rec.on{background:#c0392b;animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.55}}
#hint{font-size:.76rem;color:#666;flex:1}

/* Eval */
.eval-wrap{display:flex;flex:1;overflow:hidden}
.eval-form{width:280px;flex-shrink:0;padding:14px;border-right:1px solid #222;overflow-y:auto;display:flex;flex-direction:column;gap:10px}
.eval-form label{font-size:.74rem;color:#777;display:block;margin-bottom:3px}
.eval-form input,.eval-form textarea{width:100%;background:#1a1a1a;border:1px solid #333;color:#ccc;padding:5px 7px;border-radius:4px;font-size:.8rem;font-family:inherit}
.eval-form textarea{resize:vertical;min-height:56px}
#run-btn{padding:7px;background:#1b3b1b;border:none;border-radius:4px;color:#5c5;cursor:pointer;font-size:.82rem}
#run-btn:disabled{opacity:.45;cursor:default}
.eval-right{flex:1;display:flex;flex-direction:column;overflow:hidden}
#eval-log{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:7px}
.em{padding:7px 11px;border-radius:8px;font-size:.78rem;line-height:1.4;max-width:68%}
.em.a{background:#1a3a5c}
.em.b{background:#3a1a5c;align-self:flex-end}
.em .role{font-size:.68rem;opacity:.65;font-weight:600;margin-bottom:2px}
#eval-summary{margin:8px;padding:8px 11px;background:#1b3b1b;border:1px solid #2b5b2b;border-radius:5px;font-size:.78rem;display:none}
</style>
</head>
<body>
<header>
  <h1>S2S Interface</h1>
  <span id="badge" class="badge">disconnected</span>
  <input id="url" class="url" type="text" value="" placeholder="ws://host:port">
  <button class="sm" onclick="toggleConn()">Connect</button>
</header>
<nav>
  <button class="on" onclick="switchTab('chat',this)">Chat</button>
  <button onclick="switchTab('eval',this)">Dual-Agent Eval</button>
</nav>

<!-- Chat tab -->
<div id="tab-chat" class="tab on">
  <div id="transcript"></div>
  <div class="bar">
    <button id="rec" title="Hold to speak"
      onmousedown="startRec()" onmouseup="stopRec()"
      ontouchstart="startRec(event)" ontouchend="stopRec()">&#127897;</button>
    <span id="hint">Connect to a server, then hold the mic button to speak.</span>
  </div>
</div>

<!-- Eval tab -->
<div id="tab-eval" class="tab">
  <div class="eval-wrap">
    <div class="eval-form">
      <div>
        <label>Keyword to describe / guess</label>
        <input id="ek" type="text" placeholder="e.g. photosynthesis">
      </div>
      <div>
        <label>Max turns</label>
        <input id="et" type="number" value="10" min="2" max="200">
      </div>
      <div>
        <label>Describer prompt (optional override)</label>
        <textarea id="edp" rows="2" placeholder="Describe '{keyword}' without saying it..."></textarea>
      </div>
      <div>
        <label>Guesser prompt (optional override)</label>
        <textarea id="egp" rows="2" placeholder="Guess the word being described..."></textarea>
      </div>
      <button id="run-btn" onclick="runEval()">&#9654; Run Evaluation</button>
    </div>
    <div class="eval-right">
      <div id="eval-log"></div>
      <div id="eval-summary"></div>
    </div>
  </div>
</div>

<script>
// Tab switching
function switchTab(name, btn) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('on'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('on'));
  document.getElementById('tab-' + name).classList.add('on');
  btn.classList.add('on');
}

// Default URL from current page origin
(function() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  document.getElementById('url').value = proto + '//' + location.host;
})();

function wsBase()   { return document.getElementById('url').value.trim(); }
function httpBase() { return wsBase().replace(/^ws(s?):\/\//, 'http$1://'); }
function hint(t)    { document.getElementById('hint').textContent = t; }

function addMsg(role, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.textContent = text;
  const tr = document.getElementById('transcript');
  tr.appendChild(d);
  tr.scrollTop = tr.scrollHeight;
}

// WebSocket
let ws = null, audioCtx = null, nextPlay = 0;

function toggleConn() {
  if (ws && ws.readyState <= 1) { ws.close(); return; }
  ws = new WebSocket(wsBase() + '/ws/chat');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    setBadge('connected', true);
    document.querySelector('header button.sm').textContent = 'Disconnect';
    hint('Connected. Hold the mic button to speak.');
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextPlay = audioCtx.currentTime;
  };
  ws.onclose = () => {
    setBadge('disconnected', false);
    document.querySelector('header button.sm').textContent = 'Connect';
    hint('Disconnected.');
  };
  ws.onerror = () => setBadge('error', false);
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') {
      try { const m = JSON.parse(ev.data); if (m.text) addMsg('m', m.text); } catch(_) {}
    } else {
      playPCM(ev.data);
    }
  };
}

function setBadge(text, ok) {
  const b = document.getElementById('badge');
  b.textContent = text;
  b.className = 'badge' + (ok ? ' ok' : ' err');
}

// PCM playback (Int16 LE, 24 kHz, mono)
function playPCM(buf) {
  const i16 = new Int16Array(buf);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
  const ab = audioCtx.createBuffer(1, f32.length, 24000);
  ab.copyToChannel(f32, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = ab;
  src.connect(audioCtx.destination);
  const t = Math.max(nextPlay, audioCtx.currentTime + 0.04);
  src.start(t);
  nextPlay = t + ab.duration;
}

// Mic recording — ScriptProcessor captures raw PCM Float32 -> Int16
let micStream = null, scriptNode = null, recCtx = null;

async function startRec(ev) {
  if (ev) ev.preventDefault();
  if (!ws || ws.readyState !== 1) { hint('Not connected.'); return; }
  document.getElementById('rec').classList.add('on');
  hint('Recording...');
  addMsg('u', '[speaking...]');
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: 24000, channelCount: 1, echoCancellation: true }
  });
  if (audioCtx.state === 'suspended') await audioCtx.resume();
  recCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
  const src = recCtx.createMediaStreamSource(micStream);
  scriptNode = recCtx.createScriptProcessor(2048, 1, 1);
  scriptNode.onaudioprocess = e => {
    if (!ws || ws.readyState !== 1) return;
    const f32 = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++)
      i16[i] = Math.max(-32768, Math.min(32767, f32[i] * 32767));
    ws.send(i16.buffer);
  };
  src.connect(scriptNode);
  scriptNode.connect(recCtx.destination);
}

function stopRec() {
  document.getElementById('rec').classList.remove('on');
  hint('Processing...');
  if (scriptNode) { scriptNode.disconnect(); scriptNode = null; }
  if (recCtx)     { recCtx.close(); recCtx = null; }
  if (micStream)  { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (ws && ws.readyState === 1) ws.send('END');
}

// Keyword Q&A eval
async function runEval() {
  const keyword = document.getElementById('ek').value.trim();
  if (!keyword) { alert('Enter a keyword to describe/guess.'); return; }
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.textContent = 'Running...';
  document.getElementById('eval-log').innerHTML = '';
  document.getElementById('eval-summary').style.display = 'none';

  const body = {
    keyword,
    max_turns: parseInt(document.getElementById('et').value) || 10,
    describer_prompt: document.getElementById('edp').value.trim(),
    guesser_prompt:   document.getElementById('egp').value.trim(),
  };

  try {
    const r = await fetch(httpBase() + '/api/eval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error('Server error ' + r.status);
    showEvalResult(await r.json());
  } catch (e) {
    alert('Eval failed: ' + e.message);
  } finally {
    btn.disabled = false; btn.textContent = 'Run Evaluation';
  }
}

function showEvalResult(data) {
  const log = document.getElementById('eval-log');
  (data.transcript || []).forEach((turn, i) => {
    const d = document.createElement('div');
    const isDescriber = turn.role === 'describer';
    d.className = 'em ' + (isDescriber ? 'a' : 'b');
    d.innerHTML = '<div class="role">' + turn.role + ' &middot; turn ' + (i+1) + '</div>'
                + (turn.text || '(audio only)');
    log.appendChild(d);
  });
  log.scrollTop = log.scrollHeight;
  const s = document.getElementById('eval-summary');
  s.style.display = 'block';
  const guessInfo = data.guessed
    ? 'guessed at turn ' + (data.guessed_at_turn + 1)
    : 'not guessed';
  s.textContent = 'keyword: \u201c' + data.keyword + '\u201d  \u00b7  '
                + data.turns + ' turns  \u00b7  ' + guessInfo;
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
ui_router = APIRouter()


@ui_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return _HTML


def mount_ui(app) -> None:
    """Register the SPA GET / route on an existing FastAPI app."""
    app.include_router(ui_router)
