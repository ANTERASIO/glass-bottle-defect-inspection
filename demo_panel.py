"""
Industrial Automated Optical Inspection (AOI) - HMI Demo Panel
Developed by: Prajwal Mahendrakar
Inspiration & References: Mrchuki/bottle-defect-detection & MVTec Anomaly Detection

Generates demo.html: a self-contained industrial HMI runtime panel
(Siemens TIA Portal / WinCC look & feel: gray bevels, gauge fields, alarms).
Runs the classical computer vision pipeline (ECC alignment -> Z-score tolerance difference -> verdict)
on every test sample, renders pixel-accurate heatmaps, and embeds everything into a standalone HTML.
"""


import base64
import glob
import json

import cv2
import numpy as np

THRESHOLD_Z = 6        # per-pixel: flag when difference > 6x the normal variation
THRESHOLD_PX = 500     # per-piece: NOK when more than 500 flagged pixels
CYCLE_MS = 2000        # panel auto-advance period


def align_ecc(reference, img):
    W = np.eye(2, 3, dtype=np.float32)
    cv2.findTransformECC(reference, img, W, cv2.MOTION_AFFINE,
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5))
    return cv2.warpAffine(img, W, img.shape[::-1],
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)


def smooth(img):
    return cv2.GaussianBlur(np.float32(img), (5, 5), 0)


def build_pattern():
    ref_paths = sorted(glob.glob("dataset/references/*.png"))
    base = cv2.imread(ref_paths[0], cv2.IMREAD_GRAYSCALE)
    stack = [smooth(base)] + [smooth(align_ecc(base, cv2.imread(p, cv2.IMREAD_GRAYSCALE)))
                              for p in ref_paths[1:]]
    return base, np.mean(stack, axis=0), np.maximum(np.std(stack, axis=0), 2)


def inspect(path, base, pattern, tolerance):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    z = np.abs(smooth(align_ecc(base, img)) - pattern) / tolerance
    mask = cv2.morphologyEx(np.uint8(z > THRESHOLD_Z) * 255, cv2.MORPH_OPEN,
                            np.ones((7, 7), np.uint8))
    return img, z, int((mask > 0).sum())


def heatmap_png(z):
    scaled = np.uint8(np.clip(z / 15, 0, 1) * 255)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_JET)


def to_b64(img):
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def main():
    base, pattern, tolerance = build_pattern()

    # interleave good and defective pieces so the demo alternates verdicts
    order = ["good_00", "broken_large_00", "contamination_00", "good_01",
             "broken_small_00", "contamination_01", "broken_large_01", "good_02",
             "broken_small_01", "contamination_02", "broken_large_02", "broken_small_02"]

    pieces = []
    for name in order:
        path = f"dataset/test/{name}.png"
        img, z, px = inspect(path, base, pattern, tolerance)
        pieces.append({
            "name": name,
            "px": px,
            "ok": px <= THRESHOLD_PX,
            "img": to_b64(img),
            "heat": to_b64(heatmap_png(z)),
        })
        print(f"  {name:20s} {px:6d} px -> {'OK' if px <= THRESHOLD_PX else 'NOK'}")

    html = TEMPLATE.replace("__DATA__", json.dumps(pieces))
    html = html.replace("__CYCLE__", str(CYCLE_MS))
    html = html.replace("__THRESHOLD__", str(THRESHOLD_PX))
    with open("demo.html", "w") as f:
        f.write(html)
    print("\nWritten demo.html — open it in a browser.")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>BottleVision — AOI Console</title>
<style>
:root {
  --bg-1: #06131b;
  --bg-2: #0b1d2b;
  --panel: rgba(12, 25, 38, 0.86);
  --panel-alt: rgba(18, 35, 47, 0.9);
  --panel-soft: rgba(19, 46, 61, 0.72);
  --border: rgba(122, 220, 255, 0.28);
  --border-strong: rgba(122, 220, 255, 0.5);
  --text: #eaf7ff;
  --muted: #9ec0d0;
  --header: linear-gradient(135deg, #122d40 0%, #0b1d2d 40%, #0e2f3d 100%);
  --accent: #5fe3ff;
  --accent-2: #7ef0b8;
  --ok: #28d17c;
  --ok-strong: #1ca968;
  --nok: #ff5b61;
  --nok-strong: #d82839;
  --idle: #7f93a0;
  --shadow: 0 20px 45px rgba(0, 0, 0, 0.42);
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; }
body {
  background:
    radial-gradient(circle at top left, rgba(95, 227, 255, 0.12), transparent 32%),
    radial-gradient(circle at bottom right, rgba(126, 240, 184, 0.08), transparent 28%),
    linear-gradient(180deg, var(--bg-1), var(--bg-2));
  color: var(--text);
  font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
  font-size: 13px;
  display: flex; flex-direction: column;
  overflow: hidden;
  user-select: none;
}

.raised { background: linear-gradient(180deg, rgba(24, 46, 59, 0.96), rgba(12, 25, 38, 0.96)); border: 1px solid var(--border); box-shadow: inset 0 1px 0 rgba(255,255,255,0.08); }
.sunken { background: rgba(7, 18, 25, 0.8); border: 1px solid rgba(110, 161, 189, 0.3); }

.titlebar {
  background: var(--header);
  color: #f5fbff; display: flex; align-items: center;
  padding: 12px 18px; gap: 14px;
  border-bottom: 1px solid rgba(95, 227, 255, 0.3);
  box-shadow: inset 0 -1px 0 rgba(255,255,255,0.08);
}
.titlebar .screen {
  font-weight: 700; font-size: 15px; letter-spacing: 0.1em; text-transform: uppercase;
  color: #dff9ff;
}
.titlebar .fill { flex: 1; }
.titlebar .dt {
  background: rgba(8, 17, 25, 0.9);
  color: var(--accent);
  border: 1px solid rgba(95, 227, 255, 0.3);
  border-radius: 999px; padding: 6px 12px; font-family: 'Consolas', 'Courier New', monospace; font-weight: 700; font-size: 12px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}

main { flex: 1; display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1.2fr) 320px; gap: 14px; padding: 14px; min-height: 0; }

.group {
  position: relative; background: linear-gradient(180deg, rgba(18, 35, 47, 0.95), rgba(12, 25, 38, 0.95));
  border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow);
  padding: 18px 12px 12px; min-height: 0; display: flex; flex-direction: column; gap: 10px;
}
.group > .caption {
  position: absolute; top: 8px; left: 14px; right: 14px;
  font-size: 11px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;
}

.imgbox { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; padding: 8px; border-radius: 14px; background: linear-gradient(180deg, rgba(9, 16, 20, 0.9), rgba(5, 13, 18, 0.95)); }
.imgbox img { max-width: 100%; max-height: 100%; display: block; border-radius: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.28); }

aside { display: flex; flex-direction: column; gap: 14px; min-height: 0; }

.lamp {
  height: 92px; display: flex; align-items: center; justify-content: center;
  font-size: 36px; font-weight: 800; letter-spacing: 0.08em; color: #fff;
  border-radius: 18px; border: 1px solid rgba(255,255,255,0.2); box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 8px 18px rgba(0,0,0,0.28);
}
.lamp.ok  { background: linear-gradient(180deg, var(--ok), var(--ok-strong)); }
.lamp.nok { background: linear-gradient(180deg, var(--nok), var(--nok-strong)); animation: pulse 1.1s ease-in-out infinite; }
.lamp.idle { background: linear-gradient(180deg, var(--idle), #596b77); color: #dfeaf2; font-size: 20px; }
@keyframes pulse { 0%, 100% { box-shadow: 0 0 0 rgba(255,91,97,0), inset 0 1px 0 rgba(255,255,255,0.18); } 50% { box-shadow: 0 0 22px rgba(255,91,97,0.7), inset 0 1px 0 rgba(255,255,255,0.18); } }

.iorow { display: flex; align-items: center; gap: 8px; }
.iorow label { width: 92px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.iofield {
  flex: 1; padding: 7px 10px; font-family: 'Consolas', 'Courier New', monospace; font-weight: 700; font-size: 13px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; border-radius: 10px;
}
.iofield.num { text-align: right; }
.iofield.alarmtext { color: var(--nok); }

.counters { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.counter {
  text-align: center; padding: 10px 8px; border-radius: 12px; background: rgba(15, 25, 34, 0.72); border: 1px solid rgba(122,220,255,0.18);
}
.counter .clabel { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
.counter .cval { font-family: 'Consolas', 'Courier New', monospace; font-weight: 800; font-size: 24px; margin-top: 4px; }
.counter .cval.pass { color: var(--ok); }
.counter .cval.fail { color: var(--nok); }

.cyclebar { height: 18px; position: relative; border-radius: 999px; overflow: hidden; }
.cyclebar i { position: absolute; inset: 0; background: linear-gradient(90deg, var(--accent), var(--accent-2)); border-radius: 999px; transform-origin: left; transform: scaleX(0); box-shadow: 0 0 10px rgba(95, 227, 255, 0.7); }
.cyclebar.running i { animation: cfill __CYCLE__ms linear infinite; }
@keyframes cfill { from { transform: scaleX(0); } to { transform: scaleX(1); } }

.alarmview { margin: 0 10px; }
.alarmview table { width: 100%; border-collapse: collapse; background: rgba(5, 13, 18, 0.85); font-size: 12px; border-radius: 12px; overflow: hidden; }
.alarmview th {
  background: rgba(16, 34, 46, 0.95); text-align: left; font-weight: 700; color: var(--muted);
  border: 1px solid rgba(122, 220, 255, 0.2); padding: 6px 8px; letter-spacing: 0.05em; text-transform: uppercase;
}
.alarmview td { border: 1px solid rgba(122, 220, 255, 0.14); padding: 5px 8px; font-family: 'Consolas', 'Courier New', monospace; height: 24px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.alarmview tr.fail td { color: #ffc3c8; font-weight: 700; }
.alarmview tr.pass td { color: #baf7d8; }

.fkeys { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 8px; padding: 10px 12px 12px; }
.fkey {
  padding: 10px 6px 9px; text-align: center; cursor: pointer; font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
  border-radius: 12px; background: linear-gradient(180deg, rgba(24, 46, 59, 0.96), rgba(12, 25, 38, 0.96)); border: 1px solid var(--border);
  transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}
.fkey small { display: block; font-weight: 600; font-size: 9px; color: var(--muted); letter-spacing: 0.06em; margin-top: 4px; }
.fkey:hover { transform: translateY(-1px); border-color: var(--border-strong); }
.fkey:active, .fkey.pressed { transform: translateY(1px); background: linear-gradient(180deg, rgba(17, 33, 42, 0.96), rgba(10, 19, 28, 0.96)); }
.fkey.disabled { opacity: 0.45; cursor: default; }
</style>
</head>
<body>

<div class="titlebar">
  <span class="screen">BottleVision / AOI Console</span>
  <span class="fill"></span>
  <span class="dt" id="date">--.--.----</span>
  <span class="dt" id="clock">--:--:--</span>
</div>

<main>
  <div class="group">
    <span class="caption">Live inspection feed</span>
    <div class="iorow"><label>Piece file</label><div class="iofield sunken" id="cam-name">—</div></div>
    <div class="imgbox sunken"><img id="cam-img" alt="test piece"></div>
  </div>

  <div class="group">
    <span class="caption">Defect map / z-score overlay</span>
    <div class="iorow"><label>Scale</label><div class="iofield sunken">JET&nbsp;&nbsp;z = 0 … 15</div></div>
    <div class="imgbox sunken"><img id="heat-img" alt="anomaly heat map"></div>
  </div>

  <aside>
    <div class="group">
      <span class="caption">Result</span>
      <div class="lamp idle" id="lamp">WAIT</div>
      <div class="iorow"><label>Piece</label><div class="iofield sunken" id="pname">—</div></div>
      <div class="iorow"><label>Defect px</label><div class="iofield sunken num" id="ppx">0</div></div>
      <div class="iorow"><label>Limit</label><div class="iofield sunken num">__THRESHOLD__</div></div>
    </div>

    <div class="group">
      <span class="caption">Counters</span>
      <div class="counters">
        <div class="counter sunken"><div class="clabel">Inspected</div><div class="cval" id="t-total">0</div></div>
        <div class="counter sunken"><div class="clabel">Pass</div><div class="cval pass" id="t-pass">0</div></div>
        <div class="counter sunken"><div class="clabel">Fail</div><div class="cval fail" id="t-fail">0</div></div>
      </div>
    </div>

    <div class="group">
      <span class="caption">Cycle</span>
      <div class="iorow"><label>Mode</label><div class="iofield sunken" id="mode">AUTO</div></div>
      <div class="iorow"><label>Piece</label><div class="iofield sunken num" id="pidx">— / —</div></div>
      <div class="cyclebar sunken running" id="cyclebar"><i></i></div>
    </div>
  </aside>
</main>

<div class="alarmview">
  <table>
    <thead><tr><th style="width:70px">No.</th><th style="width:90px">Time</th><th>Message text</th><th style="width:110px">Status</th></tr></thead>
    <tbody id="alarms"></tbody>
  </table>
</div>

<div class="fkeys">
  <div class="fkey raised" id="f1">F1<small>START</small></div>
  <div class="fkey raised" id="f2">F2<small>STOP</small></div>
  <div class="fkey raised" id="f3">F3<small>PREV</small></div>
  <div class="fkey raised" id="f4">F4<small>NEXT</small></div>
  <div class="fkey raised disabled">F5<small></small></div>
  <div class="fkey raised disabled">F6<small></small></div>
  <div class="fkey raised disabled">F7<small></small></div>
  <div class="fkey raised disabled">F8<small></small></div>
</div>

<script>
const DATA = __DATA__;
const CYCLE = __CYCLE__;
let i = -1, total = 0, pass = 0, fail = 0, msgno = 0, timer = null;

const $ = id => document.getElementById(id);
const pad = n => String(n).padStart(2, '0');
const ALARM_ROWS = 5;

/* pre-fill the alarm view so it starts at its final size */
for (let k = 0; k < ALARM_ROWS; k++) {
  const tr = document.createElement('tr');
  tr.innerHTML = '<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td>';
  document.getElementById('alarms').appendChild(tr);
}

function show(k) {
  const p = DATA[k];
  $('cam-img').src  = 'data:image/png;base64,' + p.img;
  $('heat-img').src = 'data:image/png;base64,' + p.heat;
  $('cam-name').textContent = p.name + '.png';
  $('pname').textContent = p.name;
  $('ppx').textContent = p.px.toLocaleString('en');
  $('pidx').textContent = (k + 1) + ' / ' + DATA.length;
  const lamp = $('lamp');
  lamp.className = 'lamp ' + (p.ok ? 'ok' : 'nok');
  lamp.textContent = p.ok ? 'OK' : 'NOK';
}

function logRow(p) {
  msgno++;
  const d = new Date();
  const tr = document.createElement('tr');
  tr.className = p.ok ? 'pass' : 'fail';
  tr.innerHTML = '<td>' + String(msgno).padStart(4, '0') + '</td>' +
    '<td>' + pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds()) + '</td>' +
    '<td>' + (p.ok ? 'Piece ' + p.name + ' inspected — no defect'
                   : 'DEFECT DETECTED — piece ' + p.name + ' (' + p.px.toLocaleString('en') + ' px over limit)') + '</td>' +
    '<td>' + (p.ok ? 'PASS' : 'FAIL') + '</td>';
  const tb = $('alarms');
  tb.insertBefore(tr, tb.firstChild);
  while (tb.children.length > ALARM_ROWS) tb.removeChild(tb.lastChild);
}

function step(dir = 1) {
  i = (i + dir + DATA.length) % DATA.length;
  const p = DATA[i];
  total++; p.ok ? pass++ : fail++;
  $('t-total').textContent = total; $('t-pass').textContent = pass; $('t-fail').textContent = fail;
  logRow(p);
  show(i);
}

function start() {
  if (timer) return;
  $('mode').textContent = 'AUTO';
  $('cyclebar').classList.add('running');
  step(); timer = setInterval(step, CYCLE);
}
function stop() {
  clearInterval(timer); timer = null;
  $('mode').textContent = 'MANUAL';
  $('cyclebar').classList.remove('running');
}

$('f1').onclick = start;
$('f2').onclick = stop;
$('f3').onclick = () => { stop(); step(-1); };
$('f4').onclick = () => { stop(); step(1); };
document.addEventListener('keydown', e => {
  const map = { F1: 'f1', F2: 'f2', F3: 'f3', F4: 'f4' };
  if (map[e.key]) { e.preventDefault(); $(map[e.key]).click(); }
});

/* static-frame mode for video capture: demo.html?frame=4 renders the exact,
   deterministic state after inspecting pieces 0..4 and freezes */
function renderStatic(k) {
  let p = 0, f = 0;
  for (let j = 0; j <= k; j++) DATA[j].ok ? p++ : f++;
  $('t-total').textContent = k + 1; $('t-pass').textContent = p; $('t-fail').textContent = f;
  const base = 10 * 3600;  // fake shift clock starting at 10:00:00, one piece every 2 s
  const t = s => pad(Math.floor(s / 3600)) + ':' + pad(Math.floor(s / 60) % 60) + ':' + pad(s % 60);
  const tb = $('alarms');
  tb.innerHTML = '';
  for (let j = k; j >= 0 && tb.children.length < ALARM_ROWS; j--) {
    const q = DATA[j];
    const tr = document.createElement('tr');
    tr.className = q.ok ? 'pass' : 'fail';
    tr.innerHTML = '<td>' + String(j + 1).padStart(4, '0') + '</td><td>' + t(base + 2 * j) + '</td>' +
      '<td>' + (q.ok ? 'Piece ' + q.name + ' inspected — no defect'
                     : 'DEFECT DETECTED — piece ' + q.name + ' (' + q.px.toLocaleString('en') + ' px over limit)') + '</td>' +
      '<td>' + (q.ok ? 'PASS' : 'FAIL') + '</td>';
    tb.appendChild(tr);
  }
  while (tb.children.length < ALARM_ROWS) {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td><td>&nbsp;</td>';
    tb.appendChild(tr);
  }
  const d = new Date();
  $('clock').textContent = t(base + 2 * k);
  $('date').textContent = pad(d.getDate()) + '.' + pad(d.getMonth() + 1) + '.' + d.getFullYear();
  const bar = document.querySelector('#cyclebar i');
  bar.style.animation = 'none';
  bar.style.transform = 'scaleX(0.4)';
  show(k);
}

const qp = new URLSearchParams(location.search);
if (qp.has('frame')) {
  renderStatic(parseInt(qp.get('frame'), 10));
} else {
  setInterval(() => {
    const d = new Date();
    $('clock').textContent = pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
    $('date').textContent = pad(d.getDate()) + '.' + pad(d.getMonth() + 1) + '.' + d.getFullYear();
  }, 500);
  start();
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
