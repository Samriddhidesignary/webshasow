// ================================================================
//  FINAL EXHIBITION VERSION - "Web Creature"  (everything lives in this one file)
//
//  WEBCAM -> MediaPipe Pose Landmarker -> smoothed landmarks
//        -> INVISIBLE CONTROL SKELETON + GESTURES (jump / fast move / both arms up)
//        -> springs -> creature rig (main nodes)
//        -> NETWORK: secondary nodes + connections, with waves travelling through it -> glowing picture
//
//  Keys:  D = development mode on/off (camera preview, skeleton, FPS, status text)
//         F = fullscreen on/off   (double-click also works)
//
//  EXHIBITION MODE (default): only the creature. DEVELOPMENT MODE (press D): debug tools.
//
//  FRAMING: a "virtual camera" scales the creature to fill the screen (portrait TV or landscape),
//  see CONFIG.fillHeight and section 4 (getFrameBox / resizeCanvas / updateCamera).
//
//  NOTE: MediaPipe is loaded from the CDN while the app starts (see loadMediaPipe),
//  so this file has NO npm imports and the project always compiles.
// ================================================================
import './style.css';

// ----------------------------------------------------------------
// 1. SETTINGS  (numbers you may want to tweak)
// ----------------------------------------------------------------
const MEDIAPIPE_VERSION = '0.10.14'; // keep identical to package.json
const CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;

const CONFIG = {
  moduleUrl: `${CDN}/vision_bundle.mjs`,
  wasmUrl: `${CDN}/wasm`,
  modelUrl:
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',

  // Landmark smoothing: 1 = instant/jittery, 0.1 = very smooth/laggy.
  smoothing: 0.5,
  minVisibility: 0.5, // landmarks less visible than this are ignored by the debug skeleton
  lostAfterMs: 500, // a visitor counts as "gone" after this long without detection
  holdAfterLeaveMs: 1500, // when the visitor leaves, the creature holds their last pose this long, then relaxes to idle

  startWithDebug: false, // false = exhibition mode on start (press D for development mode)

  // Picture
  maxPixelRatio: 1, // 1 = lightest. Raise to 1.5 on a stronger PC for a sharper picture.
  particles: 120, // max number of floating lights / rings

  // Framing (virtual camera): how big the creature is on the screen
  fillHeight: 0.87, // resting creature = this fraction of the screen height (0.85 - 0.95). Higher = bigger, less room for raised hands
  maxBodyWidthFill: 0.96, // safety for very narrow screens: the body (arms down) never gets wider than this fraction of the screen width
  jumpKick: 1300, // strength of the jump bounce (it is reduced automatically if the screen has little free room above the head)

  // Body -> creature mapping
  moveGain: 1.5, // how far the creature travels when the visitor walks sideways
  verticalGain: 1.3, // how much the creature rises when the visitor goes up

  // Gesture thresholds
  movingThreshold: 0.2, // motion level for "person moving"
  fastThreshold: 0.9, // motion level for "fast movement" (scatter reaction)
  jumpSpeed: 0.4, // upward hip speed (frame-heights per second) that counts as a jump
  bloomHoldSeconds: 0.2, // both arms must stay up this long to trigger the bloom
};

// Virtual drawing world. The creature lives in these "world" units; the virtual camera (section 4) decides how
// many screen pixels one world unit is, so the creature can fill a portrait TV, a landscape screen or anything else.
const W = 1600;
const H = 900;
const BASE_Y = 480; // world Y of the creature's body centre when standing (the feet end just above GROUND_Y)
const GROUND_Y = 836; // world Y of the shadow on the "floor"

// MediaPipe landmark numbers used by the control skeleton
const LM = {
  NOSE: 0,
  L_SHOULDER: 11,
  R_SHOULDER: 12,
  L_ELBOW: 13,
  R_ELBOW: 14,
  L_WRIST: 15,
  R_WRIST: 16,
  L_HIP: 23,
  R_HIP: 24,
  L_KNEE: 25,
  R_KNEE: 26,
  L_ANKLE: 27,
  R_ANKLE: 28,
};

// Groups + lines, used only to draw the debug skeleton
const HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const ARMS = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
const TORSO = [11, 12, 23, 24];
const CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [15, 17],
  [15, 19],
  [15, 21],
  [17, 19],
  [16, 18],
  [16, 20],
  [16, 22],
  [18, 20],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 29],
  [29, 31],
  [27, 31],
  [28, 30],
  [30, 32],
  [28, 32],
];

// ----------------------------------------------------------------
// 2. PAGE ELEMENTS + STATE
// ----------------------------------------------------------------
const canvas = document.getElementById('stage');
const ctx = canvas.getContext('2d');
const video = document.getElementById('cam');
const debugPanel = document.getElementById('debugPanel');
const statusBox = document.getElementById('status');

const state = {
  camera: 'Starting...', // 'Starting...' | 'Ready' | 'Error'
  model: 'Not loaded', // 'Not loaded' | 'Loading...' | 'Ready' | 'Error'
  delegate: '-',
  poseDetected: false,
  visibleCount: 0,
  fps: 0,
  inferMs: 0,
};

// One smoothed point per landmark. x, y are 0..1 in camera coordinates (NOT mirrored).
// speed = how fast the point moves (frame-heights per second), vy = vertical speed (negative = up).
const smoothed = Array.from({ length: 33 }, () => ({
  x: 0.5,
  y: 0.5,
  visibility: 0,
  speed: 0,
  vy: 0,
  ready: false,
}));
let lastSeenMs = -1e9;

let poseLandmarker = null;
let debugOn = CONFIG.startWithDebug;

// ----------------------------------------------------------------
// 3. SMALL HELPERS + SPRINGS
// ----------------------------------------------------------------
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const rand = (lo, hi) => lo + Math.random() * (hi - lo);
const pick = (list) => list[Math.floor(Math.random() * list.length)];
const smoothstep = (t) => {
  t = clamp(t, 0, 1);
  return t * t * (3 - 2 * t);
};
function wrapAngle(a) {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}
// Move "current" toward "target" at a fixed speed per second (different speeds up / down).
function approach(current, target, rateUp, rateDown, dt) {
  const rate = target > current ? rateUp : rateDown;
  return current + clamp(target - current, -rate * dt, rate * dt);
}

// A spring chases a target with a little delay and overshoot, so motion feels alive, not rigid.
class Spring {
  constructor(value = 0, stiffness = 100, damping = 12) {
    this.pos = value;
    this.vel = 0;
    this.k = stiffness;
    this.d = damping;
  }
  update(target, dt) {
    this.vel += (this.k * (target - this.pos) - this.d * this.vel) * dt;
    this.pos += this.vel * dt;
  }
}
// Same for angles: always takes the short way round (no crazy 360-degree spins).
class AngleSpring extends Spring {
  update(target, dt) {
    super.update(this.pos + wrapAngle(target - this.pos), dt);
  }
}

// ----------------------------------------------------------------
// 4. CANVAS SIZE, RESPONSIVE LAYOUT + BACKGROUND
// ----------------------------------------------------------------
const view = {
  scale: 1, // screen pixels per world unit (one uniform scale: the creature is never stretched)
  offsetX: 0,
  offsetY: 0, // screen position of world (0, 0), without the soft camera pan
  pan: 0, // soft vertical camera pan in world units (picture moves down by this much), see updateCamera()
  maxPan: 0, // the most the camera may pan (keeps the feet on screen)
  zoom: 1, // 1 = normal. Slightly below 1 only while raised hands / the bloom need more room above the head (see updateCamera)
  visH: H, // visible height in world units
  topMargin: 0, // empty space above the resting creature (world units)
  topPad: 0, // small safety gap kept above raised hands (world units)
  restAbove: 0, // distance from the body centre to the top of the head at rest (world units)
  maxRise: 0, // how far the creature may rise when the visitor goes up (world units)
  hopScale: 1, // 1 = full jump bounce, smaller = lower jump (little room above the head)
  minX: 0, // creature's sideways travel limits (world units)
  maxX: W,
  background: null,
  worldLeft: 0,
  worldRight: W,
  worldTop: 0,
  worldBottom: H, // part of the world that is visible
  winW: 0,
  winH: 0, // window size used for the last layout (the main loop re-checks it)
};

// ---- THE CREATURE'S FRAME (step 1 of the virtual camera) ----
// The resting silhouette of the creature in WORLD units, taken from the design numbers (section 9), so it
// follows CREATURE_SCALE automatically: top of the head, soles of the feet and the half width of the body
// with the arms hanging (slightly splayed outwards, the way they look while following a visitor).
function getFrameBox() {
  const s = CREATURE_SCALE;
  const HEAD_TOP = 95; // head mesh radius above the head centre (design units)
  const FOOT_DROP = 28; // sole of the foot below the ankle (design units)
  const top = BASE_Y + (DESIGN.neck.y + DESIGN.headOffset.y - HEAD_TOP) * s;
  const bottom =
    BASE_Y + (DESIGN.hip.L.y + DESIGN.leg.up + DESIGN.leg.low + FOOT_DROP) * s;
  const halfWidth =
    (DESIGN.shoulder.R.x +
      Math.sin(0.38) * DESIGN.arm.up +
      Math.sin(0.32) * DESIGN.arm.low +
      Math.sin(0.32) * DESIGN.arm.hand) *
    s;
  return { top, bottom, halfWidth, restAbove: BASE_Y - top };
}

// How high does the jump bounce go with a given kick? A tiny simulation of the same spring (run on every resize, costs nothing).
function jumpPeak(kick) {
  const sp = new Spring(0, creature.hop.k, creature.hop.d);
  sp.vel = kick;
  let peak = 0;
  for (let i = 0; i < 120; i++) {
    sp.update(0, 1 / 60);
    peak = Math.max(peak, sp.pos);
  }
  return peak;
}

// ---- VIRTUAL CAMERA ----
// 1. take the creature's frame box   2. read the real canvas size   3. one uniform scale so the creature fills
// CONFIG.fillHeight of the screen   4. centre it   5. every drawing (nodes, threads, glow) uses this one transform,
// 6. margins above / below are kept free for raised hands.
// Runs at start, on every window resize and when fullscreen is switched on or off.
function resizeCanvas() {
  const ratio = Math.min(window.devicePixelRatio || 1, CONFIG.maxPixelRatio);
  canvas.width = Math.floor(window.innerWidth * ratio);
  canvas.height = Math.floor(window.innerHeight * ratio);
  view.winW = window.innerWidth;
  view.winH = window.innerHeight;
  if (canvas.width < 2 || canvas.height < 2) return; // minimised window: nothing to lay out

  const box = getFrameBox();
  const boxH = box.bottom - box.top;

  // Scale: portrait or landscape, the creature is as tall as CONFIG.fillHeight allows. The width limit only
  // matters for extremely narrow screens (the body must always fit).
  let scale = (CONFIG.fillHeight * canvas.height) / boxH;
  scale = Math.min(
    scale,
    (CONFIG.maxBodyWidthFill * canvas.width) / (box.halfWidth * 2)
  );
  view.scale = scale;

  // Centre: horizontally on the creature's idle position, vertically on the middle of its silhouette.
  view.offsetX = canvas.width / 2 - (W / 2) * scale;
  view.offsetY = canvas.height / 2 - ((box.top + box.bottom) / 2) * scale;
  view.worldLeft = -view.offsetX / scale;
  view.worldRight = (canvas.width - view.offsetX) / scale;
  view.worldTop = -view.offsetY / scale;
  view.worldBottom = (canvas.height - view.offsetY) / scale;

  // Free room above and below the resting creature, and the limits that follow from it.
  const visW = view.worldRight - view.worldLeft;
  const visH = view.worldBottom - view.worldTop;
  const bottomMargin = view.worldBottom - box.bottom;
  view.visH = visH;
  view.topMargin = box.top - view.worldTop;
  view.topPad = visH * 0.02;
  view.restAbove = box.restAbove;
  view.maxPan = Math.max(0, bottomMargin - visH * 0.012);
  view.maxRise = Math.max(25, view.topMargin - view.topPad);
  // a jump goes about as high as the free room above the head (so the head stays on screen), never above the original height
  view.hopScale = clamp(
    (view.topMargin * 0.75) / jumpPeak(CONFIG.jumpKick),
    0.3,
    1
  );
  // sideways travel: the body (arms hanging) always stays fully on screen (on landscape screens about the same share of the width as before)
  const margin = Math.min(
    Math.max(box.halfWidth * 1.05, visW * 0.1375),
    visW / 2 - 40
  );
  view.minX = view.worldLeft + margin;
  view.maxX = view.worldRight - margin;

  view.background = buildBackground(canvas.width, canvas.height);
}

// Dark stage for the light, drawn ONCE (not every frame): deep indigo, soft colour clouds, floor haze, vignette.
function buildBackground(w, h) {
  const bg = document.createElement('canvas');
  bg.width = w;
  bg.height = h;
  const g = bg.getContext('2d');
  const big = Math.max(w, h);

  const sky = g.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#04060f');
  sky.addColorStop(0.6, '#0a0d24');
  sky.addColorStop(1, '#140f33');
  g.fillStyle = sky;
  g.fillRect(0, 0, w, h);

  // big, very soft colour clouds in the dark: [x, y, radius (fraction of the window), colour, strength]
  const blobs = [
    [0.15, 0.2, 0.34, '90,70,255', 0.16],
    [0.86, 0.15, 0.3, '255,60,200', 0.09],
    [0.72, 0.62, 0.36, '40,120,255', 0.12],
    [0.1, 0.72, 0.3, '140,90,255', 0.12],
    [0.5, 0.45, 0.4, '70,60,200', 0.12],
  ];
  for (const [fx, fy, fr, rgb, strength] of blobs) {
    const cx = fx * w,
      cy = fy * h,
      r = fr * big;
    const gr = g.createRadialGradient(cx, cy, 0, cx, cy, r);
    gr.addColorStop(0, `rgba(${rgb},${strength})`);
    gr.addColorStop(1, `rgba(${rgb},0)`);
    g.fillStyle = gr;
    g.fillRect(cx - r, cy - r, r * 2, r * 2);
  }

  // a faint haze of light on the floor
  const horizon = view.offsetY + (GROUND_Y - 70) * view.scale;
  const floor = g.createLinearGradient(0, horizon, 0, h);
  floor.addColorStop(0, 'rgba(100,80,255,0)');
  floor.addColorStop(1, 'rgba(100,80,255,0.14)');
  g.fillStyle = floor;
  g.fillRect(0, horizon, w, h - horizon);

  // vignette keeps the eye on the creature
  const vg = g.createRadialGradient(
    w / 2,
    h / 2,
    Math.min(w, h) * 0.4,
    w / 2,
    h / 2,
    big * 0.75
  );
  vg.addColorStop(0, 'rgba(0,0,0,0)');
  vg.addColorStop(1, 'rgba(0,0,0,0.5)');
  g.fillStyle = vg;
  g.fillRect(0, 0, w, h);
  return bg;
}
window.addEventListener('resize', resizeCanvas);
document.addEventListener('fullscreenchange', resizeCanvas); // fullscreen on / off -> new viewport size
// (the first resizeCanvas() call is at the very end of this file, after the creature's design numbers exist)

// Debug skeleton only: the camera picture is fitted inside the canvas so proportions stay correct.
function getVideoRect() {
  const videoAspect =
    video.videoWidth && video.videoHeight
      ? video.videoWidth / video.videoHeight
      : 16 / 9;
  const canvasAspect = canvas.width / canvas.height;
  let w, h;
  if (canvasAspect > videoAspect) {
    h = canvas.height;
    w = h * videoAspect;
  } else {
    w = canvas.width;
    h = w / videoAspect;
  }
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}
// MediaPipe (0..1) -> canvas pixels, MIRRORED (1 - x) so it behaves like a mirror.
function toCanvas(point, rect) {
  return { x: rect.x + (1 - point.x) * rect.w, y: rect.y + point.y * rect.h };
}

// ----------------------------------------------------------------
// 5. STATUS MESSAGES
//    Exhibition mode: only ERRORS are shown (e.g. camera denied). Development mode shows everything.
// ----------------------------------------------------------------
let statusText = '';
let statusIsError = false;
function setStatus(message, isError = false) {
  statusText = message;
  statusIsError = isError;
  refreshStatus();
}
function hideStatus() {
  statusText = '';
  refreshStatus();
}
function refreshStatus() {
  const show = statusText !== '' && (statusIsError || debugOn);
  statusBox.textContent = statusText;
  statusBox.classList.toggle('error', statusIsError);
  statusBox.classList.toggle('hidden', !show);
}

// ----------------------------------------------------------------
// 6. CAMERA   (unchanged from Phase 1/2). The video is only an input for MediaPipe, never shown.
// ----------------------------------------------------------------
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Camera API not available (page must run on https or localhost).'
    );
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30 },
      facingMode: 'user',
    },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => {
    video.onloadedmetadata = resolve;
  });
  await video.play();
  stream.getVideoTracks()[0].addEventListener('ended', () => {
    state.camera = 'Error';
    setStatus('Camera disconnected.', true);
  });
}

function describeCameraError(err) {
  const tip =
    ' If you are in the StackBlitz preview pane, open the preview in a NEW TAB.';
  switch (err && err.name) {
    case 'NotAllowedError':
      return (
        'Camera permission denied. Allow camera access (camera icon in the address bar) and reload.' +
        tip
      );
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera found. Plug in a webcam and reload.';
    case 'NotReadableError':
      return 'The camera is in use by another app. Close it and reload.';
    default:
      return (
        'Could not start the camera: ' +
        (err && err.message ? err.message : err) +
        tip
      );
  }
}

// ----------------------------------------------------------------
// 7. MEDIAPIPE POSE LANDMARKER INITIALISATION   (unchanged from Phase 1/2)
// ----------------------------------------------------------------
async function loadMediaPipe() {
  // @vite-ignore: this is loaded by the browser from the CDN, not bundled by Vite.
  const { PoseLandmarker, FilesetResolver } = await import(
    /* @vite-ignore */ CONFIG.moduleUrl
  );
  const fileset = await FilesetResolver.forVisionTasks(CONFIG.wasmUrl);

  const makeOptions = (delegate) => ({
    baseOptions: { modelAssetPath: CONFIG.modelUrl, delegate },
    runningMode: 'VIDEO', // we feed it webcam frames
    numPoses: 1, // track one visitor
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  try {
    poseLandmarker = await PoseLandmarker.createFromOptions(
      fileset,
      makeOptions('GPU')
    );
    state.delegate = 'GPU';
  } catch (gpuError) {
    console.warn('GPU not available, using CPU instead:', gpuError);
    poseLandmarker = await PoseLandmarker.createFromOptions(
      fileset,
      makeOptions('CPU')
    );
    state.delegate = 'CPU';
  }
}

// ----------------------------------------------------------------
// 8. POSE PROCESSING: detection + smoothing (+ speeds, used by the gesture detection)
// ----------------------------------------------------------------
let lastVideoTime = -1;
let lastDetectMs = 0;
let detectGapMs = 0; // extra pause between detections. Stays 0 unless the PC is slow.

function detectPose(nowMs) {
  if (!poseLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return; // no new camera frame yet
  if (nowMs - lastDetectMs < detectGapMs) return; // slow PC: leave time for drawing
  lastVideoTime = video.currentTime;
  const dtSec = clamp((nowMs - lastDetectMs) / 1000, 0.01, 0.5);
  lastDetectMs = nowMs;

  let result;
  const startMs = performance.now();
  try {
    result = poseLandmarker.detectForVideo(video, nowMs);
  } catch (err) {
    console.warn('Detection error (ignored):', err);
    return;
  }
  // Only if detection itself is slow (> ~20 ms) do we detect a little less often so drawing stays smooth.
  state.inferMs = lerp(state.inferMs, performance.now() - startMs, 0.2);
  detectGapMs = clamp(state.inferMs * 1.8, 0, 70);

  if (result.landmarks && result.landmarks.length > 0) {
    // If the person was gone for a while, jump straight to the new position (no flying in).
    if (nowMs - lastSeenMs > CONFIG.lostAfterMs) {
      for (const p of smoothed) p.ready = false;
      control.needsBaseline = true; // measure "standing height" again for the new visitor
    }
    updateSmoothedLandmarks(result.landmarks[0], dtSec);
    lastSeenMs = nowMs;
  }
}

// SMOOTHING: each point moves a fraction (CONFIG.smoothing) of the way toward the new MediaPipe value.
// We also measure how fast every point moves, which is what the gestures (fast move, jump) use.
function updateSmoothedLandmarks(landmarks, dtSec) {
  const amount = CONFIG.smoothing;
  const aspect =
    video.videoWidth && video.videoHeight
      ? video.videoWidth / video.videoHeight
      : 4 / 3;
  for (let i = 0; i < 33; i++) {
    const raw = landmarks[i];
    const p = smoothed[i];
    if (!p.ready) {
      p.x = raw.x;
      p.y = raw.y;
      p.speed = 0;
      p.vy = 0;
      p.ready = true;
    } else {
      const nx = p.x + (raw.x - p.x) * amount;
      const ny = p.y + (raw.y - p.y) * amount;
      p.speed = lerp(
        p.speed,
        Math.hypot((nx - p.x) * aspect, ny - p.y) / dtSec,
        0.4
      );
      p.vy = lerp(p.vy, (ny - p.y) / dtSec, 0.4);
      p.x = nx;
      p.y = ny;
    }
    p.visibility = raw.visibility ?? 1;
  }
}

// ----------------------------------------------------------------
// 9. THE WEB CREATURE: PALETTE, SIZES, THE RIG
//    The creature is an abstract humanoid made of a living network: glowing nodes + thin elastic threads.
//    The MediaPipe body drives an invisible "rig" (springs). The rig's joints are the main nodes of the web.
// ----------------------------------------------------------------
const CREATURE_SCALE = 0.86; // the whole creature is drawn at this size (design units * scale = world pixels)
const DESIGN = {
  neck: { x: 0, y: -150 }, // the head pivots here
  headOffset: { x: 0, y: -88 }, // head centre relative to the neck
  shoulder: { L: { x: -66, y: -122 }, R: { x: 66, y: -122 } },
  hip: { L: { x: -34, y: 120 }, R: { x: 34, y: 120 } },
  arm: { up: 132, low: 126, hand: 56 },
  leg: { up: 124, low: 118 },
};
// Half-width of the torso at height y (design units): used to place the nodes of the chest and belly.
const TORSO_PROFILE = [
  [-165, 40],
  [-148, 60],
  [-122, 76],
  [-70, 72],
  [0, 56],
  [70, 58],
  [125, 70],
  [158, 58],
  [174, 36],
];
function torsoHalfWidth(y) {
  const P = TORSO_PROFILE;
  if (y <= P[0][0]) return P[0][1];
  for (let i = 1; i < P.length; i++) {
    if (y <= P[i][0])
      return lerp(
        P[i - 1][1],
        P[i][1],
        (y - P[i - 1][0]) / (P[i][0] - P[i - 1][0])
      );
  }
  return P[P.length - 1][1];
}

// ---- colour zones: head violet/pink, upper body blue/violet, arms blue/cyan, lower body violet/magenta ----
const ZONE = { HEAD: 0, UPPER: 1, ARM: 2, LOWER: 3 };
const ZONE_RAMPS = [
  ['#8b5cff', '#c46bff', '#ff7ad9'],
  ['#3b82ff', '#5f6dff', '#8b5cff'],
  ['#3b82ff', '#2fb4ff', '#33e1ff'],
  ['#8b5cff', '#c13dff', '#ff3dcb'],
];
const STEPS = 6; // every zone is a smooth gradient made of 6 colours (so lines can be drawn in a few batches)
const hexRgb = (h) => {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const rgbHex = (r, g, b) =>
  '#' +
  [r, g, b]
    .map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, '0'))
    .join('');
function rampAt(stops, t) {
  t = clamp(t, 0, 1);
  const f = t * (stops.length - 1),
    i = Math.min(stops.length - 2, Math.floor(f)),
    k = f - i;
  const a = hexRgb(stops[i]),
    b = hexRgb(stops[i + 1]);
  return rgbHex(lerp(a[0], b[0], k), lerp(a[1], b[1], k), lerp(a[2], b[2], k));
}
const PALETTE = [];
for (let z = 0; z < ZONE_RAMPS.length; z++)
  for (let s = 0; s < STEPS; s++)
    PALETTE.push(rampAt(ZONE_RAMPS[z], s / (STEPS - 1)));
const colorIndex = (zone, ramp) =>
  zone * STEPS + Math.round(clamp(ramp, 0, 1) * (STEPS - 1));

// a soft glow, painted once per colour and re-used (white-hot centre, coloured halo)
function makeGlowSprite(hex) {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d'),
    [r, gg, b] = hexRgb(hex);
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, 'rgba(255,255,255,0.95)');
  gr.addColorStop(0.14, `rgba(${r},${gg},${b},0.85)`);
  gr.addColorStop(0.5, `rgba(${r},${gg},${b},0.2)`);
  gr.addColorStop(1, `rgba(${r},${gg},${b},0)`);
  g.fillStyle = gr;
  g.fillRect(0, 0, 64, 64);
  return c;
}
const GLOW = PALETTE.map(makeGlowSprite);

// The design of the web is random, but with a fixed seed: it looks the same every time the page loads.
function mulberry32(seed) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const wrng = mulberry32(20240612);
const wr = (lo, hi) => lo + wrng() * (hi - lo);

// ----------------------------------------------------------------
// 10. THE RIG: invisible skeleton driven by springs (smooth, slightly springy)
// ----------------------------------------------------------------
const makeLimb = (up, low, k, d) => ({
  up: new AngleSpring(up, k, d),
  low: new AngleSpring(low, k, d),
});
const ARM_K = 85,
  ARM_D = 9;
const creature = {
  x: new Spring(W / 2, 55, 10),
  y: new Spring(0, 70, 11), // vertical offset (down is +)
  hop: new Spring(0, 45, 5.5), // jump / bounce (up is +)
  tilt: new Spring(0, 60, 9),
  bend: new Spring(0, 80, 12),
  headX: new Spring(0, 70, 10),
  headRot: new Spring(0, 40, 5.5),
  expand: new Spring(0, 70, 7),
  armL: makeLimb(-0.16, -0.1, ARM_K, ARM_D), // L / R = left / right side of the SCREEN
  armR: makeLimb(0.16, 0.1, ARM_K, ARM_D),
  legL: makeLimb(-0.07, -0.03, 130, 16),
  legR: makeLimb(0.07, 0.03, 130, 16),
  flopL: new Spring(0, 70, 6),
  flopR: new Spring(0, 70, 6), // hands trail a little behind the forearm
  lookX: 0,
  lookY: 0,
  blinkIn: 2,
  blinkTime: 0,
};

// sk = the creature's current joints in world coordinates (also read by the debug view). Re-used every frame.
const J = () => ({ x: 0, y: 0 });
const sk = {
  scale: CREATURE_SCALE,
  xf: { cx: 0, cy: 0, cos: 1, sin: 0, sx: 1, sy: 1 },
  head: { x: 0, y: 0, rot: 0 },
  neck: J(),
  chest: J(),
  pelvis: J(),
  arms: {
    L: { shoulder: J(), elbow: J(), wrist: J() },
    R: { shoulder: J(), elbow: J(), wrist: J() },
  },
  legs: {
    L: { hip: J(), knee: J(), ankle: J() },
    R: { hip: J(), knee: J(), ankle: J() },
  },
  feet: { L: J(), R: J() },
  handTip: { L: J(), R: J() },
};
// The 20 MAIN NODES of the web. The ids are used everywhere below.
const JOINTS = [
  sk.head,
  sk.neck,
  sk.arms.L.shoulder,
  sk.arms.R.shoulder,
  sk.arms.L.elbow,
  sk.arms.R.elbow,
  sk.arms.L.wrist,
  sk.arms.R.wrist,
  sk.handTip.L,
  sk.handTip.R,
  sk.chest,
  sk.pelvis,
  sk.legs.L.hip,
  sk.legs.R.hip,
  sk.legs.L.knee,
  sk.legs.R.knee,
  sk.legs.L.ankle,
  sk.legs.R.ankle,
  sk.feet.L,
  sk.feet.R,
];

// local body coordinates (design units) -> world coordinates (rotated by the body tilt, squashed when bending)
function toWorld(lx, ly, out) {
  const f = sk.xf;
  out.x = f.cx + f.cos * lx * f.sx - f.sin * ly * f.sy;
  out.y = f.cy + f.sin * lx * f.sx + f.cos * ly * f.sy;
  return out;
}

function updateSkeleton() {
  const c = creature,
    f = sk.xf;
  const breath = Math.sin(time * 1.5) * 0.012 * (1 + presenceE * 0.4);
  const scale = CREATURE_SCALE * (1 + breath + c.expand.pos);
  const bend = clamp(c.bend.pos, 0, 0.7);
  f.sx = scale * (1 + bend * 0.22);
  f.sy = scale * (1 - bend * 0.38);
  f.cx = c.x.pos;
  f.cy = BASE_Y + c.y.pos - c.hop.pos + bend * 60;
  f.cos = Math.cos(c.tilt.pos);
  f.sin = Math.sin(c.tilt.pos);
  sk.scale = scale;

  // head: pivots at the neck. When the visitor bends, the head drops forward.
  toWorld(DESIGN.neck.x, DESIGN.neck.y, sk.neck);
  const hr = c.headRot.pos,
    hc = Math.cos(hr),
    hs = Math.sin(hr);
  const hx = (DESIGN.headOffset.x + c.headX.pos * 0.5) * scale,
    hy = (DESIGN.headOffset.y + bend * 42 - bloomE * 10) * scale;
  sk.head.x = sk.neck.x + hc * hx - hs * hy;
  sk.head.y = sk.neck.y + hs * hx + hc * hy;
  sk.head.rot = hr;
  toWorld(0, -62, sk.chest);
  toWorld(0, 96, sk.pelvis);

  for (const side of [-1, 1]) {
    const k = side < 0 ? 'L' : 'R',
      arm = c['arm' + k],
      leg = c['leg' + k],
      A = sk.arms[k],
      Lg = sk.legs[k];
    toWorld(DESIGN.shoulder[k].x, DESIGN.shoulder[k].y, A.shoulder);
    A.elbow.x = A.shoulder.x + Math.sin(arm.up.pos) * DESIGN.arm.up * scale;
    A.elbow.y = A.shoulder.y + Math.cos(arm.up.pos) * DESIGN.arm.up * scale;
    A.wrist.x = A.elbow.x + Math.sin(arm.low.pos) * DESIGN.arm.low * scale;
    A.wrist.y = A.elbow.y + Math.cos(arm.low.pos) * DESIGN.arm.low * scale;
    const ha = arm.low.pos + c['flop' + k].pos;
    sk.handTip[k].x = A.wrist.x + Math.sin(ha) * DESIGN.arm.hand * scale;
    sk.handTip[k].y = A.wrist.y + Math.cos(ha) * DESIGN.arm.hand * scale;
    toWorld(DESIGN.hip[k].x, DESIGN.hip[k].y, Lg.hip);
    Lg.knee.x = Lg.hip.x + Math.sin(leg.up.pos) * DESIGN.leg.up * scale;
    Lg.knee.y = Lg.hip.y + Math.cos(leg.up.pos) * DESIGN.leg.up * scale;
    Lg.ankle.x = Lg.knee.x + Math.sin(leg.low.pos) * DESIGN.leg.low * scale;
    Lg.ankle.y = Lg.knee.y + Math.cos(leg.low.pos) * DESIGN.leg.low * scale;
    sk.feet[k].x = Lg.ankle.x + side * 22 * scale;
    sk.feet[k].y = Lg.ankle.y + 28 * scale;
  }
}

// ----------------------------------------------------------------
// 11a. THE NETWORK: nodes, connections and how they are generated (built once, then re-used every frame)
//
//   MAIN NODES    the 20 joints of the rig (head, neck, shoulders, elbows, wrists, hands, chest, pelvis, hips, knees, ankles, feet)
//   SECONDARY     small nodes generated around the body: rings around the head, a lattice in the torso,
//                 ladders of nodes along arms and legs, and small fans at hands and feet.
//                 EVERY node belongs to the head, torso, an arm or a leg: nothing is generated outside the body.
//                 Each one is "attached" to a bone or body part, so it follows the main nodes.
//   CONNECTIONS   tier 0 = primary structure (bright), tier 1 = secondary web (thin, from nearest neighbours,
//                 so triangles appear), tier 2 = a few long, very subtle background threads.
// ----------------------------------------------------------------
const NB = PALETTE.length * 3 * 3; // connection batches: colour x tier x brightness level
const web = {
  ready: false,
  warm: false,
  nodes: [],
  edges: [],
  tier1Start: 0,
  tier2Start: 0,
  dist: [],
  drawOrder: [],
  eCx: null,
  eCy: null,
  eBucket: null,
  eOrder: null,
  counts: new Int32Array(NB),
  starts: new Int32Array(NB),
  fill: new Int32Array(NB),
  pulses: Array.from({ length: 16 }, () => ({
    alive: false,
    src: 0,
    t0: 0,
    amp: 1,
    speed: 600,
    width: 80,
    maxR: 900,
    r: 0,
  })),
  pulseIdx: 0,
  jx: new Float32Array(20),
  jy: new Float32Array(20),
  js: new Float32Array(20),
  cool: new Float32Array(20),
  idlePulseT: 1,
  flash: 0,
  bright: 1,
  restScale: 1,
};

// node types: where the node is attached
const N_JOINT = 0,
  N_BONE = 1,
  N_TORSO = 2,
  N_HEAD = 3;
function makeNode(o) {
  const a = wr(0, Math.PI * 2);
  const primary = o.type === N_JOINT;
  const k = primary ? 300 : wr(70, 150);
  return {
    type: o.type,
    zone: o.zone,
    ramp: o.ramp,
    cidx: colorIndex(o.zone, o.ramp),
    a: o.a ?? 0,
    b: o.b ?? 0,
    t: o.t ?? 0,
    off: o.off ?? 0,
    jit: o.jit ?? 0,
    lx: o.lx ?? 0,
    ly: o.ly ?? 0,
    primary,
    size: o.size ?? 2,
    bright: !!o.bright,
    role: o.role ?? 0, // role: 1 = eye, 2 = core
    x: 0,
    y: 0,
    vx: 0,
    vy: 0,
    hx: 0,
    hy: 0,
    e: 0, // e = energy of waves passing through
    dirx: Math.cos(a),
    diry: Math.sin(a),
    f1: wr(0.3, 0.9),
    f2: wr(0.3, 0.9),
    ph1: wr(0, 6.28),
    ph2: wr(0, 6.28),
    amp: primary ? 0.12 : wr(0.5, 1.4),
    k,
    d: 2 * Math.sqrt(k) * wr(0.4, 0.6),
  };
}

// Where should this node be right now? (the "home" it springs toward) Nothing is allocated.
const HT = { x: 0, y: 0 };
function computeHome(n, scale) {
  switch (n.type) {
    case N_JOINT: {
      const j = JOINTS[n.a];
      n.hx = j.x;
      n.hy = j.y;
      break;
    }
    case N_BONE: {
      // a point on a bone (between two main nodes), pushed sideways + along the bone
      const A = JOINTS[n.a],
        B = JOINTS[n.b],
        dx = B.x - A.x,
        dy = B.y - A.y,
        len = Math.hypot(dx, dy) || 1,
        tx = dx / len,
        ty = dy / len;
      n.hx = A.x + dx * n.t - ty * n.off * scale + tx * n.jit * scale;
      n.hy = A.y + dy * n.t + tx * n.off * scale + ty * n.jit * scale;
      break;
    }
    case N_TORSO:
      toWorld(n.lx, n.ly, HT);
      n.hx = HT.x;
      n.hy = HT.y;
      break;
    case N_HEAD: {
      const c = Math.cos(sk.head.rot),
        s = Math.sin(sk.head.rot);
      n.hx = sk.head.x + (c * n.lx - s * n.ly) * scale;
      n.hy = sk.head.y + (s * n.lx + c * n.ly) * scale;
      break;
    }
  }
}

// An irregular triangle mesh through a set of points (Bowyer-Watson Delaunay). Used for the head, so its inside is a
// rich web of irregular triangles instead of a hollow outline. Returns pairs of point indices.
function delaunayEdges(pts) {
  const n = pts.length,
    B = 1e4;
  const P = pts
    .map((q) => ({ x: q[0], y: q[1] }))
    .concat([
      { x: -B, y: -B },
      { x: B, y: -B },
      { x: 0, y: B },
    ]);
  let tris = [[n, n + 1, n + 2]];
  const inCircle = (t, p) => {
    const a = P[t[0]],
      b = P[t[1]],
      c = P[t[2]];
    const ax = a.x - p.x,
      ay = a.y - p.y,
      bx = b.x - p.x,
      by = b.y - p.y,
      cx = c.x - p.x,
      cy = c.y - p.y;
    const det =
      (ax * ax + ay * ay) * (bx * cy - cx * by) -
      (bx * bx + by * by) * (ax * cy - cx * ay) +
      (cx * cx + cy * cy) * (ax * by - bx * ay);
    const orient = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    return orient > 0 ? det > 0 : det < 0;
  };
  for (let i = 0; i < n; i++) {
    const bad = tris.filter((t) => inCircle(t, P[i]));
    const count = new Map();
    for (const t of bad)
      for (const [u, v] of [
        [t[0], t[1]],
        [t[1], t[2]],
        [t[2], t[0]],
      ]) {
        const k = u < v ? u * 8192 + v : v * 8192 + u;
        count.set(k, (count.get(k) || 0) + 1);
      }
    tris = tris.filter((t) => !bad.includes(t));
    for (const [k, c] of count)
      if (c === 1) tris.push([Math.floor(k / 8192), k % 8192, i]);
  }
  const out = new Set();
  for (const t of tris)
    for (const [u, v] of [
      [t[0], t[1]],
      [t[1], t[2]],
      [t[2], t[0]],
    ])
      if (u < n && v < n) out.add(u < v ? u * 8192 + v : v * 8192 + u);
  return [...out].map((k) => [Math.floor(k / 8192), k % 8192]);
}

function buildWeb() {
  updateSkeleton(); // the skeleton in its idle pose
  const nodes = web.nodes,
    edges = web.edges;
  const add = (o) => {
    nodes.push(makeNode(o));
    return nodes.length - 1;
  };

  // ---- MAIN NODES (index = joint id) ----
  const PRIM = [
    [ZONE.HEAD, 0.5, 5],
    [ZONE.UPPER, 0.1, 4],
    [ZONE.UPPER, 0.4, 5],
    [ZONE.UPPER, 0.4, 5],
    [ZONE.ARM, 0.35, 5],
    [ZONE.ARM, 0.35, 5],
    [ZONE.ARM, 0.7, 5.5],
    [ZONE.ARM, 0.7, 5.5],
    [ZONE.ARM, 1, 4.5],
    [ZONE.ARM, 1, 4.5],
    [ZONE.UPPER, 0.7, 5],
    [ZONE.LOWER, 0.1, 5],
    [ZONE.LOWER, 0.25, 4.5],
    [ZONE.LOWER, 0.25, 4.5],
    [ZONE.LOWER, 0.55, 5],
    [ZONE.LOWER, 0.55, 5],
    [ZONE.LOWER, 0.85, 5],
    [ZONE.LOWER, 0.85, 5],
    [ZONE.LOWER, 1, 4.5],
    [ZONE.LOWER, 1, 4.5],
  ];
  PRIM.forEach(([zone, ramp, size], id) =>
    add({ type: N_JOINT, zone, ramp, size, a: id })
  );

  // ---- HEAD: built exactly like the torso: a dense mesh of nodes through the WHOLE head ----
  //   1. a rounded silhouette ring   2. nodes spread evenly through the inside (no two closer than ~19 units)
  //   3. two tiny bright "eyes" (the centre of the head is the main head node, which acts as the bright core)
  // Later, the nodes are connected with an irregular triangle mesh (Delaunay) + extra cross-connections.
  const HEAD_RX = 98,
    HEAD_RY = 90;
  const headPts = []; // local positions (design units) of every head node
  const addHeadNode = (lx, ly, o) => {
    headPts.push([lx, ly]);
    return add(
      Object.assign(
        {
          type: N_HEAD,
          zone: ZONE.HEAD,
          ramp: clamp((ly + 100) / 200, 0, 1),
          lx,
          ly,
          size: wr(1.5, 2.6),
        },
        o || {}
      )
    );
  };
  headPts.push([0, 0]); // the main head node (joint 0) sits here
  const eyeL = addHeadNode(-27, -8, {
    size: 2.8,
    bright: true,
    role: 1,
    ramp: 0.9,
  });
  const eyeR = addHeadNode(27, -8, {
    size: 2.8,
    bright: true,
    role: 1,
    ramp: 0.9,
  });
  const RING = 22;
  for (let i = 0; i < RING; i++) {
    // rounded, slightly irregular silhouette
    const ang = (i / RING) * Math.PI * 2 + wr(-0.07, 0.07),
      rr = 1 + wr(-0.05, 0.05);
    addHeadNode(Math.cos(ang) * HEAD_RX * rr, Math.sin(ang) * HEAD_RY * rr);
  }
  let inner = 0;
  for (let tries = 0; tries < 2500 && inner < 40; tries++) {
    // interior: random points, kept only if not too close to another
    const ang = wr(0, Math.PI * 2),
      r = Math.sqrt(wrng()) * 0.9;
    const lx = Math.cos(ang) * HEAD_RX * r,
      ly = Math.sin(ang) * HEAD_RY * r;
    if (headPts.every(([qx, qy]) => Math.hypot(qx - lx, qy - ly) >= 19)) {
      addHeadNode(lx, ly);
      inner++;
    }
  }
  const headNodeIds = []; // (filled after the main nodes exist: indices >= 20)
  for (let i = 20; i < nodes.length; i++)
    if (nodes[i].type === N_HEAD) headNodeIds.push(i);
  nodes[0].role = 2;
  nodes[0].bright = true; // the main head node is the core of the head

  // ---- TORSO: an irregular lattice of nodes + a few nodes sticking out of the silhouette ----
  for (const y0 of [-132, -90, -48, -6, 36, 78, 120]) {
    const y = y0 + wr(-6, 6),
      hw = torsoHalfWidth(y) * 0.9,
      cols = Math.max(3, Math.round((hw * 1.6) / 34) + 1);
    const zone = y < 30 ? ZONE.UPPER : ZONE.LOWER,
      ramp =
        y < 30
          ? clamp((y + 135) / 165, 0, 1)
          : clamp(((y - 30) / 160) * 0.45, 0, 1);
    for (let c = 0; c < cols; c++)
      add({
        type: N_TORSO,
        zone,
        ramp,
        lx: lerp(-hw * 0.78, hw * 0.78, c / (cols - 1)) + wr(-8, 8),
        ly: y + wr(-8, 8),
        size: wr(1.5, 2.6),
      });
    for (const side of [-1, 1])
      add({
        type: N_TORSO,
        zone,
        ramp,
        lx: side * (hw + wr(4, 22)),
        ly: y + wr(-10, 10),
        size: wr(1.4, 2.2),
      });
  }

  // ---- LIMBS: ladders of nodes along each bone, and little fans at hands and feet ----
  const bone = (a, b, zone, r0, r1, rad, count) => {
    for (let i = 0; i < count; i++) {
      const t = lerp(0.12, 0.88, i / (count - 1)),
        ramp = lerp(r0, r1, t);
      for (const side of [-1, 1])
        add({
          type: N_BONE,
          zone,
          ramp,
          a,
          b,
          t,
          off: side * rad * wr(0.8, 1.5),
          jit: wr(-8, 8),
          size: wr(1.4, 2.4),
        });
      if (wrng() < 0.5)
        add({
          type: N_BONE,
          zone,
          ramp,
          a,
          b,
          t: clamp(t + wr(-0.05, 0.05), 0.05, 0.95),
          off: wr(-6, 6),
          size: wr(1.2, 2),
        });
    }
  };
  const fan = (a, b, zone, ramp, count, rad) => {
    for (let i = 0; i < count; i++)
      add({
        type: N_BONE,
        zone,
        ramp,
        a,
        b,
        t: wr(0.5, 1.02),
        off: (i / (count - 1) - 0.5) * 2 * rad * wr(0.7, 1.1),
        size: wr(1.3, 2.1),
      });
  };
  for (const [sh, el, wr_, hd, hip, kn, an, ft] of [
    [2, 4, 6, 8, 12, 14, 16, 18],
    [3, 5, 7, 9, 13, 15, 17, 19],
  ]) {
    bone(sh, el, ZONE.ARM, 0.0, 0.45, 26, 5);
    bone(el, wr_, ZONE.ARM, 0.45, 0.8, 22, 5);
    fan(wr_, hd, ZONE.ARM, 0.95, 6, 20);
    bone(hip, kn, ZONE.LOWER, 0.15, 0.5, 30, 5);
    bone(kn, an, ZONE.LOWER, 0.5, 0.8, 26, 5);
    fan(an, ft, ZONE.LOWER, 0.95, 5, 24);
  }

  // a few secondary nodes shine more than the others
  for (let i = 20; i < nodes.length; i++)
    if (!nodes[i].role && wrng() < 0.12) {
      nodes[i].bright = true;
      nodes[i].size *= 1.3;
    }
  const N = nodes.length;

  // start every node at its home
  for (const n of nodes) {
    computeHome(n, sk.scale);
    n.x = n.hx;
    n.y = n.hy;
  }

  // ---- CONNECTIONS ----
  const seen = new Set();
  const addEdge = (a, b, tier) => {
    if (a === b) return false;
    const key = a < b ? a * 4096 + b : b * 4096 + a;
    if (seen.has(key)) return false;
    seen.add(key);
    const na = nodes[a],
      nb = nodes[b];
    const cidx =
      na.zone === nb.zone
        ? colorIndex(na.zone, (na.ramp + nb.ramp) / 2)
        : wrng() < 0.5
        ? na.cidx
        : nb.cidx;
    edges.push({
      a,
      b,
      tier,
      cidx,
      base: [0.9, 0.5, 0.2][tier] * wr(0.75, 1.2),
      f: wr(0.5, 1.6),
      ph: wr(0, 6.28), // brightness pulse
      vf: wr(0.6, 1.8),
      vph: wr(0, 6.28),
      vamp: [1.2, 2.4, 4][tier] * wr(0.6, 1.2), // vibration of the thread
      sag: tier === 2 ? wr(0.04, 0.1) : 0,
      es: [0, 14, 3][tier],
      rest: 0,
    });
    return true;
  };
  // tier 0: the primary structure (bright)
  const PRIMARY_EDGES = [
    [0, 1],
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [4, 6],
    [6, 8],
    [3, 5],
    [5, 7],
    [7, 9],
    [1, 10],
    [10, 11],
    [2, 10],
    [3, 10],
    [11, 12],
    [11, 13],
    [12, 13],
    [12, 14],
    [14, 16],
    [16, 18],
    [13, 15],
    [15, 17],
    [17, 19],
    [2, 12],
    [3, 13],
  ];
  for (const [a, b] of PRIMARY_EDGES) addEdge(a, b, 0);
  addEdge(2, 13, 1);
  addEdge(3, 12, 1);
  addEdge(10, 12, 1);
  addEdge(10, 13, 1);

  // the head mesh: irregular triangles through all head nodes (long skinny ones are dropped) ...
  {
    const ids = [0].concat(headNodeIds); // the main head node + every head node (same order as headPts)
    for (const [u, v] of delaunayEdges(headPts)) {
      const [x1, y1] = headPts[u],
        [x2, y2] = headPts[v];
      if (Math.hypot(x1 - x2, y1 - y2) < 54) addEdge(ids[u], ids[v], 1);
    }
    // ... and the head is joined to the neck and shoulders: the lowest nodes of the head reach the neck and the top of the shoulders
    for (const id of headNodeIds) {
      const n = nodes[id];
      if (n.ly > 55 && Math.abs(n.lx) < 60) addEdge(id, 1, 1); // neck
      else if (n.ly > 35 && n.lx < -45) addEdge(id, 2, 1); // left shoulder
      else if (n.ly > 35 && n.lx > 45) addEdge(id, 3, 1); // right shoulder
    }
  }

  // tier 1 + 2: each node connects to its nearest neighbours (this naturally makes small triangles / polygons)
  const px = nodes.map((n) => n.hx),
    py = nodes.map((n) => n.hy);
  const maxD = 95 * sk.scale;
  for (let i = 0; i < N; i++) {
    const cand = [];
    for (let j = 0; j < N; j++) {
      if (j === i || (nodes[i].primary && nodes[j].primary)) continue;
      const d = Math.hypot(px[i] - px[j], py[i] - py[j]);
      if (d < maxD) cand.push([d, j]);
    }
    cand.sort((p, q) => p[0] - q[0]);
    cand.forEach(([, j], r) => {
      if (r < 2) addEdge(i, j, 1);
      else if (r === 2 && wrng() < 0.65) addEdge(i, j, 1);
      else if (r === 3 && wrng() < 0.3) addEdge(i, j, 2);
    });
    // tie bone nodes to the main nodes they hang on
    if (nodes[i].type === N_BONE && wrng() < 0.6)
      addEdge(i, wrng() < nodes[i].t ? nodes[i].b : nodes[i].a, 1);
  }
  // tier 2: a few long, very subtle threads across the body (they glow more when the arms go up)
  let longCount = 0;
  for (let tries = 0; tries < 900 && longCount < 130; tries++) {
    const i = Math.floor(wrng() * N),
      j = Math.floor(wrng() * N);
    const d = Math.hypot(px[i] - px[j], py[i] - py[j]);
    if (
      d > 100 * sk.scale &&
      d < 260 * sk.scale &&
      !(nodes[i].primary && nodes[j].primary) &&
      addEdge(i, j, 2)
    )
      longCount++;
  }
  edges.sort((p, q) => p.tier - q.tier); // tier 0 first, tier 2 last (so detail can be cut when the PC is slow)
  web.tier1Start = edges.findIndex((e) => e.tier >= 1);
  web.tier2Start = edges.findIndex((e) => e.tier === 2);
  for (const e of edges)
    e.rest =
      Math.hypot(nodes[e.a].hx - nodes[e.b].hx, nodes[e.a].hy - nodes[e.b].hy) /
      sk.scale; // rest length (design units)

  // ---- distances along the web (used by the waves). Only tier 0/1 count, so waves travel along the structure. ----
  const adj = nodes.map(() => []);
  for (const e of edges)
    if (e.tier < 2) {
      const L = e.rest * sk.scale;
      adj[e.a].push([e.b, L]);
      adj[e.b].push([e.a, L]);
    }
  for (let src = 0; src < 20; src++) {
    const dist = new Float32Array(N).fill(1e9),
      done = new Uint8Array(N);
    dist[src] = 0;
    for (let it = 0; it < N; it++) {
      let u = -1,
        best = 1e9;
      for (let i = 0; i < N; i++)
        if (!done[i] && dist[i] < best) {
          best = dist[i];
          u = i;
        }
      if (u < 0) break;
      done[u] = 1;
      for (const [v, w] of adj[u])
        if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
    }
    web.dist.push(dist);
  }
  // drawing order: nodes grouped by colour, so each colour is one batch
  web.drawOrder = nodes
    .map((_, i) => i)
    .sort((p, q) => nodes[p].cidx - nodes[q].cidx);
  const E = edges.length;
  web.eCx = new Float32Array(E);
  web.eCy = new Float32Array(E);
  web.eBucket = new Int16Array(E);
  web.eOrder = new Int32Array(E);
  for (let j = 0; j < 20; j++) {
    web.jx[j] = JOINTS[j].x;
    web.jy[j] = JOINTS[j].y;
  }
  web.ready = true;
  web.warm = true;
}

// ----------------------------------------------------------------
// 11. BODY MAPPING + GESTURE DETECTION  (the invisible control skeleton)
//
//   visitor's left shoulder/elbow/wrist  -> creature's left (screen-left) arm
//   visitor's right shoulder/elbow/wrist -> creature's right (screen-right) arm
//   visitor's hips                        -> creature body position + height
//   visitor's shoulders vs hips           -> creature lean + bend
//   visitor's nose                        -> creature head
//   visitor's knees + ankles              -> creature knees + feet
//
//   Only the ANGLE of each bone is copied, so the creature keeps its own short, chubby proportions.
//   Landmarks are mirrored (1 - x): the creature acts like a mirror image of the visitor.
// ----------------------------------------------------------------
const control = {
  needsBaseline: true,
  hipBaseline: 0.6, // slowly-adapting "standing height" of the hips (camera coordinates)
  torsoRef: 0.25, // slowly-adapting "upright torso length"
  hopRequest: false, // set when a jump is detected
  lastHopMs: 0,
  target: {
    x: W / 2,
    y: 0,
    tilt: 0,
    bend: 0,
    headX: 0,
    crouch: 0,
    motion: 0,
    raisedL: false,
    raisedR: false,
    bothArmsUp: false,
    armL: { up: 0, low: 0, weight: 0 },
    armR: { up: 0, low: 0, weight: 0 },
    legL: { up: 0, low: 0, weight: 0 },
    legR: { up: 0, low: 0, weight: 0 },
  },
};

function computeControlTargets(dt, nowMs) {
  const T = control.target;
  const aspect =
    video.videoWidth && video.videoHeight
      ? video.videoWidth / video.videoHeight
      : 4 / 3;
  const mx = (i) => 1 - smoothed[i].x; // mirrored x
  const my = (i) => smoothed[i].y;
  const vis = (i) => smoothed[i].visibility;

  const shoulderX = (mx(LM.L_SHOULDER) + mx(LM.R_SHOULDER)) / 2;
  const shoulderY = (my(LM.L_SHOULDER) + my(LM.R_SHOULDER)) / 2;
  const hipsVisible = (vis(LM.L_HIP) + vis(LM.R_HIP)) / 2 > 0.45;
  // If the hips are out of the camera picture, guess them just below the shoulders.
  const hipX = hipsVisible ? (mx(LM.L_HIP) + mx(LM.R_HIP)) / 2 : shoulderX;
  const hipY = hipsVisible
    ? (my(LM.L_HIP) + my(LM.R_HIP)) / 2
    : shoulderY + 0.22;

  // --- sideways position: visitor moves left -> creature moves left ---
  // (the travel range follows the visible screen width, so it also works on a narrow portrait screen)
  const visW = view.worldRight - view.worldLeft;
  T.x = clamp(
    W / 2 + (hipX - 0.5) * visW * CONFIG.moveGain,
    view.minX,
    view.maxX
  );

  // --- height: up = creature rises, crouch = creature goes down ---
  if (control.needsBaseline) control.hipBaseline = hipY;
  control.hipBaseline += (hipY - control.hipBaseline) * Math.min(1, dt * 0.35);
  const rawHeight = hipY - control.hipBaseline; // negative = visitor went UP
  T.crouch = clamp(rawHeight / 0.12, 0, 1);
  T.y =
    rawHeight < 0
      ? clamp(rawHeight * H * CONFIG.verticalGain, -view.maxRise, 0) // rises at most as far as the free room above the head
      : T.crouch * 50;

  // --- body lean + bend (bend = the torso looks shorter when the visitor bends forward) ---
  const torsoDx = (shoulderX - hipX) * aspect,
    torsoDy = hipY - shoulderY;
  const torsoLen = Math.hypot(torsoDx, torsoDy);
  if (hipsVisible) {
    T.tilt = clamp(Math.atan2(torsoDx, torsoDy), -0.6, 0.6);
    if (control.needsBaseline) control.torsoRef = torsoLen;
    control.torsoRef = Math.max(
      torsoLen,
      control.torsoRef * (1 - dt * 0.03),
      0.12
    );
    T.bend = clamp(1 - torsoLen / control.torsoRef, 0, 0.7);
  } else {
    T.tilt = 0;
    T.bend = 0;
  }
  control.needsBaseline = false;

  // --- head: nose position relative to the shoulders ---
  const shoulderWidth = Math.max(
    Math.abs(mx(LM.L_SHOULDER) - mx(LM.R_SHOULDER)) * aspect,
    0.05
  );
  T.headX =
    clamp(((mx(LM.NOSE) - shoulderX) * aspect) / shoulderWidth, -0.8, 0.8) * 50;

  // --- limbs. Angle 0 = pointing down, +90deg = pointing to the right of the screen, 180deg = up ---
  const boneAngle = (i, j) =>
    Math.atan2((mx(j) - mx(i)) * aspect, my(j) - my(i));
  const confidence = (i, j, k) =>
    clamp((Math.min(vis(i), vis(j), vis(k)) - 0.35) / 0.3, 0, 1);
  const setLimb = (name, s, e, w, side, splay) => {
    const up = boneAngle(s, e),
      low = boneAngle(e, w);
    const hanging = (ang) => 0.5 + 0.5 * Math.cos(ang); // 1 when hanging down, 0 when pointing up
    // "splay" holds hanging limbs slightly away from the body so they look chubby and cute
    T[name].up = up + side * splay * hanging(up);
    T[name].low = low + side * splay * hanging(low);
    T[name].weight = confidence(s, e, w); // limbs the camera can't see fall back to idle
  };
  setLimb('armL', LM.L_SHOULDER, LM.L_ELBOW, LM.L_WRIST, -1, 0.22);
  setLimb('armR', LM.R_SHOULDER, LM.R_ELBOW, LM.R_WRIST, +1, 0.22);
  setLimb('legL', LM.L_HIP, LM.L_KNEE, LM.L_ANKLE, -1, 0.1);
  setLimb('legR', LM.R_HIP, LM.R_KNEE, LM.R_ANKLE, +1, 0.1);

  // ---------------- GESTURE DETECTION ----------------
  // Arms raised: a wrist clearly above its own shoulder.
  T.raisedL =
    vis(LM.L_WRIST) > 0.4 &&
    smoothed[LM.L_WRIST].y < smoothed[LM.L_SHOULDER].y - 0.03;
  T.raisedR =
    vis(LM.R_WRIST) > 0.4 &&
    smoothed[LM.R_WRIST].y < smoothed[LM.R_SHOULDER].y - 0.03;
  T.bothArmsUp = T.raisedL && T.raisedR;

  // Motion amount: the biggest of hands, torso and feet speed (visible landmarks only).
  const seen = (i) => (vis(i) > 0.4 ? smoothed[i].speed : 0);
  const hands = (seen(LM.L_WRIST) + seen(LM.R_WRIST)) / 2;
  const core =
    (seen(LM.L_SHOULDER) +
      seen(LM.R_SHOULDER) +
      seen(LM.L_HIP) +
      seen(LM.R_HIP)) /
    4;
  const feet = (seen(LM.L_ANKLE) + seen(LM.R_ANKLE)) / 2;
  T.motion = Math.max(hands * 0.8, core * 1.3, feet * 0.8);

  // Jump: hips AND ankles move up quickly (standing up from a crouch moves only the hips, so it does not count).
  const hipVy = hipsVisible
    ? (smoothed[LM.L_HIP].vy + smoothed[LM.R_HIP].vy) / 2
    : (smoothed[LM.L_SHOULDER].vy + smoothed[LM.R_SHOULDER].vy) / 2;
  const anklesVisible = (vis(LM.L_ANKLE) + vis(LM.R_ANKLE)) / 2 > 0.4;
  const ankleVy = anklesVisible
    ? (smoothed[LM.L_ANKLE].vy + smoothed[LM.R_ANKLE].vy) / 2
    : hipVy;
  if (
    hipVy < -CONFIG.jumpSpeed &&
    ankleVy < -CONFIG.jumpSpeed * 0.5 &&
    nowMs - control.lastHopMs > 700
  ) {
    control.hopRequest = true;
    control.lastHopMs = nowMs;
  }
}

// ----------------------------------------------------------------
// 12. INTERACTION STATES + EFFECTS (waves, particles)
//
//   1 NO PERSON  -> calm idle          2 PERSON ENTERED -> notices, head turns, small hop
//   3 STANDING / MOVING -> follows     4 FAST MOVEMENT -> stronger waves, brighter threads
//   5 BOTH ARMS UP -> NETWORK EXPANDS  6 JUMP -> stretch + rebound   7 BEND -> compresses
//   8 PERSON LEFT -> holds the last pose briefly, then slowly relaxes to idle
//   Every reaction is driven by values that rise and fall smoothly, so states blend instead of switching.
// ----------------------------------------------------------------
let time = 0;
let present = false,
  wasPresent = false;
let presence = 0,
  presenceE = 0; // 0 = nobody (idle), 1 = visitor here (tracking)
let motion = 0,
  motionNorm = 0; // how much the visitor moves
let scatter = 0; // 0..1 strength of the fast-movement reaction
let lift = 0; // 0..1 how many arms are raised
let bloom = 0,
  bloomE = 0,
  bothUpTimer = 0,
  wasBloomActive = false;
let notice = 0,
  noticeSide = 1; // 1 -> 0 over ~1.8 s after a visitor enters
let stateName = '1 NO PERSON (idle)';
const quality = { fraction: 1 }; // 1 = everything, lower = fewer faint threads / particles (auto-adjusted)

function onVisitorEnter() {
  // The network wakes up: it scatters a little, a wave runs through it, then it forms around the visitor.
  noticeSide =
    Math.sign(control.target.x - W / 2) || (Math.random() < 0.5 ? -1 : 1);
  notice = 1;
  creature.hop.vel += 300;
  creature.headRot.vel += noticeSide * 2.0;
  webWake();
}
function onBloomStart() {
  creature.expand.vel += 1.0; // quick swell, then the spring settles
  creature.hop.vel += 300;
  webBloomStart(); // the network expands outward, a wave + a ring of light
}

function updateInteraction(dt, nowMs) {
  const T = control.target;
  present = nowMs - lastSeenMs < CONFIG.lostAfterMs;
  state.poseDetected = present;

  // Quick to notice a visitor (~0.5 s), slow to let go after they leave (~2 s back to idle).
  // When the visitor leaves, the creature HOLDS their last pose for a moment, then slowly relaxes into its idle form.
  const held = nowMs - lastSeenMs < CONFIG.holdAfterLeaveMs;
  presence = approach(presence, held ? 1 : 0, 1.8, 0.35, dt);
  presenceE = smoothstep(presence);

  if (present) {
    computeControlTargets(dt, nowMs);
    let count = 0;
    for (const p of smoothed)
      if (p.ready && p.visibility >= CONFIG.minVisibility) count++;
    state.visibleCount = count;
  } else {
    state.visibleCount = 0;
  }
  if (present && !wasPresent) onVisitorEnter();
  wasPresent = present;

  // movement amount (rises fast, falls slowly) + the fast-movement reaction strength
  const motionTarget = present ? T.motion : 0;
  motion +=
    (motionTarget - motion) *
    (1 - Math.exp(-dt * (motionTarget > motion ? 10 : 2.5)));
  motionNorm = clamp(motion / 0.9, 0, 1);
  scatter = clamp((motion - CONFIG.fastThreshold * 0.55) / 0.8, 0, 1);

  const raised = present ? (T.raisedL ? 1 : 0) + (T.raisedR ? 1 : 0) : 0;
  lift += (raised / 2 - lift) * (1 - Math.exp(-dt * 3));

  // BLOOM: both arms must stay up for a moment (avoids accidental triggers). It fades out slowly.
  bothUpTimer = present && T.bothArmsUp ? bothUpTimer + dt : 0;
  const bloomActive = bothUpTimer > CONFIG.bloomHoldSeconds;
  if (bloomActive && !wasBloomActive) onBloomStart();
  wasBloomActive = bloomActive;
  bloom = approach(bloom, bloomActive ? 1 : 0, 2.2, 0.7, dt);
  bloomE = smoothstep(bloom);

  notice = Math.max(0, notice - dt / 1.8);

  // human-readable state, shown in development mode
  if (!present && !held && presence < 0.02) stateName = '1 NO PERSON (idle)';
  else if (!present)
    stateName = held
      ? '8 PERSON LEFT (holding last pose)'
      : '8 PERSON LEFT (relaxing to idle)';
  else if (bloomActive) stateName = '5 BOTH ARMS UP (bloom)';
  else if (motion > CONFIG.fastThreshold) stateName = '4 FAST MOVEMENT';
  else if (notice > 0.05) stateName = '2 PERSON ENTERED (noticing)';
  else if (motion > CONFIG.movingThreshold) stateName = '3 PERSON MOVING';
  else stateName = '3 STANDING STILL';
}

// ----------------------------------------------------------------
// 11b. LIVING NETWORK: waves (pulses), node springs, thread tension
// ----------------------------------------------------------------
// A "pulse" is a wave that starts at one main node and travels along the web. Every node knows its distance
// (along the web) from every main node, so a pulse is just "light up the nodes at distance ~ speed * age".
// So a move of the right hand travels through the arm to the shoulder, the torso and the other side of the body.
function emitPulse(src, amp, speed, width) {
  if (!web.ready) return;
  const p = web.pulses[web.pulseIdx];
  web.pulseIdx = (web.pulseIdx + 1) % web.pulses.length; // fixed pool: the oldest is recycled
  p.alive = true;
  p.src = src;
  p.t0 = time;
  p.amp = amp;
  p.speed = speed;
  p.width = width;
  p.maxR = 1100;
  p.r = 0;
}
const PULSE_JOINTS = new Set([
  0, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19,
]);

function webWake() {
  // a visitor arrives: the network "wakes up", scatters a little and then forms around them
  if (!web.ready) return;
  for (const n of web.nodes)
    if (!n.primary) {
      n.vx += rand(-1, 1) * 260;
      n.vy += rand(-1, 1) * 260;
    }
  emitPulse(0, 1.0, 900, 110);
  emitPulse(10, 0.8, 700, 100);
  web.flash = 1;
}
function webBloomStart() {
  // both arms up: the network expands outward, a big wave from the chest, a ring of light
  emitPulse(10, 1.1, 950, 130);
  emitPulse(0, 0.8, 800, 100);
  emitRing(sk.xf.cx, sk.xf.cy - 60, 520);
  burstMotes(34);
  web.flash = Math.max(web.flash, 0.7);
}
function webJump() {
  // a jump: waves rush up from the feet; the stretch itself comes from the rig's bounce spring
  emitPulse(16, 1.0, 1100, 100);
  emitPulse(17, 1.0, 1100, 100);
  for (const k of ['L', 'R'])
    for (let i = 0; i < 4; i++)
      emitMote(
        sk.feet[k].x + rand(-30, 30),
        sk.feet[k].y,
        rand(-60, 60),
        rand(-220, -80),
        rand(6, 12),
        rand(0.8, 1.4),
        0,
        colorIndex(ZONE.LOWER, rand(0.4, 1))
      );
}

function updateWeb(dt) {
  if (!web.ready) buildWeb();
  const nodes = web.nodes,
    edges = web.edges,
    N = nodes.length,
    E = edges.length;
  const sc = sk.scale,
    cx = sk.xf.cx,
    cy = sk.xf.cy - 30,
    tw = time;

  // 1. how fast does each main node move? Fast movement sends a wave through the web.
  const still = !present || motion < 0.08;
  for (let j = 0; j < 20; j++) {
    const p = JOINTS[j],
      speed =
        Math.hypot(p.x - web.jx[j], p.y - web.jy[j]) / Math.max(dt, 0.001);
    web.js[j] = lerp(web.js[j], speed, 0.25);
    web.jx[j] = p.x;
    web.jy[j] = p.y;
    web.cool[j] -= dt;
    if (web.js[j] > 150 && web.cool[j] <= 0 && PULSE_JOINTS.has(j)) {
      emitPulse(
        j,
        clamp(web.js[j] / 850, 0.25, 1.1),
        600 + Math.min(web.js[j], 900) * 0.3,
        70
      );
      web.cool[j] = 0.22 - 0.1 * scatter;
    }
  }
  // 2. calm waves: small ones drift through the idle body now and then, slow ones when someone stands still
  web.idlePulseT -= dt;
  if (web.idlePulseT <= 0) {
    web.idlePulseT = present ? (still ? 2.6 : 5) : 1.8;
    if (still)
      emitPulse(
        [0, 6, 7, 10, 16, 17][Math.floor(Math.random() * 6)],
        0.4,
        present ? 260 : 340,
        present ? 110 : 90
      );
  }
  web.flash = Math.max(0, web.flash - dt * 1.1);

  // 3. every node's home: attached to its bone / body part, then expanded (arms up) and stretched (jump), plus a slow drift
  const st =
    clamp(creature.hop.vel * 0.00035, -0.12, 0.12) * view.hopScale + // (a lower jump stretches the network less)
    0.03 * bloomE;
  const sx = (1 + 0.2 * bloomE) * (1 - st * 0.35),
    sy = (1 + 0.05 * bloomE) * (1 + st); // expand sideways, stretch upward
  const fy = (sk.feet.L.y + sk.feet.R.y) * 0.5; // vertical scaling is anchored at the feet
  const drift = (3 + 6 * (1 - presenceE) + 7 * scatter + 3 * motionNorm) * sc;
  for (let i = 0; i < N; i++) {
    const n = nodes[i];
    computeHome(n, sc);
    n.hx = cx + (n.hx - cx) * sx + Math.sin(tw * n.f1 + n.ph1) * drift * n.amp;
    n.hy = fy + (n.hy - fy) * sy + Math.cos(tw * n.f2 + n.ph2) * drift * n.amp;
  }

  // 4. thread tension: connected nodes pull each other toward the thread's rest length (so neighbours react to movement)
  const restScale = sc * (sx + sy) * 0.5;
  web.restScale = restScale;
  for (let k = web.tier1Start; k < E; k++) {
    const e = edges[k],
      a = nodes[e.a],
      b = nodes[e.b];
    const dx = b.x - a.x,
      dy = b.y - a.y,
      len = Math.hypot(dx, dy) || 1;
    if (len > e.rest * restScale * 1.8) continue; // overstretched (e.g. an arm raised away from the body): no pull
    const f = clamp((len - e.rest * restScale) * e.es, -60, 60) * dt;
    const fx = (dx / len) * f,
      fy = (dy / len) * f;
    if (!a.primary) {
      a.vx += fx;
      a.vy += fy;
    }
    if (!b.primary) {
      b.vx -= fx;
      b.vy -= fy;
    }
  }

  // 5. waves: light up and push the nodes the wavefront is passing
  let alive = 0;
  for (const p of web.pulses)
    if (p.alive) {
      p.r = (tw - p.t0) * p.speed;
      if (p.r > p.maxR) p.alive = false;
      else alive++;
    }
  const decay = Math.exp(-dt * 2.0);
  for (let i = 0; i < N; i++) {
    const n = nodes[i];
    n.e *= decay;
    if (!alive) continue;
    for (const p of web.pulses) {
      if (!p.alive) continue;
      const x = (web.dist[p.src][i] - p.r) / p.width;
      if (x > -3 && x < 3) {
        const w = Math.exp(-x * x) * p.amp * (1 - p.r / p.maxR);
        if (w > n.e) n.e = w;
        if (!n.primary) {
          n.vx += n.dirx * w * 420 * dt;
          n.vy += n.diry * w * 420 * dt;
        }
      }
    }
  }

  // 6. move: main nodes sit exactly on the rig; all other nodes are springs (they lag, overshoot and settle)
  for (let i = 0; i < N; i++) {
    const n = nodes[i];
    if (n.primary) {
      n.x = n.hx;
      n.y = n.hy;
      n.vx = n.vy = 0;
      continue;
    }
    n.vx += (n.k * (n.hx - n.x) - n.d * n.vx) * dt;
    n.vy += (n.k * (n.hy - n.y) - n.d * n.vy) * dt;
    n.x += n.vx * dt;
    n.y += n.vy * dt;
  }
}

// ----------------------------------------------------------------
// 14. RENDERING THE WEB (additive light on a dark stage)
// ----------------------------------------------------------------
const TIER_W = [1.9, 1.05, 0.75]; // line widths per tier
const TIER_A = [
  [0.35, 0.65, 0.95],
  [0.2, 0.42, 0.7],
  [0.08, 0.18, 0.32],
]; // opacity per tier and brightness level

function drawWeb() {
  const g = ctx,
    nodes = web.nodes,
    edges = web.edges,
    E = edges.length,
    t = time,
    sc = sk.scale;
  // overall brightness: a slow breathing glow, brighter when excited, a flash when someone arrives
  const bright =
    0.7 +
    0.3 * presenceE +
    0.12 * Math.sin(t * (0.9 + 0.8 * motionNorm)) +
    0.35 * scatter +
    0.3 * bloomE +
    0.5 * web.flash;
  const vibe = 1 + 2 * scatter + 0.8 * motionNorm;

  // ---- connections: sorted into batches (colour x tier x brightness), so they are drawn with a few stroke() calls ----
  const counts = web.counts,
    starts = web.starts,
    fill = web.fill,
    eB = web.eBucket,
    eCx = web.eCx,
    eCy = web.eCy,
    order = web.eOrder;
  counts.fill(0);
  const limit =
    web.tier2Start +
    Math.floor((E - web.tier2Start) * lerp(0.35, 1, quality.fraction)); // fewer faint threads on a slow PC
  for (let k = 0; k < E; k++) {
    if (k >= limit) {
      eB[k] = -1;
      continue;
    }
    const e = edges[k],
      a = nodes[e.a],
      b = nodes[e.b],
      en = (a.e + b.e) * 0.5;
    const dx = b.x - a.x,
      dy = b.y - a.y,
      len = Math.hypot(dx, dy) || 1;
    let al =
      e.base * (0.55 + 0.45 * Math.sin(t * e.f + e.ph)) * bright + en * 0.9;
    if (e.tier > 0) {
      // a thread stretched beyond its rest length fades out and vanishes, so no long strands appear
      const stretch = len / (e.rest * web.restScale + 1),
        from = e.tier === 2 ? 1.25 : 1.6,
        to = e.tier === 2 ? 1.65 : 2.2;
      if (stretch > to || len > 280 * sc) {
        eB[k] = -1;
        continue;
      }
      if (stretch > from) al *= (to - stretch) / (to - from);
    }
    if (e.tier === 2) al += bloomE * 0.4 * e.base * 3; // the long threads light up when the network expands
    if (al < 0.03) {
      eB[k] = -1;
      continue;
    }
    const level = al < 0.3 ? 0 : al < 0.6 ? 1 : 2,
      bk = (e.cidx * 3 + e.tier) * 3 + level;
    eB[k] = bk;
    counts[bk]++;
    // the thread is a slightly curved line: it vibrates, ripples when a wave passes, and the long ones sag a little
    const wob = Math.sin(t * e.vf + e.vph) * e.vamp * (1 + en * 2) * vibe;
    eCx[k] = (a.x + b.x) / 2 - (dy / len) * wob;
    eCy[k] = (a.y + b.y) / 2 + (dx / len) * wob + e.sag * len;
  }
  let acc = 0;
  for (let b = 0; b < NB; b++) {
    starts[b] = acc;
    fill[b] = acc;
    acc += counts[b];
  }
  for (let k = 0; k < E; k++) if (eB[k] >= 0) order[fill[eB[k]]++] = k;
  // pass 1: the thin secondary threads (normal blending: cheap). pass 2: the primary structure + its glow (additive light).
  for (let pass = 0; pass < 2; pass++) {
    g.globalCompositeOperation = pass === 0 ? 'source-over' : 'lighter';
    for (let b = 0; b < NB; b++) {
      if (!counts[b]) continue;
      const level = b % 3,
        tier = Math.floor(b / 3) % 3,
        ci = Math.floor(b / 9);
      if ((tier === 0) !== (pass === 1)) continue;
      g.strokeStyle = PALETTE[ci];
      g.lineWidth = (TIER_W[tier] * sc) / CREATURE_SCALE;
      g.globalAlpha = TIER_A[tier][level] * (pass === 0 ? 1.15 : 1);
      g.beginPath();
      for (let j = starts[b], end = starts[b] + counts[b]; j < end; j++) {
        const k = order[j],
          e = edges[k];
        g.moveTo(nodes[e.a].x, nodes[e.a].y);
        g.quadraticCurveTo(eCx[k], eCy[k], nodes[e.b].x, nodes[e.b].y);
      }
      g.stroke();
      if (tier === 0) {
        g.lineWidth *= 4.5;
        g.globalAlpha = TIER_A[0][level] * 0.13;
        g.stroke();
      } // soft glow around the primary structure
    }
  }
  g.globalCompositeOperation = 'lighter'; // glows and dots add up as light

  // ---- node glows: main nodes, bright nodes, and any node a wave is passing ----
  // glows are the most expensive part to draw, so there is a budget: main nodes first, then bright and wave-lit nodes
  const glowBudget = Math.round(34 + 40 * quality.fraction);
  let glows = 0;
  for (let i = 0, N = nodes.length; i < N; i++) {
    const n = nodes[i];
    let inten =
      (n.primary ? 0.5 : n.bright ? 0.38 : 0) + (n.e > 0.3 ? n.e * 0.9 : 0);
    if (n.role === 2) inten += 0.4 + 0.25 * Math.sin(t * 2.2); // the head's core pulses
    if (inten < 0.12 || (!n.primary && glows >= glowBudget)) continue;
    glows++;
    let eyeBlink = 1;
    if (n.role === 1)
      eyeBlink =
        creature.blinkTime > 0
          ? 1 - Math.sin(Math.PI * (1 - creature.blinkTime / 0.15)) * 0.9
          : 1;
    const s = (n.primary ? 44 : n.bright ? 28 : 22) * (1 + n.e * 1.0) * sc;
    g.globalAlpha = clamp(inten * bright, 0, 1) * eyeBlink;
    g.drawImage(GLOW[n.cidx], n.x - s / 2, n.y - s / 2, s, s);
  }

  // ---- node dots: grouped by colour; main nodes and joints are a little larger, some pulse ----
  g.globalAlpha = 0.95;
  let cur = -1;
  for (const i of web.drawOrder) {
    const n = nodes[i];
    if (n.cidx !== cur) {
      if (cur >= 0) g.fill();
      g.beginPath();
      g.fillStyle = PALETTE[n.cidx];
      cur = n.cidx;
    }
    const r =
      n.size *
      sc *
      (1 + (n.bright ? 0.35 : 0.12) * Math.sin(t * 1.6 + n.ph1) + n.e * 1.2);
    g.moveTo(n.x + r, n.y);
    g.arc(n.x, n.y, r, 0, Math.PI * 2);
  }
  if (cur >= 0) g.fill();
  // white-hot centres on bright nodes, and the two small "eyes" of the head
  g.fillStyle = '#ffffff';
  g.beginPath();
  for (let i = 0, N = nodes.length; i < N; i++) {
    const n = nodes[i];
    if (!n.bright && !n.primary) continue;
    const eyeOpen =
      n.role === 1
        ? creature.blinkTime > 0
          ? 1 - Math.sin(Math.PI * (1 - creature.blinkTime / 0.15))
          : 1
        : 1;
    const r =
      n.size *
      sc *
      (n.role === 1 ? 0.7 : 0.4) *
      (0.8 + 0.4 * Math.sin(t * 2 + n.ph2)) *
      (1 + n.e);
    g.moveTo(n.x + r, n.y);
    g.ellipse(
      n.x + (n.role === 1 ? creature.lookX * 0.4 : 0),
      n.y,
      r,
      Math.max(0.5, r * eyeOpen),
      0,
      0,
      Math.PI * 2
    );
  }
  g.globalAlpha = 0.85;
  g.fill();
  g.globalAlpha = 1;
  g.globalCompositeOperation = 'source-over';
}

// the soft light behind the creature + a faint pool of light under the feet
function drawAura() {
  const cx = sk.xf.cx,
    cy = sk.xf.cy - 40;
  const k =
    0.55 +
    0.12 * presenceE +
    0.25 * bloomE +
    0.2 * web.flash +
    0.05 * Math.sin(time * 0.9);
  const a = ctx.createRadialGradient(cx, cy, 0, cx, cy, 560);
  a.addColorStop(0, `rgba(110,90,255,${0.2 * k})`);
  a.addColorStop(0.5, `rgba(70,60,200,${0.08 * k})`);
  a.addColorStop(1, 'rgba(60,50,180,0)');
  ctx.fillStyle = a;
  ctx.fillRect(cx - 600, cy - 600, 1200, 1200);
  const heightAbove = -creature.y.pos + creature.hop.pos,
    s = clamp(1 - heightAbove / 800, 0.5, 1.1);
  ctx.fillStyle = `rgba(120,100,255,${0.1 * s})`;
  ctx.beginPath();
  ctx.ellipse(creature.x.pos, GROUND_Y, 230 * s, 22 * s, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = `rgba(150,130,255,${0.1 * s})`;
  ctx.beginPath();
  ctx.ellipse(creature.x.pos, GROUND_Y, 130 * s, 12 * s, 0, 0, Math.PI * 2);
  ctx.fill();
}

// ----------------------------------------------------------------
// 12b. PARTICLES: tiny floating lights + expanding rings (a fixed pool, nothing is created while running)
// ----------------------------------------------------------------
const motePool = Array.from({ length: CONFIG.particles }, () => ({
  alive: false,
  x: 0,
  y: 0,
  vx: 0,
  vy: 0,
  life: 0,
  maxLife: 1,
  size: 8,
  kind: 0,
  ci: 0,
}));
let moteIndex = 0;
function emitMote(x, y, vx, vy, size, life, kind, ci) {
  // kind 0 = floating light, 1 = ring
  const m = motePool[moteIndex];
  moteIndex = (moteIndex + 1) % motePool.length; // recycles the oldest
  m.alive = true;
  m.x = x;
  m.y = y;
  m.vx = vx;
  m.vy = vy;
  m.size = size;
  m.life = life;
  m.maxLife = life;
  m.kind = kind;
  m.ci = ci;
}
function emitRing(x, y, maxR) {
  emitMote(x, y, 0, 0, maxR, 1.1, 1, colorIndex(ZONE.HEAD, 0.8));
}
function burstMotes(count) {
  const n = Math.round(count * (0.5 + 0.5 * quality.fraction));
  for (let i = 0; i < n; i++) {
    const a = rand(0, Math.PI * 2),
      sp = rand(200, 520);
    emitMote(
      sk.xf.cx,
      sk.xf.cy - 60,
      Math.cos(a) * sp,
      Math.sin(a) * sp - 80,
      rand(6, 14),
      rand(1.2, 2.2),
      0,
      Math.floor(rand(0, PALETTE.length))
    );
  }
}

let ambAcc = 0,
  trailAcc = 0,
  bloomAcc = 0;
function updateEffects(dt) {
  const q = quality.fraction;
  updateWeb(dt);

  // a few particles float up from the network all the time
  ambAcc += dt * (present ? 1.3 : 2.4) * q;
  if (ambAcc >= 1 && web.ready) {
    ambAcc -= 1;
    const n = web.nodes[Math.floor(Math.random() * web.nodes.length)];
    emitMote(
      n.x,
      n.y,
      rand(-10, 10),
      rand(-40, -14),
      rand(4, 9),
      rand(3, 5),
      0,
      n.cidx
    );
  }
  // fast movement: the hands leave a trail of light
  if (present && scatter > 0.2) {
    trailAcc += dt * 9 * scatter * q;
    for (let n = 0; trailAcc >= 1 && n < 3; n++, trailAcc -= 1) {
      const h = Math.random() < 0.5 ? sk.arms.L.wrist : sk.arms.R.wrist;
      emitMote(
        h.x,
        h.y,
        rand(-60, 60),
        rand(-60, 30),
        rand(5, 11),
        rand(0.8, 1.5),
        0,
        colorIndex(ZONE.ARM, rand(0.5, 1))
      );
    }
  } else trailAcc = 0;
  // network expanded: light rises all around
  if (bloomE > 0.3) {
    bloomAcc += dt * 16 * bloomE * q;
    for (let n = 0; bloomAcc >= 1 && n < 3; n++, bloomAcc -= 1) {
      emitMote(
        sk.xf.cx +
          rand(-1, 1) * Math.min(320, (view.worldRight - view.worldLeft) * 0.5),
        sk.xf.cy + rand(-200, 240),
        rand(-20, 20),
        rand(-120, -50),
        rand(5, 11),
        rand(1.4, 2.4),
        0,
        Math.floor(rand(0, PALETTE.length))
      );
    }
  } else bloomAcc = 0;

  for (const m of motePool) {
    if (!m.alive) continue;
    m.life -= dt;
    if (m.life <= 0) {
      m.alive = false;
      continue;
    }
    const drag = Math.exp(-dt * 1.2);
    m.vx *= drag;
    m.vy *= drag;
    m.x += m.vx * dt;
    m.y += m.vy * dt;
  }
}
function drawParticles() {
  const g = ctx;
  for (const m of motePool) {
    if (!m.alive) continue;
    const age = 1 - m.life / m.maxLife,
      fade = clamp(m.life / 0.6, 0, 1);
    if (m.kind === 0) {
      const s = (m.size * 3 * sk.scale) / CREATURE_SCALE;
      g.globalAlpha = fade * 0.9 * (0.75 + 0.25 * Math.sin(time * 6 + m.x));
      g.drawImage(GLOW[m.ci], m.x - s / 2, m.y - s / 2, s, s);
    } else {
      // an expanding ring of light
      g.globalAlpha = (1 - age) * 0.5;
      g.strokeStyle = PALETTE[m.ci];
      g.lineWidth = 2.5 * (1 - age) + 0.5;
      g.beginPath();
      g.arc(m.x, m.y, m.size * smoothstep(age), 0, Math.PI * 2);
      g.stroke();
    }
  }
  g.globalAlpha = 1;
}

// ----------------------------------------------------------------
// 13. CREATURE BEHAVIOUR: idle <-> following the visitor, then the springs
// ----------------------------------------------------------------
function updateCreature(dt) {
  const T = control.target,
    p = presenceE,
    t = time,
    c = creature;
  const life = 1 + p * 0.5;
  const wobble = (phase, speed) => Math.sin(t * speed + phase) * 0.04;
  // Blend an idle angle with the visitor's angle (taking the shortest way round).
  const swing = (idle, pose, weight) =>
    idle + wrapAngle(pose - idle) * weight * p;

  // JUMP: the whole network stretches upward and rebounds (the bounce spring), waves rush up from the feet
  if (control.hopRequest) {
    if (present) {
      c.hop.vel += CONFIG.jumpKick * view.hopScale; // (lower on screens with little room above the head)
      webJump();
    }
    control.hopRequest = false;
  }

  // FAST MOVEMENT: slightly exaggerated + floppier
  const exaggerate = 1 + 0.12 * scatter,
    loose = 1 - 0.3 * scatter;
  for (const key of ['armL', 'armR']) {
    c[key].up.k = c[key].low.k = ARM_K * loose;
    c[key].up.d = c[key].low.d = ARM_D * (1 - 0.4 * scatter);
  }

  // Body: idle = floating in the centre. Tracking = follows the visitor. p blends smoothly between the two.
  c.x.update(
    lerp(
      W / 2 + Math.sin(t * 0.45) * 24 * life,
      clamp(W / 2 + (T.x - W / 2) * exaggerate, view.minX, view.maxX),
      p
    ),
    dt
  );
  c.y.update(lerp(Math.sin(t * 0.9) * 8 * life, T.y, p), dt);
  c.tilt.update(lerp(Math.sin(t * 0.6) * 0.03 * life, T.tilt, p), dt);
  c.bend.update(lerp(0, T.bend, p), dt);
  c.hop.update(0, dt);
  c.expand.update(bloomE * 0.04, dt);

  c.headX.update(
    lerp(Math.sin(t * 0.7) * 10 * life, T.headX, p) + noticeSide * notice * 26,
    dt
  );
  c.headRot.update(
    c.tilt.pos * 0.7 +
      Math.sin(t * 0.9 + 1) * 0.04 * life +
      c.headX.pos * 0.003 +
      noticeSide * notice * 0.2,
    dt
  );

  for (const [key, side, phase] of [
    ['armL', -1, 0],
    ['armR', 1, 1.7],
  ]) {
    const idleUp = side * (0.16 + 0.05 * Math.sin(t * 1.1 + phase) * life);
    const idleLow =
      side * (0.1 + 0.07 * Math.sin(t * 1.1 + phase + 0.8) * life);
    const pose = T[key];
    let up = swing(idleUp, pose.up, pose.weight),
      low = swing(idleLow, pose.low, pose.weight);
    if (bloomE > 0.01) {
      // both arms up: reach even higher (the network expands)
      const reach = side * (Math.PI - 0.32);
      up += wrapAngle(reach - up) * 0.3 * bloomE;
      low += wrapAngle(reach - low) * 0.3 * bloomE;
    }
    c[key].up.update(up + wobble(phase, 2.7), dt);
    c[key].low.update(low + wobble(phase + 1, 3.3), dt);
  }
  c.flopL.update(clamp(-c.armL.low.vel * 0.08, -0.55, 0.55), dt);
  c.flopR.update(clamp(-c.armR.low.vel * 0.08, -0.55, 0.55), dt);

  const crouch = T.crouch * p;
  for (const [key, side, phase] of [
    ['legL', -1, 0],
    ['legR', 1, 2.1],
  ]) {
    const idleUp = side * (0.07 + 0.015 * Math.sin(t * 1.1 + phase)),
      idleLow = side * 0.03;
    const pose = T[key];
    c[key].up.update(
      swing(idleUp, pose.up, pose.weight) + side * crouch * 0.7,
      dt
    );
    c[key].low.update(
      swing(idleLow, pose.low, pose.weight) - side * crouch * 0.7,
      dt
    );
  }

  // the two little "eyes" follow the head direction; a slow blink
  const ease = 1 - Math.exp(-dt * 5);
  c.lookX +=
    (T.headX * 0.1 * p +
      noticeSide * notice * 8 +
      Math.sin(t * 0.5) * 4 * (1 - p) -
      c.lookX) *
    ease;
  c.lookY += (Math.sin(t * 0.37) * 3 - 2 * p - c.lookY) * ease;
  c.blinkIn -= dt;
  if (c.blinkIn < 0) {
    c.blinkTime = 0.15;
    c.blinkIn = 2.8 + Math.random() * 3.5;
  }
  c.blinkTime = Math.max(0, c.blinkTime - dt);
}

// ---- HEADROOM CAMERA ----
// The creature is framed to be as large as possible, so there is only a little free room above its head. Normally
// the camera does NOT move at all (no pulsing). Only when raised hands / the bloom reach higher than that free room:
//   1. the picture first moves down a little (soft pan, the feet have spare room below), and
//   2. if that is still not enough, it zooms out just enough to keep the hands on screen.
// Both are slow and smooth, and both return exactly to normal when the arms come down. The camera only looks at how far
// the creature reaches above its OWN head height, so walking, jumping and bending never move the camera.
function updateCamera(dt) {
  const s = sk.scale;
  const topOfCreature = Math.min(
    sk.head.y - 96 * s, // top of the head mesh
    sk.handTip.L.y - 10 * s,
    sk.handTip.R.y - 10 * s,
    sk.arms.L.wrist.y - 10 * s,
    sk.arms.R.wrist.y - 10 * s
  );
  // how far the creature reaches above its resting head height (+ a little: the expanded network bulges above the rig)
  const reach =
    sk.xf.cy - topOfCreature - view.restAbove + bloomE * view.restAbove * 0.12;
  const pad = view.topPad;
  const half = view.visH / 2; // screen centre, in world units from the top of the screen
  const overflow = reach - (view.topMargin - pad); // the part that would leave the screen

  const pan = clamp(overflow, 0, view.maxPan);
  // zoom (about the screen centre) so that what is left of the overflow fits: top edge stays at least `pad` below the screen top
  const zoom = clamp(
    (half - pad) / Math.max(1, half - view.topMargin + reach - pan),
    0.6,
    1
  );
  const ease = 1 - Math.exp(-dt * 2.2);
  view.pan += (pan - view.pan) * ease;
  view.zoom += (zoom - view.zoom) * ease;
}

function update(dt, nowMs) {
  time += dt;
  updateInteraction(dt, nowMs);
  updateCreature(dt);
  updateSkeleton();
  updateCamera(dt);
  updateEffects(dt); // includes the network (updateWeb)
}

// ----------------------------------------------------------------
// 16. RENDER (back to front)
// ----------------------------------------------------------------
function render() {
  if (!view.background) return; // (window minimised: nothing laid out yet)
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(view.background, 0, 0);

  // from here on we draw in "world" coordinates, through the virtual camera (scale + centring + soft pan)
  // the camera: base scale + centring, plus the soft pan and the (rarely used) headroom zoom about the screen centre
  const z = view.zoom,
    cxs = canvas.width / 2,
    cys = canvas.height / 2;
  ctx.setTransform(
    view.scale * z,
    0,
    0,
    view.scale * z,
    cxs + (view.offsetX - cxs) * z,
    cys + (view.offsetY + view.pan * view.scale - cys) * z
  );
  drawAura();
  if (web.ready) {
    drawWeb(); // (switches between normal and additive blending by itself)
    ctx.globalCompositeOperation = 'lighter';
    drawParticles();
    ctx.globalCompositeOperation = 'source-over';
  }
  if (debugOn) drawControlPoints();

  // back to screen pixels for the camera-space debug skeleton
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (debugOn && present) drawDebugSkeleton();
}

// ----------------------------------------------------------------
// 15. DEVELOPMENT MODE DRAWING (only when debug mode is ON)
// ----------------------------------------------------------------
// Control points: the creature's own joints (the main nodes), in world coordinates.
function drawControlPoints() {
  const dot = (p, r, color) => {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fill();
  };
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(255,255,255,0.45)';
  for (const k of ['L', 'R']) {
    const A = sk.arms[k],
      Lg = sk.legs[k];
    ctx.beginPath();
    ctx.moveTo(A.shoulder.x, A.shoulder.y);
    ctx.lineTo(A.elbow.x, A.elbow.y);
    ctx.lineTo(A.wrist.x, A.wrist.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(Lg.hip.x, Lg.hip.y);
    ctx.lineTo(Lg.knee.x, Lg.knee.y);
    ctx.lineTo(Lg.ankle.x, Lg.ankle.y);
    ctx.stroke();
    dot(A.shoulder, 8, '#6ea8ff');
    dot(A.elbow, 8, '#6ea8ff');
    dot(A.wrist, 10, '#33e1ff');
    dot(Lg.hip, 8, '#7dffb0');
    dot(Lg.knee, 8, '#7dffb0');
    dot(Lg.ankle, 10, '#a6ffcf');
  }
  dot(sk.head, 10, '#ff6b8b');
  dot({ x: sk.xf.cx, y: sk.xf.cy }, 10, '#ff6b8b');
}

// The MediaPipe skeleton, drawn semi-transparent over the creature.
function drawDebugSkeleton() {
  const rect = getVideoRect();
  const isVisible = (i) =>
    smoothed[i].ready && smoothed[i].visibility >= CONFIG.minVisibility;
  const unit = rect.h / 480;
  const colorFor = (i) =>
    HEAD.includes(i)
      ? '#7ee8fa'
      : ARMS.includes(i)
      ? '#ffd166'
      : TORSO.includes(i)
      ? '#ff7aa2'
      : '#7ee8a9';

  ctx.globalAlpha = 0.65;
  ctx.lineCap = 'round';
  ctx.lineWidth = 4 * unit;
  ctx.strokeStyle = 'rgba(255,255,255,0.8)';
  for (const [a, b] of CONNECTIONS) {
    if (!isVisible(a) || !isVisible(b)) continue;
    const pa = toCanvas(smoothed[a], rect),
      pb = toCanvas(smoothed[b], rect);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
  for (let i = 0; i < 33; i++) {
    if (!isVisible(i)) continue;
    const p = toCanvas(smoothed[i], rect);
    ctx.fillStyle = colorFor(i);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6 * unit, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ----------------------------------------------------------------
// 17. DEVELOPMENT MODE: debug panel, FPS, automatic quality
// ----------------------------------------------------------------
let frameCount = 0;
let lastFpsTime = performance.now();
let lastPanelTime = 0;
let goodSeconds = 0;

// FPS counter + automatic quality: if the frame rate drops, DECORATIVE detail is reduced
// (faint threads, particles). Body tracking is not touched.
function updateFps(nowMs) {
  frameCount++;
  if (nowMs - lastFpsTime < 1000) return;
  state.fps = frameCount;
  frameCount = 0;
  lastFpsTime = nowMs;

  if (state.fps < 30 && quality.fraction > 0.4) {
    quality.fraction = Math.max(0.4, quality.fraction - 0.12);
    goodSeconds = 0;
  } else if (state.fps >= 57 && quality.fraction < 1) {
    if (++goodSeconds >= 4) {
      quality.fraction = Math.min(1, quality.fraction + 0.05);
      goodSeconds = 0;
    }
  }
}

function updateDebugPanel(nowMs) {
  if (nowMs - lastPanelTime < 250) return; // refresh text 4 times per second
  lastPanelTime = nowMs;
  debugPanel.textContent = [
    `Camera: ${state.camera}`,
    `Model:  ${state.model}${
      state.model === 'Ready'
        ? ' (' + state.delegate + ', ' + state.inferMs.toFixed(1) + ' ms)'
        : ''
    }`,
    `Pose:   ${state.poseDetected ? 'Detected' : 'Not detected'}`,
    `FPS:    ${state.fps}`,
    `Landmarks: ${state.visibleCount} / 33 visible`,
    `Network: ${web.nodes.length} nodes, ${web.edges.length} connections`,
    `State:  ${stateName}`,
    `Creature: ${Math.round(
      presenceE * 100
    )}% following visitor, expansion ${Math.round(bloomE * 100)}%`,
    `Detail quality: ${Math.round(quality.fraction * 100)}%`,
  ].join('\n');
}

function applyDebugVisibility() {
  debugPanel.classList.toggle('hidden', !debugOn);
  document.body.classList.toggle('debug', debugOn); // CSS shows the small camera preview in development mode
  refreshStatus();
}

// ----------------------------------------------------------------
// 18. FULLSCREEN, KEYS, CURSOR
// ----------------------------------------------------------------
function toggleFullscreen() {
  if (document.fullscreenElement) {
    if (document.exitFullscreen) document.exitFullscreen();
  } else if (document.documentElement.requestFullscreen) {
    const result = document.documentElement.requestFullscreen();
    if (result && result.catch) result.catch(() => {});
  }
}

window.addEventListener('keydown', (event) => {
  if (event.key === 'd' || event.key === 'D') {
    debugOn = !debugOn;
    applyDebugVisibility();
  }
  if (event.key === 'f' || event.key === 'F') toggleFullscreen();
});
window.addEventListener('dblclick', toggleFullscreen);

// the mouse cursor disappears after 2 seconds without movement (clean look on the projector)
let cursorTimer = null;
window.addEventListener('mousemove', () => {
  document.body.classList.remove('hide-cursor');
  clearTimeout(cursorTimer);
  cursorTimer = setTimeout(
    () => document.body.classList.add('hide-cursor'),
    2000
  );
});

// ----------------------------------------------------------------
// 19. MAIN LOOP + START-UP
// ----------------------------------------------------------------
let lastFrameMs = performance.now();

function loop() {
  const nowMs = performance.now();
  const dt = Math.min((nowMs - lastFrameMs) / 1000, 0.05);
  lastFrameMs = nowMs;

  // safety net: if the window / fullscreen size changed without a resize event (TV rotated, fullscreen on Windows), re-frame
  if (window.innerWidth !== view.winW || window.innerHeight !== view.winH)
    resizeCanvas();

  detectPose(nowMs); // MediaPipe (only when the camera has a new frame)
  update(dt, nowMs); // interaction states + creature + effects
  render();
  updateFps(nowMs);
  if (debugOn) updateDebugPanel(nowMs);
  requestAnimationFrame(loop);
}

async function start() {
  // 1) Camera
  setStatus('Starting camera...');
  try {
    await startCamera();
    state.camera = 'Ready';
  } catch (err) {
    console.error(err);
    state.camera = 'Error';
    setStatus(describeCameraError(err), true);
    return; // the creature keeps idling; nothing crashes
  }

  // 2) MediaPipe model
  setStatus('Loading pose model...');
  state.model = 'Loading...';
  try {
    await loadMediaPipe();
    state.model = 'Ready';
    hideStatus();
  } catch (err) {
    console.error(err);
    state.model = 'Error';
    setStatus(
      'Pose model failed to load: ' +
        (err && err.message ? err.message : err) +
        '  (check your internet connection)',
      true
    );
  }
}

resizeCanvas(); // first layout (needs the creature design numbers, so it runs here and not earlier)
applyDebugVisibility();
requestAnimationFrame(loop); // the creature is on screen immediately and idles while everything loads
start();
