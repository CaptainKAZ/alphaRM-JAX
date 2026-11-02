const EMBEDDED_ASSETS = globalThis.RM_EMBEDDED_ASSETS;
const textDecoder = new TextDecoder();

let engine = null;
let paused = false;
let running = true;
let timer = null;
let stepMs = 0;

function decodeBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function decodeText(value) {
  return textDecoder.decode(decodeBytes(value));
}

function moduleUrl(source, type = "text/javascript") {
  return URL.createObjectURL(new Blob([source], { type }));
}

function postStatus(stage, detail = "") {
  self.postMessage({ type: "status", stage, detail });
}

function postState(state, elapsedMs = 0) {
  self.postMessage({
    type: "state",
    state: { ...state, paused },
    timings: { stepMs, roundTripMs: elapsedMs },
  });
}

async function loadModel(ort, bytes, name) {
  const started = performance.now();
  postStatus("model", `${name} 编译中`);
  const buffer = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  );
  const session = await ort.InferenceSession.create(buffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  return { session, ms: performance.now() - started };
}

async function tick() {
  timer = null;
  if (!running) return;
  if (!engine) {
    timer = setTimeout(tick, 100);
    return;
  }
  if (!paused && !engine.done) {
    const started = performance.now();
    try {
      const state = await engine.step();
      stepMs = performance.now() - started;
      state.paused = paused;
      postState(state, stepMs);
    } catch (error) {
      paused = true;
      postStatus("error", String(error?.stack || error));
    }
  }
  schedule(Math.max(0, 100 - stepMs));
}

function schedule(delay) {
  if (!running || timer !== null) return;
  timer = setTimeout(tick, delay);
}

async function initialize() {
  try {
    postStatus("fetch", "解包本地 ORT 与模型");
    const ortUrl = moduleUrl(decodeText(EMBEDDED_ASSETS.ortBundle));
    const ort = await import(ortUrl);
    URL.revokeObjectURL(ortUrl);

    const loaderUrl = moduleUrl(decodeBytes(EMBEDDED_ASSETS.ortLoader));
    const wasmUrl = URL.createObjectURL(
      new Blob([decodeBytes(EMBEDDED_ASSETS.ortWasm)], {
        type: "application/wasm",
      }),
    );
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.wasmPaths = { mjs: loaderUrl, wasm: wasmUrl };

    const engineUrl = moduleUrl(decodeText(EMBEDDED_ASSETS.engine));
    const { RmWebEngine } = await import(engineUrl);
    URL.revokeObjectURL(engineUrl);

    const actorBytes = decodeBytes(EMBEDDED_ASSETS.actor);
    const envBytes = decodeBytes(EMBEDDED_ASSETS.env);
    const fixture = JSON.parse(decodeText(EMBEDDED_ASSETS.fixture));
    const [actor, env] = await Promise.all([
      loadModel(ort, actorBytes, "actor"),
      loadModel(ort, envBytes, "environment"),
    ]);
    engine = new RmWebEngine({
      ort,
      actorSession: actor.session,
      envSession: env.session,
      fixture,
      noise: true,
    });
    postStatus(
      "ready",
      `actor ${actor.ms.toFixed(0)} ms · env ${env.ms.toFixed(0)} ms`,
    );
    const state = engine.snapshot();
    state.paused = paused;
    postState(state);
    schedule(0);
  } catch (error) {
    postStatus("error", String(error?.stack || error));
  }
}

self.onmessage = async (event) => {
  const message = event.data || {};
  if (message.type === "input") {
    engine?.setControls(message.value || {});
    if (
      engine &&
      (
        "fog_enabled" in (message.value || {}) ||
        "enemy_truth" in (message.value || {}) ||
        "spectating" in (message.value || {})
      )
    ) {
      const state = engine.snapshot();
      state.paused = paused;
      postState(state);
    }
    return;
  }
  if (message.type === "pause") {
    paused = Boolean(message.value);
    if (engine) {
      const state = engine.snapshot();
      state.paused = paused;
      postState(state);
    }
    return;
  }
  if (message.type === "step") {
    if (!engine || engine.done) return;
    paused = true;
    const started = performance.now();
    try {
      const state = await engine.step();
      stepMs = performance.now() - started;
      state.paused = paused;
      postState(state, stepMs);
    } catch (error) {
      postStatus("error", String(error?.stack || error));
    }
    return;
  }
  if (message.type === "reset") {
    if (!engine) return;
    paused = false;
    engine.reset();
    engine.setControls(message.value || {});
    const state = engine.snapshot();
    state.paused = paused;
    postState(state);
    schedule(0);
    return;
  }
  if (message.type === "close") {
    running = false;
    if (timer !== null) clearTimeout(timer);
    self.close();
  }
};

initialize();
