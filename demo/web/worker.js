import * as ort from "./vendor/ort.wasm.bundle.min.mjs";
import { RmWebEngine } from "./engine.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;

let engine = null;
let paused = false;
let running = true;
let timer = null;
let stepMs = 0;

function postStatus(stage, detail = "") {
  self.postMessage({ type: "status", stage, detail });
}

function postState(state, elapsedMs = 0) {
  self.postMessage({
    type: "state",
    state: { ...state, paused },
    timings: {
      stepMs,
      roundTripMs: elapsedMs,
    },
  });
}

async function loadModel(buffer, name) {
  const started = performance.now();
  postStatus("model", `${name} 编译中`);
  const session = await ort.InferenceSession.create(buffer, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  return { session, ms: performance.now() - started };
}

async function initialize() {
  try {
    postStatus("fetch", "加载本地模型与初始状态");
    const [actorBuffer, envBuffer, fixtureResponse] = await Promise.all([
      fetch("./actor.onnx").then((response) => response.arrayBuffer()),
      fetch("./env_step.onnx").then((response) => response.arrayBuffer()),
      fetch("./fixture.json"),
    ]);
    const fixture = await fixtureResponse.json();
    const [actor, env] = await Promise.all([
      loadModel(actorBuffer, "actor"),
      loadModel(envBuffer, "environment"),
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
