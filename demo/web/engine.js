const OBS_DIM = 140;
const ENEMY_OBS_START = 24 + 4 + 4 + 2 * 18;
const ENEMY_OBS_DIM = 17;
const HIDDEN_DIM = 256;
const N_AGENTS = 6;

const STATE_INDEX = Object.freeze({
  pos: 0,
  yaw: 1,
  vel: 2,
  spin: 6,
  hp: 8,
  heat: 9,
  overheated: 10,
  alive: 11,
  invuln: 12,
  invulnTimer: 13,
  respawnTimer: 14,
  target: 15,
  lastFired: 20,
  score: 23,
  time: 24,
  occ: 32,
  outcome: 33,
  winner: 34,
  done: 40,
});

const OUTCOME_NAMES = Object.freeze({
  0: "进行中",
  1: "红方先到目标分",
  2: "蓝方先到目标分",
  3: "红方时间胜",
  4: "蓝方时间胜",
  5: "红方伤害 tiebreak",
  6: "蓝方伤害 tiebreak",
  7: "红方血量 tiebreak",
  8: "蓝方血量 tiebreak",
  9: "红方胜（蓝方碰撞犯规）",
  10: "蓝方胜（红方碰撞犯规）",
  20: "平局（时间 tiebreak）",
  21: "平局（双方同帧目标分）",
  22: "平局（双方犯规）",
  23: "空局（10 秒无事件）",
  24: "红方胜（蓝方推墙犯规）",
  25: "蓝方胜（红方推墙犯规）",
  26: "双方推墙犯规",
  27: "红方胜（蓝方推人犯规）",
  28: "蓝方胜（红方推人犯规）",
  29: "双方推人犯规",
});

const NAMES = Object.freeze(["R1", "R2", "R3", "B1", "B2", "B3"]);
const ENEMIES = Object.freeze({
  0: [3, 4, 5],
  1: [3, 4, 5],
  2: [3, 4, 5],
  3: [0, 1, 2],
  4: [0, 1, 2],
  5: [0, 1, 2],
});

const DTYPE_CTORS = {
  float32: Float32Array,
  float64: Float64Array,
  int8: Int8Array,
  uint8: Uint8Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  bool: Uint8Array,
};

function decodeBase64(value) {
  const binary = globalThis.atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function typedArrayFromBase64(encoded) {
  const bytes = decodeBase64(encoded.data);
  const Ctor = DTYPE_CTORS[encoded.dtype];
  if (!Ctor) throw new Error(`unsupported fixture dtype: ${encoded.dtype}`);
  if (bytes.byteOffset % Ctor.BYTES_PER_ELEMENT !== 0) {
    return new Ctor(bytes.slice().buffer);
  }
  return new Ctor(
    bytes.buffer,
    bytes.byteOffset,
    bytes.byteLength / Ctor.BYTES_PER_ELEMENT,
  );
}

function makeTensor(ort, encoded) {
  return new ort.Tensor(
    encoded.dtype,
    typedArrayFromBase64(encoded),
    encoded.shape,
  );
}

function wrapAngle(value) {
  return ((value + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  const u = Math.max(random(), 1e-9);
  const v = Math.max(random(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function asFloat(tensor) {
  return tensor.data;
}

function asInt(tensor) {
  return tensor.data;
}

function asBool(tensor) {
  return tensor.data;
}

export class RmWebEngine {
  constructor({ ort, actorSession, envSession, fixture, noise = true }) {
    this.ort = ort;
    this.actorSession = actorSession;
    this.envSession = envSession;
    this.fixture = fixture;
    this.noise = noise;
    this.human = 0;
    this.ai = true;
    this.keys = {
      w: false,
      a: false,
      s: false,
      d: false,
      shift: false,
    };
    this.aimAngle = 0;
    this.lockDown = false;
    this.fireDown = false;
    this.fogEnabled = true;
    this.enemyTruth = false;
    this.spectating = true;
    this.lockedTarget = 0;
    this.stepIndex = 0;
    this.events = [];
    this.reset();
  }

  reset() {
    this.stepIndex = 0;
    this.done = false;
    this.lockedTarget = 0;
    this.state = this.fixture.state.map((value) =>
      makeTensor(this.ort, value));
    this.obs = makeTensor(this.ort, this.fixture.obs);
    this.hidden = makeTensor(this.ort, this.fixture.hidden);
    this.events = [];
    this.lastScore = [0, 0];
    this.lastAlive = [true, true, true, true, true, true];
    this.cleanObs = new Float32Array(asFloat(this.obs));
    this.applyObservationNoise();
    this.actionMasks = this.computeActionMasks();
    return this.snapshot();
  }

  setControls(controls) {
    if (controls.keys) {
      for (const key of Object.keys(this.keys)) {
        if (key in controls.keys) this.keys[key] = Boolean(controls.keys[key]);
      }
    }
    const aimAngle = controls.aimAngle ?? controls.aim_angle;
    const lockDown = controls.lockDown ?? controls.lock_down;
    const fireDown = controls.fireDown ?? controls.fire_down;
    if (aimAngle !== undefined) this.aimAngle = Number(aimAngle);
    if (lockDown !== undefined) this.lockDown = Boolean(lockDown);
    if (fireDown !== undefined) this.fireDown = Boolean(fireDown);
    const fogEnabled = controls.fogEnabled ?? controls.fog_enabled;
    const enemyTruth = controls.enemyTruth ?? controls.enemy_truth;
    if (fogEnabled !== undefined) this.fogEnabled = Boolean(fogEnabled);
    if (enemyTruth !== undefined) this.enemyTruth = Boolean(enemyTruth);
    const spectating = controls.spectating;
    if (spectating !== undefined) {
      this.spectating = Boolean(spectating);
      if (this.spectating) {
        for (const key of Object.keys(this.keys)) this.keys[key] = false;
        this.fireDown = false;
        this.lockDown = false;
        this.lockedTarget = 0;
      }
    }
    if (controls.human !== undefined) {
      const next = Math.max(0, Math.min(5, Number(controls.human) | 0));
      if (next !== this.human) {
        this.human = next;
        this.lockedTarget = 0;
        this.lockDown = false;
        this.fireDown = false;
      }
    }
    if (controls.ai !== undefined) this.ai = Boolean(controls.ai);
  }

  team(index) {
    return index < 3 ? 0 : 1;
  }

  applyObservationNoise() {
    this.obs = new this.ort.Tensor(
      "float32",
      new Float32Array(this.cleanObs),
      [N_AGENTS, OBS_DIM],
    );
    if (!this.noise) return;

    const cfg = this.fixture.config;
    const data = asFloat(this.obs);
    const spin = asFloat(this.state[STATE_INDEX.spin]);
    const yaw = asFloat(this.state[STATE_INDEX.yaw]);
    const lastFired = asBool(this.state[STATE_INDEX.lastFired]);
    const random = mulberry32(
      (this.fixture.seed * 2654435761 + this.stepIndex * 2246822519) >>> 0,
    );
    for (let i = 0; i < N_AGENTS; i++) {
      for (let slot = 0; slot < 3; slot++) {
        const base = i * OBS_DIM + ENEMY_OBS_START + slot * ENEMY_OBS_DIM;
        if (data[base] <= 0.5 || data[base + 14] <= 0.5) continue;
        const enemy = ENEMIES[i][slot];
        const hpSigma = cfg.hp_noise_base + cfg.hp_noise_spin * spin[enemy];
        const hp = Math.round(
          Math.max(
            0,
            Math.min(
              cfg.hp_max,
              asFloat(this.state[STATE_INDEX.hp])[enemy] +
                gaussian(random) * hpSigma,
            ),
          ),
        );
        data[base + 7] = hp / cfg.hp_max;
        let dyaw = wrapAngle(yaw[enemy] - yaw[i]);
        if (!lastFired[enemy]) {
          const yawNoise =
            (random() * 2 - 1) * cfg.yaw_noise_deg * Math.PI / 180;
          dyaw = wrapAngle(dyaw + yawNoise);
        }
        data[base + 9] = Math.sin(dyaw);
        data[base + 10] = Math.cos(dyaw);
      }
    }
  }

  computeActionMasks() {
    const cfg = this.fixture.config;
    const obs = asFloat(this.obs);
    const pos = asFloat(this.state[STATE_INDEX.pos]);
    const alive = asBool(this.state[STATE_INDEX.alive]);
    const masks = Array.from({ length: N_AGENTS }, () => [true, false, false, false]);
    for (let i = 0; i < N_AGENTS; i++) {
      for (let slot = 0; slot < 3; slot++) {
        const enemy = ENEMIES[i][slot];
        const visible =
          obs[i * OBS_DIM + ENEMY_OBS_START + slot * ENEMY_OBS_DIM] > 0.5;
        const dx = pos[enemy * 2] - pos[i * 2];
        const dy = pos[enemy * 2 + 1] - pos[i * 2 + 1];
        masks[i][slot + 1] =
          visible &&
          Boolean(alive[enemy]) &&
          Math.hypot(dx, dy) < cfg.lock_range;
      }
    }
    return masks;
  }

  computeFireMasks() {
    const cfg = this.fixture.config;
    const state = this.state;
    const heat = asFloat(state[STATE_INDEX.heat]);
    const overheated = asBool(state[STATE_INDEX.overheated]);
    const alive = asBool(state[STATE_INDEX.alive]);
    const invuln = asFloat(state[STATE_INDEX.invuln]);
    const target = asInt(state[STATE_INDEX.target]);
    return Array.from({ length: N_AGENTS }, (_, i) => {
      let fire = false;
      if (target[i] > 0) {
        const enemy = ENEMIES[i][target[i] - 1];
        fire =
          Boolean(alive[i]) &&
          invuln[i] <= 0 &&
          !Boolean(overheated[i]) &&
          heat[i] < cfg.heat_max &&
          invuln[enemy] <= 0;
      }
      return [true, fire];
    });
  }

  decodeActorActions(raw, actionMasks, fireMasks) {
    const actions = new Float32Array(N_AGENTS * 6);
    for (let i = 0; i < N_AGENTS; i++) {
      const base = i * 14;
      for (let j = 0; j < 4; j++) {
        const loc = 3 * Math.tanh(raw[base + j] / 3);
        actions[i * 6 + j] = Math.tanh(loc);
      }
      let bestTarget = 0;
      for (let k = 1; k < 4; k++) {
        if (
          actionMasks[i][k] &&
          raw[base + 8 + k] > raw[base + 8 + bestTarget]
        ) {
          bestTarget = k;
        }
      }
      let bestFire = 0;
      if (fireMasks[i][1] && raw[base + 13] > raw[base + 12]) bestFire = 1;
      actions[i * 6 + 4] = bestTarget;
      actions[i * 6 + 5] = bestFire;
    }
    return actions;
  }

  humanAction(actionMasks) {
    const action = new Float32Array(6);
    const state = this.state;
    const i = this.human;
    const sign = this.team(i) === 0 ? 1 : -1;
    let vx = (this.keys.d ? 1 : 0) - (this.keys.a ? 1 : 0);
    let vy = (this.keys.w ? 1 : 0) - (this.keys.s ? 1 : 0);
    const norm = Math.hypot(vx, vy);
    if (norm > 1) {
      vx /= norm;
      vy /= norm;
    }
    action[0] = vx * sign;
    action[1] = vy * sign;
    action[3] = this.keys.shift ? 1 : 0;

    if (this.lockedTarget > 0 && !actionMasks[i][this.lockedTarget]) {
      this.lockedTarget = 0;
    }
    if (this.lockDown) {
      const pos = asFloat(state[STATE_INDEX.pos]);
      let best = 0;
      let bestError = Math.PI + 1;
      for (let slot = 1; slot <= 3; slot++) {
        if (!actionMasks[i][slot]) continue;
        const enemy = ENEMIES[i][slot - 1];
        const angle = Math.atan2(
          pos[enemy * 2 + 1] - pos[i * 2 + 1],
          pos[enemy * 2] - pos[i * 2],
        );
        const error = Math.abs(wrapAngle(angle - this.aimAngle));
        if (error < bestError) {
          bestError = error;
          best = slot;
        }
      }
      this.lockedTarget = best;
    }

    if (this.lockedTarget > 0) {
      action[2] = 0;
      action[4] = this.lockedTarget;
      action[5] = this.fireDown ? 1 : 0;
    } else {
      const yaw = asFloat(state[STATE_INDEX.yaw])[i];
      const error = wrapAngle(this.aimAngle - yaw);
      const maxStep =
        this.fixture.config.yaw_rate_max * this.fixture.config.dt;
      action[2] = Math.max(-1, Math.min(1, error / maxStep));
      action[4] = 0;
      action[5] = 0;
    }
    return action;
  }

  envFeeds(actions) {
    const feeds = {};
    const inputNames = this.envSession.inputNames;
    for (let i = 0; i < this.state.length; i++) {
      feeds[inputNames[i]] = this.state[i];
    }
    feeds[inputNames[inputNames.length - 1]] = new this.ort.Tensor(
      "float32",
      actions,
      [N_AGENTS, 6],
    );
    return feeds;
  }

  async step() {
    if (this.done) return this.snapshot();

    this.actionMasks = this.computeActionMasks();
    const fireMasks = this.computeFireMasks();
    const actorFeeds = {
      [this.actorSession.inputNames[0]]: this.obs,
      [this.actorSession.inputNames[1]]: this.hidden,
    };
    const actorOut = await this.actorSession.run(actorFeeds);
    const raw = asFloat(actorOut[this.actorSession.outputNames[0]]);
    this.lastRaw = new Float32Array(raw);
    this.hidden = actorOut[this.actorSession.outputNames[1]];

    const actions = this.decodeActorActions(raw, this.actionMasks, fireMasks);
    if (!this.spectating) {
      const human = this.humanAction(this.actionMasks);
      if (!this.ai) actions.fill(0);
      actions.set(human, this.human * 6);
    }
    this.lastActions = new Float32Array(actions);

    const envOut = await this.envSession.run(this.envFeeds(actions));
    const nextState = [];
    for (let i = 0; i < this.state.length; i++) {
      nextState.push(envOut[this.envSession.outputNames[i]]);
    }
    this.cleanObs = new Float32Array(
      asFloat(envOut[this.envSession.outputNames[this.state.length]]),
    );
    this.done = Boolean(
      asBool(envOut[this.envSession.outputNames[this.state.length + 1]])[0],
    );
    this.state = nextState;
    this.obs = new this.ort.Tensor("float32", this.cleanObs, [N_AGENTS, OBS_DIM]);
    this.applyObservationNoise();
    this.stepIndex += 1;
    this.updateEvents();
    return this.snapshot();
  }

  updateEvents() {
    const score = asFloat(this.state[STATE_INDEX.score]);
    const alive = asBool(this.state[STATE_INDEX.alive]);
    const time = asFloat(this.state[STATE_INDEX.time])[0];
    for (let team = 0; team < 2; team++) {
      if (score[team] > this.lastScore[team] + 0.01) {
        this.events.unshift({
          t: Number(time.toFixed(1)),
          kind: "得分",
          text: `${team === 0 ? "红方" : "蓝方"} ${score[team].toFixed(1)}`,
        });
      }
    }
    for (let i = 0; i < N_AGENTS; i++) {
      if (this.lastAlive[i] && !Boolean(alive[i])) {
        this.events.unshift({
          t: Number(time.toFixed(1)),
          kind: "击杀",
          text: `${NAMES[i]} 阵亡`,
        });
      }
    }
    this.lastScore = [score[0], score[1]];
    this.lastAlive = Array.from(alive, Boolean);
    if (this.done) {
      this.events.unshift({
        t: Number(time.toFixed(1)),
        kind: "终局",
        text: this.outcomeName(),
      });
    }
    this.events = this.events.slice(0, 30);
  }

  outcomeName() {
    const outcome = asInt(this.state[STATE_INDEX.outcome])[0];
    return OUTCOME_NAMES[outcome] || `终局 ${outcome}`;
  }

  snapshot() {
    const state = this.state;
    const cfg = this.fixture.config;
    const pos = asFloat(state[STATE_INDEX.pos]);
    const yaw = asFloat(state[STATE_INDEX.yaw]);
    const vel = asFloat(state[STATE_INDEX.vel]);
    const spin = asFloat(state[STATE_INDEX.spin]);
    const hp = asFloat(state[STATE_INDEX.hp]);
    const heat = asFloat(state[STATE_INDEX.heat]);
    const overheated = asBool(state[STATE_INDEX.overheated]);
    const alive = asBool(state[STATE_INDEX.alive]);
    const invuln = asFloat(state[STATE_INDEX.invuln]);
    const invulnTimer = asFloat(state[STATE_INDEX.invulnTimer]);
    const respawnTimer = asFloat(state[STATE_INDEX.respawnTimer]);
    const target = asInt(state[STATE_INDEX.target]);
    const lastFired = asBool(state[STATE_INDEX.lastFired]);
    const score = asFloat(state[STATE_INDEX.score]);
    const time = asFloat(state[STATE_INDEX.time])[0];
    const occ = asFloat(state[STATE_INDEX.occ]);
    const obs = asFloat(this.obs);
    const human = this.human;
    const humanTeam = this.team(human);
    const robots = [];

    for (let i = 0; i < N_AGENTS; i++) {
      const sameTeam = this.team(i) === humanTeam;
      let visible = i === human || sameTeam;
      let slot = -1;
      if (!visible) {
        slot = ENEMIES[human].indexOf(i);
        const observed =
          obs[human * OBS_DIM + ENEMY_OBS_START + slot * ENEMY_OBS_DIM] > 0.5;
        visible = observed || !this.fogEnabled;
        if (!visible) {
          robots.push({ name: NAMES[i], team: this.team(i), visible: false });
          continue;
        }
        if (this.enemyTruth) {
          robots.push({
            name: NAMES[i],
            team: this.team(i),
            x: pos[i * 2],
            y: pos[i * 2 + 1],
            vx: vel[i * 2],
            vy: vel[i * 2 + 1],
            yaw: yaw[i],
            hp: hp[i],
            heat: heat[i],
            heat_known: true,
            info_known: true,
            overheated: Boolean(overheated[i]),
            spin: spin[i],
            alive: Boolean(alive[i]),
            invuln: invuln[i] > 0.5,
            invuln_timer: invulnTimer[i],
            respawn_timer: respawnTimer[i],
            target: target[i],
            p_t: asFloat(state[18])[i],
            fired: Boolean(lastFired[i]),
            visible: true,
            observed,
          });
          continue;
        }
        if (!observed) {
          robots.push({
            name: NAMES[i],
            team: this.team(i),
            x: pos[i * 2],
            y: pos[i * 2 + 1],
            vx: 0,
            vy: 0,
            yaw: yaw[i],
            hp: null,
            heat: null,
            heat_known: false,
            info_known: false,
            overheated: false,
            spin: 0,
            alive: Boolean(alive[i]),
            invuln: false,
            invuln_timer: 0,
            respawn_timer: 0,
            target: 0,
            p_t: 0,
            fired: false,
            visible: true,
            observed: false,
          });
          continue;
        }
      }
      if (i !== human && !sameTeam) {
        const base = human * OBS_DIM + ENEMY_OBS_START + slot * ENEMY_OBS_DIM;
        const sign = humanTeam === 0 ? 1 : -1;
        const x = sign * obs[base + 3] * cfg.field_hx;
        const y = sign * obs[base + 4] * cfg.field_hy;
        const rvx = obs[base + 5] * 2 * cfg.v_max;
        const rvy = obs[base + 6] * 2 * cfg.v_max;
        const dyaw = Math.atan2(obs[base + 9], obs[base + 10]);
        robots.push({
          name: NAMES[i],
          team: this.team(i),
          x,
          y,
          vx: vel[human * 2] + sign * rvx,
          vy: vel[human * 2 + 1] + sign * rvy,
          yaw: yaw[human] + dyaw,
          hp: obs[base + 7] * cfg.hp_max,
          heat: null,
          heat_known: false,
          info_known: true,
          overheated: false,
          spin: obs[base + 11],
          alive: obs[base + 14] > 0.5,
          invuln: obs[base + 8] > 0.5,
          invuln_timer: 0,
          respawn_timer: obs[base + 16] * cfg.respawn_time,
          target: 0,
          p_t: 0,
          fired: obs[base + 12] > 0.5,
          visible: true,
          observed: true,
        });
      } else {
        robots.push({
          name: NAMES[i],
          team: this.team(i),
          x: pos[i * 2],
          y: pos[i * 2 + 1],
          vx: vel[i * 2],
          vy: vel[i * 2 + 1],
          yaw: yaw[i],
          hp: hp[i],
          heat: heat[i],
          heat_known: true,
          info_known: true,
          overheated: Boolean(overheated[i]),
          spin: spin[i],
          alive: Boolean(alive[i]),
          invuln: invuln[i] > 0.5,
          invuln_timer: invulnTimer[i],
          respawn_timer: respawnTimer[i],
          target: target[i],
          p_t: asFloat(state[18])[i],
          fired: Boolean(lastFired[i]),
          visible: true,
          observed: false,
        });
      }
    }

    return {
      ok: true,
      mode: "fixed",
      seed: this.fixture.seed,
      human,
      human_name: NAMES[human],
      ai: this.ai,
      spectating: this.spectating,
      terminal_fouls: false,
      paused: false,
      done: this.done,
      time,
      score: [score[0], score[1]],
      occ: [Math.round(occ[0]), Math.round(occ[1])],
      outcome: asInt(state[STATE_INDEX.outcome])[0],
      outcome_name: this.outcomeName(),
      winner: asInt(state[STATE_INDEX.winner])[0],
      robots,
      mask: this.actionMasks[human].map(Number),
      target: this.lockedTarget,
      fire_down: this.fireDown,
      lock_down: this.lockDown,
      fog_enabled: this.fogEnabled,
      enemy_truth: this.enemyTruth,
      events: this.events,
    };
  }
}

export { NAMES, ENEMIES };
