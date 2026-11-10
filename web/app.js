// The c4a0 browser player: the Rust search (WebAssembly) asks for one position at a time and the
// network answers on the GPU (net.js). The page re-renders from `state` after every change.

import init, { Search, legal_moves, outcome } from "./pkg/c4a0_web.js";
import { loadNet, NoWebGPU, selfCheck } from "./net.js";

const ROWS = 6;
const COLS = 7;
// The agent behind the README results table: 256 iterations, mirror averaging (net.js) and a
// 10-ply exact search at every leaf.
const params = new URLSearchParams(location.search);
const ITERATIONS = Number(params.get("iterations") ?? 256);
const LEAF_DEPTH = Number(params.get("depth") ?? 10);
const RENDER_EVERY = 8; // iterations between redraws while thinking
// Largest gap from PyTorch's outputs accepted on this GPU; float32 reorderings stay near 1e-5.
const TOLERANCE = 1e-3;

const state = {
  phase: "loading", // loading | human | thinking | over | error
  moves: "", // 0-indexed columns, first player first
  human: 0, // 0: the human plays first
  search: null, // c4a0's latest search: see `think`
  history: [], // { ply, qPenalty, qNoPenalty, proven } at each c4a0 move
  message: "",
};

let net;

const $ = (id) => document.getElementById(id);

function board(moves) {
  const cells = Array.from({ length: ROWS }, () => Array(COLS).fill(-1)); // row 0 = bottom
  [...moves].forEach((ch, ply) => {
    const c = Number(ch);
    const r = cells.findIndex((row) => row[c] === -1);
    cells[r][c] = ply % 2;
  });
  return cells;
}

/** The four cells of the winning line through the last stone, for highlighting. */
function winningLine(cells, moves) {
  if (!moves) return [];
  const c = Number(moves.at(-1));
  let r = ROWS - 1;
  while (cells[r][c] === -1) r--;
  const who = cells[r][c];
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const line = [[r, c]];
    for (const s of [1, -1]) {
      for (let k = 1; k < 4; k++) {
        const [rr, cc] = [r + s * k * dr, c + s * k * dc];
        if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || cells[rr][cc] !== who) break;
        line.push([rr, cc]);
      }
    }
    if (line.length >= 4) return line;
  }
  return [];
}

/**
 * c4a0's two root values in words. Both are from c4a0's side. A decisive result that ends on
 * ply n scores q_penalty = ±(1 - 0.01 n), so once proven the ending move is exact, and before
 * that q_penalty / q_no_penalty estimates it.
 */
function assess(s) {
  const penalty = Search.ply_penalty();
  const { qPenalty: qp, qNoPenalty: qn } = s;
  if (s.proven) {
    const end = Math.round((1 - Math.abs(qp)) / penalty);
    if (qn === 0) return "Proven draw: neither side can force a win.";
    return qn > 0
      ? `Proven: c4a0 wins by move ${end}, whatever you play.`
      : `Proven: you can win by move ${end}. c4a0 will make it as long as it can.`;
  }
  if (Math.abs(qn) < 0.3) return "c4a0 sees an even game.";
  const end = Math.min(42, Math.max(s.ply + 1, Math.round((1 - qp / qn) / penalty)));
  return `c4a0 expects to ${qn > 0 ? "win" : "lose"}, around move ${end}.`;
}

const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

/** A bar from the center: right toward c4a0 for positive values, left toward you. */
function meter(el, v) {
  const fill = el.querySelector(".fill");
  const x = v ?? 0;
  fill.style.left = `${50 + Math.min(x, 0) * 50}%`;
  fill.style.width = `${Math.abs(x) * 50}%`;
  fill.dataset.side = x >= 0 ? "them" : "you";
  el.querySelector(".num").textContent = v === undefined ? "–" : signed(v);
}

function render() {
  const cells = board(state.moves);
  const result = state.phase === "over" ? outcome(state.moves) : "playing";
  const line = result === "win" ? winningLine(cells, state.moves) : [];
  const legal = state.phase === "human" ? new Set(legal_moves(state.moves)) : new Set();
  const last = state.moves ? Number(state.moves.at(-1)) : -1;

  const boardEl = $("board");
  boardEl.replaceChildren(
    ...Array.from({ length: COLS }, (_, c) => {
      const col = document.createElement("button");
      col.className = "col";
      col.disabled = !legal.has(c);
      col.setAttribute("aria-label", `Drop in column ${c + 1}`);
      col.onclick = () => humanMove(c);
      for (let r = ROWS - 1; r >= 0; r--) {
        const cell = document.createElement("span");
        const who = cells[r][c];
        cell.className = "cell" + (who === 0 ? " first" : who === 1 ? " second" : "");
        if (line.some(([lr, lc]) => lr === r && lc === c)) cell.classList.add("won");
        if (who !== -1 && c === last && (r === ROWS - 1 || cells[r + 1][c] === -1)) cell.classList.add("last");
        col.append(cell);
      }
      const ghost = document.createElement("span");
      ghost.className = "ghost " + (state.moves.length % 2 === 0 ? "first" : "second");
      const free = cells.findIndex((row) => row[c] === -1);
      ghost.style.setProperty("--row", String(ROWS - 1 - free));
      col.append(ghost);
      return col;
    }),
  );

  const s = state.search;
  $("bars").replaceChildren(
    ...Array.from({ length: COLS }, (_, c) => {
      const bar = document.createElement("div");
      bar.className = "bar";
      if (s) {
        const prior = document.createElement("span");
        prior.className = "prior";
        prior.style.height = `${(s.prior?.[c] ?? 0) * 100}%`;
        const visits = document.createElement("span");
        visits.className = "visits";
        visits.style.height = `${s.policy[c] * 100}%`;
        const pct = document.createElement("span");
        pct.className = "pct";
        pct.textContent = s.policy[c] >= 0.005 ? `${Math.round(s.policy[c] * 100)}%` : "";
        bar.append(prior, visits, pct);
      }
      return bar;
    }),
  );

  const you = state.human === 0 ? "first" : "second";
  $("you").className = `swatch ${you}`;
  $("them").className = `swatch ${you === "first" ? "second" : "first"}`;
  const status = {
    loading: "Loading the network…",
    human: "Your move",
    thinking: "c4a0 is thinking",
    over: result === "draw" ? "Draw" : state.moves.length % 2 === (state.human === 0 ? 1 : 0) ? "You win!" : "c4a0 wins",
    error: "Can't run here",
  }[state.phase];
  $("status").textContent = status;
  $("iterations").textContent = s ? `${s.visits}` : "0";
  $("move").textContent = state.moves.length ? `Move ${state.moves.length}` : "";
  meter($("result"), s?.qNoPenalty);
  meter($("steer"), s?.qPenalty);
  $("summary").textContent = s ? assess(s) : "c4a0's evaluation appears once it starts thinking.";
  document.querySelector(".history-wrap").hidden = state.history.length === 0;
  $("history").replaceChildren(
    ...state.history.map((h) => {
      const tick = document.createElement("span");
      tick.className = "tick";
      tick.dataset.side = h.qNoPenalty >= 0 ? "them" : "you";
      tick.style.setProperty("--v", String(Math.abs(h.qNoPenalty)));
      tick.title = `Move ${h.ply + 1}: ${signed(h.qNoPenalty)}`;
      return tick;
    }),
  );
  $("detail").textContent =
    state.phase === "error"
      ? state.message
      : s
        ? "Top bars: hollow is the network's first instinct for each column, filled is the share of the search's iterations."
        : "Tap a column to drop a stone.";
  document.body.dataset.phase = state.phase;
}

async function humanMove(c) {
  if (state.phase !== "human") return;
  state.moves += c;
  await advance();
}

/** Plays c4a0's moves until it is the human's turn or the game ends. */
async function advance() {
  while (true) {
    if (outcome(state.moves) !== "playing") {
      state.phase = "over";
      return render();
    }
    if (state.moves.length % 2 === state.human) {
      state.phase = "human";
      return render();
    }
    state.phase = "thinking";
    render();
    const move = await think(state.moves, (s) => { state.search = s; render(); });
    const { qPenalty, qNoPenalty, proven } = state.search;
    state.history.push({ ply: state.moves.length, qPenalty, qNoPenalty, proven });
    state.moves += move;
  }
}

/** One search from a fresh root; reports snapshots and resolves to the chosen column. */
async function think(moves, report) {
  const search = new Search(moves, ITERATIONS, LEAF_DEPTH);
  const legal = legal_moves(moves);
  let prior = null;
  const snapshot = () => {
    const [qPenalty, qNoPenalty] = search.values();
    return {
      ply: moves.length,
      prior,
      policy: search.policy(),
      visits: search.visits(),
      qPenalty,
      qNoPenalty,
      proven: search.proven(),
    };
  };
  try {
    for (let leaf; (leaf = search.next_leaf()); ) {
      const r = await net.evaluate(leaf);
      if (!prior) {
        // The first leaf is the root: its masked policy is the network's instinct.
        const p = r.logp.map((lp, c) => (legal.includes(c) ? Math.exp(lp) : 0));
        const z = p.reduce((a, b) => a + b, 0);
        prior = p.map((x) => x / z);
      }
      search.receive(Float32Array.from(r.logp), r.qPenalty, r.qNoPenalty);
      if (search.visits() % RENDER_EVERY === 0) report(snapshot());
    }
    report(snapshot());
    return search.best_move();
  } finally {
    search.free();
  }
}

function newGame(human) {
  if (state.phase === "thinking" || state.phase === "loading" || state.phase === "error") return;
  Object.assign(state, { moves: "", human, search: null, history: [] });
  advance();
}

/** `?test`: the self-check's gaps and one full game, posted to the test runner (scripts/web_test.py). */
async function selfTest(check) {
  const times = [];
  let moves = "";
  while (outcome(moves) === "playing") {
    const t = performance.now();
    moves += await think(moves, () => {});
    times.push(performance.now() - t);
    fetch("/progress", { method: "POST", body: JSON.stringify({ moves, ms: times.at(-1) }) });
  }
  return { check, moves, outcome: outcome(moves), ms_per_move: times, adapter: net.adapter };
}

async function main() {
  await init();
  render();
  $("budget").textContent = ` / ${ITERATIONS}`;
  $("first").onclick = () => newGame(0);
  $("second").onclick = () => newGame(1);
  try {
    net = await loadNet("model");
  } catch (e) {
    state.phase = "error";
    state.message =
      e instanceof NoWebGPU
        ? `${e.message} c4a0 runs its network on your GPU through WebGPU: try a current Chrome, Edge or Safari.`
        : String(e);
    return render();
  }
  $("params").textContent = net.manifest.parameters.toLocaleString("en-US");
  const cases = await fetch("model/cases.json").then((r) => r.json());
  const gap = await selfCheck(net, cases);
  if (!(gap.policy < TOLERANCE && gap.value < TOLERANCE)) {
    state.phase = "error";
    state.message = `This GPU computes the network differently from PyTorch (gaps ${gap.policy.toExponential(1)}, ${gap.value.toExponential(1)}), so this would not be c4a0 playing.`;
    return render();
  }
  if (params.has("test")) {
    const result = await selfTest(gap).catch((e) => ({ error: String(e) }));
    await fetch("/result", { method: "POST", body: JSON.stringify(result) });
    return;
  }
  state.phase = "human";
  await advance();
}

main();
