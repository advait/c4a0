// The c4a0 browser player: the Rust search (WebAssembly) asks for one position at a time and the
// network answers on the GPU (net.js). The page renders from `state` after every change.

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
const SAVED = "c4a0-game";

const state = {
  phase: "loading", // loading | human | thinking | over | error
  moves: "", // 0-indexed columns, first player first
  human: 0, // the human's player index: 0 moves first
  search: null, // c4a0's latest search: see `think`
  history: [], // { ply, qPenalty, qNoPenalty, proven } at each c4a0 move
  epoch: 0, // bumped to abandon an in-flight search
};

let net;
const $ = (id) => document.getElementById(id);

class Cancelled extends Error {}

// ---------------------------------------------------------------------------------------------
// Game rules the page needs beyond the engine's: the grid and the winning line.

function grid(moves) {
  const cells = Array.from({ length: ROWS }, () => Array(COLS).fill(-1)); // row 0 = bottom
  [...moves].forEach((ch, ply) => {
    const c = Number(ch);
    cells[cells.findIndex((row) => row[c] === -1)][c] = ply % 2;
  });
  return cells;
}

function winningLine(cells, moves) {
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

/** The latest position before `moves` where it is the human's turn, or null. */
function undoTarget(moves, human) {
  for (let n = moves.length - 1; n >= 0; n--) if (n % 2 === human) return moves.slice(0, n);
  return null;
}

// ---------------------------------------------------------------------------------------------
// c4a0's values in words. Both are from c4a0's side. A decisive result ending on ply n scores
// q_penalty = ±(1 - 0.01 n), so a proven result names its last move exactly and an unproven one
// estimates it from q_penalty / q_no_penalty.

function verdict(s) {
  const penalty = Search.ply_penalty();
  const { qPenalty: qp, qNoPenalty: qn } = s;
  if (s.proven) {
    if (qn === 0) return "Draw with best play";
    const end = Math.round((1 - Math.abs(qp)) / penalty);
    return qn > 0 ? `c4a0 wins by move ${end}` : `You can win by move ${end}`;
  }
  if (Math.abs(qn) < 0.3) return "Even";
  const end = Math.min(42, Math.max(s.ply + 1, Math.round((1 - qp / qn) / penalty)));
  return `${qn > 0 ? "c4a0" : "You"} ahead · ends ~move ${end}`;
}

const signed = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

// ---------------------------------------------------------------------------------------------
// Rendering. The board and bars are built once and updated in place, so stones can animate.

const view = { cols: [], cells: [], bars: [], shown: "" };

function build() {
  for (let c = 0; c < COLS; c++) {
    const col = document.createElement("button");
    col.className = "col";
    col.setAttribute("aria-label", `Column ${c + 1}`);
    col.onclick = () => humanMove(c);
    const cells = [];
    for (let r = ROWS - 1; r >= 0; r--) {
      const cell = document.createElement("span");
      cell.className = "cell";
      col.append(cell);
      cells[r] = cell;
    }
    const ghost = document.createElement("span");
    ghost.className = "ghost";
    col.append(ghost);
    $("board").append(col);
    view.cols.push(col);
    view.cells.push(cells);

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = '<span class="visits"></span><span class="prior"></span><span class="pct"></span>';
    $("bars").append(bar);
    view.bars.push(bar);
  }
}

function owner(player) {
  return player === state.human ? "you" : "them";
}

function renderBoard() {
  const cells = grid(state.moves);
  const over = state.phase === "over";
  const won = over && outcome(state.moves) === "win" ? winningLine(cells, state.moves) : [];
  const legal = state.phase === "human" ? new Set(legal_moves(state.moves)) : new Set();
  const lastCol = state.moves ? Number(state.moves.at(-1)) : -1;
  // Drop only a stone that was just added, not stones restored by an undo.
  const dropped = state.moves.length === view.shown.length + 1 && state.moves.startsWith(view.shown);
  for (let c = 0; c < COLS; c++) {
    const free = cells.findIndex((row) => row[c] === -1);
    view.cols[c].disabled = !legal.has(c);
    view.cols[c].style.setProperty("--free", String(free === -1 ? ROWS : ROWS - 1 - free));
    for (let r = 0; r < ROWS; r++) {
      const cell = view.cells[c][r];
      const who = cells[r][c];
      const top = c === lastCol && who !== -1 && (r === ROWS - 1 || cells[r + 1][c] === -1);
      cell.className = "cell";
      if (who !== -1) cell.classList.add("stone", owner(who));
      if (top) cell.classList.add("last");
      if (won.some(([wr, wc]) => wr === r && wc === c)) cell.classList.add("won");
      if (top && dropped) {
        cell.style.setProperty("--fall", String(ROWS - r));
        void cell.offsetWidth; // restart the animation
        cell.classList.add("drop");
      }
    }
  }
  $("board").classList.toggle("finished", won.length > 0);
  $("board").dataset.next = owner(state.moves.length % 2);
  view.shown = state.moves;
}

function renderSearch() {
  const s = state.search;
  view.bars.forEach((bar, c) => {
    const [visits, prior, pct] = bar.children;
    visits.style.height = `${(s?.policy[c] ?? 0) * 100}%`;
    prior.style.height = `${(s?.prior?.[c] ?? 0) * 100}%`;
    pct.textContent = s && s.policy[c] >= 0.01 ? `${Math.round(s.policy[c] * 100)}` : "";
  });
  for (const [id, v] of [["result", s?.qNoPenalty], ["steer", s?.qPenalty]]) {
    const el = $(id);
    const fill = el.querySelector(".fill");
    const x = v ?? 0;
    fill.style.left = `${50 + Math.min(x, 0) * 50}%`;
    fill.style.width = `${Math.abs(x) * 50}%`;
    fill.dataset.side = x >= 0 ? "them" : "you";
    el.querySelector(".num").textContent = v === undefined ? "" : signed(v);
  }
  $("verdict").textContent = s ? verdict(s) : "–";
  $("history").replaceChildren(
    ...state.history.map((h) => {
      const tick = document.createElement("span");
      tick.dataset.side = h.qNoPenalty >= 0 ? "them" : "you";
      tick.style.setProperty("--v", String(Math.max(0.06, Math.abs(h.qNoPenalty))));
      tick.title = `Move ${h.ply + 1}: ${signed(h.qNoPenalty)}`;
      return tick;
    }),
  );
}

function render() {
  const result = state.phase === "over" ? outcome(state.moves) : "";
  const youWon = result === "win" && (state.moves.length - 1) % 2 === state.human;
  $("status").textContent = {
    loading: "Loading network",
    human: "Your move",
    thinking: "c4a0 is thinking",
    over: result === "draw" ? "Draw" : youWon ? "You win" : "c4a0 wins",
    error: "",
  }[state.phase];
  $("again").hidden = state.phase !== "over";
  const visits = state.phase === "thinking" && state.search ? state.search.visits : 0;
  $("progress").style.width = `${(visits / ITERATIONS) * 100}%`;
  $("move").textContent = state.moves.length ? `Move ${state.moves.length}` : "";
  const live = state.phase !== "loading" && state.phase !== "error";
  const canUndo = live && undoTarget(state.moves, state.human) !== null;
  $("undo").disabled = !canUndo;
  $("menu-undo").disabled = !canUndo;
  document.body.dataset.phase = state.phase;
  if (live) renderBoard();
  renderSearch();
}

// ---------------------------------------------------------------------------------------------
// Play.

function save() {
  const { moves, human, history } = state;
  localStorage.setItem(SAVED, JSON.stringify({ moves, human, history }));
}

function restore() {
  try {
    const saved = JSON.parse(localStorage.getItem(SAVED) ?? "null");
    if (saved && typeof saved.moves === "string" && (saved.human === 0 || saved.human === 1)) {
      outcome(saved.moves); // throws on a corrupt move string
      Object.assign(state, { moves: saved.moves, human: saved.human, history: saved.history ?? [] });
    }
  } catch {
    localStorage.removeItem(SAVED);
  }
}

function humanMove(c) {
  if (state.phase !== "human" || !legal_moves(state.moves).includes(c)) return;
  navigator.vibrate?.(8);
  state.moves += c;
  advance();
}

/** Plays c4a0's moves until it is the human's turn or the game ends. */
async function advance() {
  const epoch = state.epoch;
  try {
    while (true) {
      save();
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
      const move = await think(state.moves, epoch, (s) => {
        state.search = s;
        render();
      });
      const { qPenalty, qNoPenalty, proven } = state.search;
      state.history.push({ ply: state.moves.length, qPenalty, qNoPenalty, proven });
      state.moves += move;
    }
  } catch (e) {
    if (!(e instanceof Cancelled)) throw e;
  }
}

/** One search from a fresh root; reports snapshots and resolves to the chosen column. */
async function think(moves, epoch, report) {
  const search = new Search(moves, ITERATIONS, LEAF_DEPTH);
  const legal = legal_moves(moves);
  let prior = null;
  const snapshot = () => {
    const [qPenalty, qNoPenalty] = search.values();
    const policy = search.policy();
    return { ply: moves.length, prior, policy, visits: search.visits(), qPenalty, qNoPenalty, proven: search.proven() };
  };
  try {
    for (let leaf; (leaf = search.next_leaf()); ) {
      const r = await net.evaluate(leaf);
      if (epoch !== state.epoch) throw new Cancelled();
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

/** Abandons any search and continues from `moves`. */
function reset(moves, human) {
  state.epoch++;
  state.moves = moves;
  state.human = human;
  state.history = state.history.filter((h) => h.ply < moves.length);
  const last = state.history.at(-1);
  state.search = last ? { ...last, policy: Array(COLS).fill(0), prior: null, visits: 0 } : null;
  advance();
}

function undo() {
  if (state.phase === "loading" || state.phase === "error") return;
  const target = undoTarget(state.moves, state.human);
  if (target !== null) reset(target, state.human);
}

function newGame(human) {
  if (state.phase === "loading" || state.phase === "error") return;
  state.history = [];
  reset("", human);
}

function wire() {
  $("undo").onclick = undo;
  $("again").onclick = () => newGame(state.human);
  const menu = $("menu");
  menu.addEventListener("click", (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (!action) return;
    menu.hidePopover();
    if (action === "new-you") newGame(0);
    if (action === "new-them") newGame(1);
    if (action === "undo") undo();
  });
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.altKey) return;
    if ((e.ctrlKey && e.key === "z") || (!e.ctrlKey && (e.key === "u" || e.key === "Backspace"))) {
      e.preventDefault();
      undo();
    } else if (!e.ctrlKey && e.key === "n") {
      newGame(state.human);
    } else if (!e.ctrlKey && e.key >= "1" && e.key <= "7") {
      humanMove(Number(e.key) - 1);
    }
  });
}

function fail(message) {
  state.phase = "error";
  $("error-message").textContent = message;
  $("error").hidden = false;
  render();
}

/** `?test`: the self-check's gaps and one full game, posted to the test runner (scripts/web_test.py). */
async function selfTest(check) {
  const times = [];
  let moves = "";
  while (outcome(moves) === "playing") {
    const t = performance.now();
    moves += await think(moves, state.epoch, () => {});
    times.push(performance.now() - t);
    fetch("/progress", { method: "POST", body: JSON.stringify({ moves, ms: times.at(-1) }) });
  }
  return { check, moves, outcome: outcome(moves), ms_per_move: times, adapter: net.adapter };
}

async function main() {
  await init();
  build();
  wire();
  render();
  try {
    net = await loadNet("model");
  } catch (e) {
    return fail(e instanceof NoWebGPU ? `${e.message} Try a current Chrome, Edge or Safari.` : String(e));
  }
  $("params").textContent = net.manifest.parameters.toLocaleString("en-US");
  const cases = await fetch("model/cases.json").then((r) => r.json());
  const gap = await selfCheck(net, cases);
  if (!(gap.policy < TOLERANCE && gap.value < TOLERANCE)) {
    const worst = Math.max(gap.policy, gap.value).toExponential(1);
    return fail(`This GPU computes the network differently from PyTorch (gap ${worst}), so it would not be c4a0 playing.`);
  }
  if (params.has("test")) {
    const result = await selfTest(gap).catch((e) => ({ error: String(e) }));
    await fetch("/result", { method: "POST", body: JSON.stringify(result) });
    return;
  }
  if (!params.has("fresh")) restore();
  state.phase = "human";
  advance();
}

main();
