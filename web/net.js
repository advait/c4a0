// The c4a0 network on WebGPU: the layers scripts/export_web.py writes, one compute dispatch each.
//
// Activations are (batch, channels, 6, 7) float32, row-major, as PyTorch lays them out. A linear
// layer is a 1x1 convolution over a 1x1 "board" whose channels are the flattened input, which is
// exactly how nn.Flatten orders it.

const ROWS = 6;
const COLS = 7;
const CELLS = ROWS * COLS;
const ACT = { none: 0, relu: 1, tanh: 2 };

const SHADER = /* wgsl */ `
struct Layer { batch: u32, cin: u32, cout: u32, h: u32, w: u32, k: u32, woff: u32, boff: u32,
               act: u32, residual: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> skip: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> L: Layer;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let hw = L.h * L.w;
  let i = id.x;
  if (i >= L.batch * L.cout * hw) { return; }
  let b = i / (L.cout * hw);
  let co = (i / hw) % L.cout;
  let row = i32((i % hw) / L.w);
  let col = i32(i % L.w);
  let pad = i32(L.k / 2u);
  var acc = weights[L.boff + co];
  for (var ci = 0u; ci < L.cin; ci++) {
    let xbase = (b * L.cin + ci) * hw;
    let wbase = L.woff + (co * L.cin + ci) * L.k * L.k;
    for (var ky = 0u; ky < L.k; ky++) {
      let r = row + i32(ky) - pad;
      if (r < 0 || r >= i32(L.h)) { continue; }
      for (var kx = 0u; kx < L.k; kx++) {
        let c = col + i32(kx) - pad;
        if (c < 0 || c >= i32(L.w)) { continue; }
        acc += x[xbase + u32(r) * L.w + u32(c)] * weights[wbase + ky * L.k + kx];
      }
    }
  }
  if (L.act == 1u) { acc = max(acc, 0.0); }
  if (L.act == 2u) { acc = tanh(acc); }
  if (L.residual == 1u) { acc += skip[i]; }
  y[i] = acc;
}`;

/** Thrown when this browser or GPU cannot run the network. */
export class NoWebGPU extends Error {}

/**
 * Loads the exported model onto the GPU. Resolves to a Net whose `forward(input, batch)` takes
 * `batch` encoded positions (2 x 6 x 7 floats each) and returns their policy log-probabilities
 * (7 each) and both values.
 */
export async function loadNet(base) {
  if (!navigator.gpu) throw new NoWebGPU("This browser has no WebGPU.");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new NoWebGPU("WebGPU found no usable GPU.");
  const device = await adapter.requestDevice();
  const [manifest, weights] = await Promise.all([
    fetch(`${base}/manifest.json`).then((r) => r.json()),
    fetch(`${base}/weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  const net = new Net(device, manifest, new Float32Array(weights));
  net.adapter = `${adapter.info?.vendor ?? "?"}/${adapter.info?.architecture ?? "?"}`;
  return net;
}

class Net {
  constructor(device, manifest, weights) {
    this.device = device;
    this.manifest = manifest;
    this.weights = device.createBuffer({
      size: weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.weights, 0, weights);
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: SHADER }), entryPoint: "main" },
    });
    this.plans = new Map(); // batch size -> prepared forward pass
    this.busy = Promise.resolve();
  }

  /** Runs the network; calls are serialized because each batch size reuses its buffers. */
  forward(input, batch) {
    const run = this.busy.then(() => this.plan(batch).run(input));
    this.busy = run.catch(() => {});
    return run;
  }

  plan(batch) {
    if (!this.plans.has(batch)) this.plans.set(batch, this.prepare(batch));
    return this.plans.get(batch);
  }

  /** Buffers, uniforms and bind groups for one batch size, built once. */
  prepare(batch) {
    const { device, pipeline, manifest } = this;
    const S = GPUBufferUsage.STORAGE;
    const buffer = (floats, usage = S) => device.createBuffer({ size: floats * 4, usage });
    const width = Math.max(...manifest.trunk.map((l) => l.cout));
    const input = buffer(batch * 2 * CELLS, S | GPUBufferUsage.COPY_DST);
    const trunk = [0, 1, 2].map(() => buffer(batch * width * CELLS));
    const head = (layers) =>
      layers.map((l) => buffer(batch * l.cout * (l.kind === "conv" ? CELLS : 1), S | GPUBufferUsage.COPY_SRC));
    const policy = head(manifest.policy);
    const value = head(manifest.value);
    const readback = buffer(batch * 9, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);

    const steps = [];
    const step = (layer, x, y, skip) => {
      const [h, w] = layer.kind === "conv" ? [ROWS, COLS] : [1, 1];
      const uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const fields = [batch, layer.cin, layer.cout, h, w, layer.k, layer.w, layer.b, ACT[layer.act], layer.residual ? 1 : 0];
      device.queue.writeBuffer(uniform, 0, new Uint32Array(fields));
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [x, this.weights, skip ?? x, y, uniform].map((b, binding) => ({ binding, resource: { buffer: b } })),
      });
      steps.push({ bindGroup, workgroups: Math.ceil((batch * layer.cout * h * w) / 64) });
    };

    // Trunk: the stem, then per block conv1 into a scratch buffer and conv2 plus the block input
    // into the third buffer, which becomes the next block's input.
    let [current, scratch, next] = trunk;
    const [stem, ...blocks] = manifest.trunk;
    step(stem, input, current);
    for (const layer of blocks) {
      if (layer.residual) {
        step(layer, scratch, next, current);
        [current, next] = [next, current];
      } else {
        step(layer, current, scratch);
      }
    }
    let x = current;
    manifest.policy.forEach((l, i) => { step(l, x, policy[i]); x = policy[i]; });
    x = current;
    manifest.value.forEach((l, i) => { step(l, x, value[i]); x = value[i]; });
    const logits = policy.at(-1);
    const values = value.at(-1);

    return {
      run: async (data) => {
        device.queue.writeBuffer(input, 0, data);
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        for (const s of steps) {
          pass.setBindGroup(0, s.bindGroup);
          pass.dispatchWorkgroups(s.workgroups);
        }
        pass.end();
        encoder.copyBufferToBuffer(logits, 0, readback, 0, batch * 7 * 4);
        encoder.copyBufferToBuffer(values, 0, readback, batch * 7 * 4, batch * 2 * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return Array.from({ length: batch }, (_, i) => ({
          logp: logSoftmax(out.subarray(i * 7, i * 7 + 7)),
          qPenalty: out[batch * 7 + i * 2],
          qNoPenalty: out[batch * 7 + i * 2 + 1],
        }));
      },
    };
  }

  /**
   * The search's evaluator: the position and its mirror image in one batch, averaged as
   * ConnectFourNet.infer does with mirror_inference (probabilities averaged, columns flipped).
   */
  async evaluate(planes) {
    const both = new Float32Array(4 * CELLS);
    both.set(planes);
    both.set(mirror(planes), 2 * CELLS);
    const [a, m] = await this.forward(both, 2);
    const logp = a.logp.map((lp, c) => logAddExp(lp, m.logp[COLS - 1 - c]) - Math.LN2);
    return {
      logp,
      qPenalty: (a.qPenalty + m.qPenalty) / 2,
      qNoPenalty: (a.qNoPenalty + m.qNoPenalty) / 2,
    };
  }
}

function mirror(planes) {
  const out = new Float32Array(planes.length);
  for (let i = 0; i < planes.length; i++) {
    const c = i % COLS;
    out[i - c + (COLS - 1 - c)] = planes[i];
  }
  return out;
}

function logSoftmax(logits) {
  const m = Math.max(...logits);
  const lse = m + Math.log(logits.reduce((s, l) => s + Math.exp(l - m), 0));
  return Array.from(logits, (l) => l - lse);
}

function logAddExp(a, b) {
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

/** Largest gaps between this GPU's outputs and PyTorch's on the exported reference positions. */
export async function selfCheck(net, cases) {
  const n = cases.moves.length;
  const out = await net.forward(new Float32Array(cases.input), n);
  let policy = 0;
  let value = 0;
  out.forEach((o, i) => {
    o.logp.forEach((lp, c) => { policy = Math.max(policy, Math.abs(lp - cases.policy[i * 7 + c])); });
    value = Math.max(value, Math.abs(o.qPenalty - cases.q_penalty[i]), Math.abs(o.qNoPenalty - cases.q_no_penalty[i]));
  });
  return { policy, value };
}
