// Pure-JS forward pass for the chunppo PPO policy.
//
// The network is small enough (3.40 M params) that a plain JS implementation is
// fast: about 10 M multiply-adds per decision, and the policy only decides once
// every 4 frames (15 Hz).  That removes any need for ONNX Runtime or a CDN.
//
import { download } from "./assets.js?v=27";
import { t } from "./i18n.js?v=27";

// Architecture (mirrors chunppo/policy.py + SB3 NatureCNN):
//   image 84x84x3 --conv8/4--> 32@20 --conv4/2--> 64@9 --conv3/1--> 64@7
//                 --linear--> 1024 --ReLU--
//   hist 648 --linear--> 128 --ReLU--
//   concat 1152 --> action_net(27) / value_net(1)

function conv2d(dense, x, w, b, inC, inH, inW, outC, k, stride) {
  const outH = Math.floor((inH - k) / stride) + 1;
  const outW = Math.floor((inW - k) / stride) + 1;
  const cells = outH * outW;
  const width = inC * k * k;
  const patches = new Float32Array(cells * width);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      let to = (oy * outW + ox) * width;
      const base = oy * stride * inW + ox * stride;
      for (let ic = 0; ic < inC; ic++) {
        let from = ic * inH * inW + base;
        for (let ky = 0; ky < k; ky++, from += inW) {
          for (let kx = 0; kx < k; kx++) patches[to++] = x[from + kx];
        }
      }
    }
  }
  return { data: dense(patches, w, b, width, outC, cells, true), h: outH, w: outW, c: outC };
}

function denseJs(input, w, b, width, outC, cells, relu) {
  const out = new Float32Array(outC * cells);
  for (let oc = 0; oc < outC; oc++) {
    const channel = oc * width;
    for (let cell = 0; cell < cells; cell++) {
      const start = cell * width;
      let sum0 = b[oc], sum1 = 0, sum2 = 0, sum3 = 0;
      let i = 0;
      for (; i + 3 < width; i += 4) {
        sum0 += input[start + i] * w[channel + i];
        sum1 += input[start + i + 1] * w[channel + i + 1];
        sum2 += input[start + i + 2] * w[channel + i + 2];
        sum3 += input[start + i + 3] * w[channel + i + 3];
      }
      let sum = sum0 + sum1 + sum2 + sum3;
      for (; i < width; i++) sum += input[start + i] * w[channel + i];
      out[oc * cells + cell] = relu && sum < 0 ? 0 : sum;
    }
  }
  return out;
}

function wasmDense(blob, { memory, __heap_base, dense }) {
  const weights = Number(__heap_base.value);
  const scratch = weights + blob.byteLength;
  const reserve = scratch + 512 * 1024;
  memory.grow(Math.ceil((reserve - memory.buffer.byteLength) / 65536));
  const heap = new Float32Array(memory.buffer);
  heap.set(blob, weights / 4);
  const ptr = tensor => weights + tensor.byteOffset - blob.byteOffset;
  return (input, w, b, width, channels, cells, relu) => {
    const output = scratch + input.byteLength;
    heap.set(input, scratch / 4);
    dense(scratch, ptr(w), ptr(b), output, width, channels, cells, Number(relu));
    return heap.slice(output / 4, output / 4 + channels * cells);
  };
}

export class Policy {
  constructor(blob, meta, acceleration = null) {
    this.meta = meta;
    this.blob = blob; // single Float32Array of all tensors
    this.t = new Map();
    for (const t of meta.tensors) {
      this.t.set(t.name, { data: blob.subarray(t.offset / 4, t.offset / 4 + t.count), shape: t.shape });
    }
    this.nActions = meta.nActions;
    this.imageSize = meta.imageSize;
    this.histDim = meta.histDim;
    this.dense = acceleration ? wasmDense(blob, acceleration) : denseJs;
  }

  static async load(basePath = "models/", { signal, onProgress } = {}) {
    const acceleration = download(`${basePath}policy_simd.wasm?v=15`, { signal, idleMs: 5_000 })
      .then(async bytes => (await WebAssembly.instantiate(bytes)).instance.exports)
      .catch(() => null);
    const [index, weights] = await Promise.all([
      download(`${basePath}chunppo_p27.json`, { signal }),
      download(`${basePath}chunppo_p27.bin`, { signal, onProgress }),
    ]);
    const meta = JSON.parse(new TextDecoder().decode(index));
    if (weights.byteLength !== meta.params * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error(t("policy.incomplete"));
    }
    const blob = new Float32Array(weights.buffer);
    const policy = new Policy(blob, meta);
    acceleration.then(wasm => {
      if (wasm && !signal?.aborted) policy.dense = wasmDense(blob, wasm);
    }).catch(() => {});
    return policy;
  }

  get(name) {
    const t = this.t.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return t;
  }

  /**
   * @param {Float32Array} image 3*84*84 in [0,1], channel-major (R plane, G plane, B plane)
   * @param {Float32Array} hist  648 floats, 24 steps x 27 one-hot
   * @returns {{logits: Float32Array, value: number}}
   */
  forward(image, hist) {
    const S = this.imageSize;

    const dense = this.dense;
    const a = conv2d(dense, image, this.get("features_extractor.cnn.cnn.0.weight").data,
      this.get("features_extractor.cnn.cnn.0.bias").data, 3, S, S, 32, 8, 4);
    const b1 = conv2d(dense, a.data, this.get("features_extractor.cnn.cnn.2.weight").data,
      this.get("features_extractor.cnn.cnn.2.bias").data, a.c, a.h, a.w, 64, 4, 2);
    const c1 = conv2d(dense, b1.data, this.get("features_extractor.cnn.cnn.4.weight").data,
      this.get("features_extractor.cnn.cnn.4.bias").data, b1.c, b1.h, b1.w, 64, 3, 1);

    const feat = dense(c1.data,
      this.get("features_extractor.cnn.linear.0.weight").data,
      this.get("features_extractor.cnn.linear.0.bias").data,
      c1.c * c1.h * c1.w, 1024, 1, true);

    const histFeat = dense(hist,
      this.get("features_extractor.hist.0.weight").data,
      this.get("features_extractor.hist.0.bias").data,
      this.histDim, 128, 1, true);

    const joined = new Float32Array(1152);
    joined.set(feat, 0);
    joined.set(histFeat, 1024);

    const logits = dense(joined, this.get("action_net.weight").data,
      this.get("action_net.bias").data, 1152, this.nActions, 1, false);
    const value = dense(joined, this.get("value_net.weight").data,
      this.get("value_net.bias").data, 1152, 1, 1, false)[0];

    return { logits, value };
  }

  argmax(logits) {
    let best = 0;
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
}
