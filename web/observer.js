// Port of chunppo/env.py Observer: the 84x84 motion-foreground observation the
// policy was trained on.
//
// Python reference (kept in sync by hand -- parity is checked by
// scripts/web_test_observer_parity.py):
//
//   crop centre to min(256,w) x min(200,h)
//   gray = RGB -> BT.601 luma
//   motion = max(|g[t-4]-g[t-2]|, |g[t-2]-g[t]|) blurred 5x5, Otsu threshold,
//            dilated 3x3
//   frame  = motion_mask ? frame : frame * 0.35      (dim the static parts)
//   resize 84x84 bilinear
//   obs    = [stack[0].R, stack[2].G, stack[4].B]  + 24x27 one-hot action history

export const IMG = 84;
export const HIST_LEN = 24;
export const N_ACTIONS = 27;

// cv2 resize: coefficients are 11-bit fixed point, two passes give 22 bits.
const RESIZE_BITS = 11;
const RESIZE_SCALE = 1 << RESIZE_BITS;
const RESIZE_ROUND = 1 << (2 * RESIZE_BITS - 1);

function reflect101(i, size) {
  return i < 0 ? -i : i >= size ? 2 * (size - 1) - i : i;
}

/** Decode one emulator frame into packed RGB888. */
export function decodeFrame(data, width, height, pitch, format) {
  const out = new Uint8Array(width * height * 3);
  const heap = data instanceof Uint8Array ? data : new Uint8Array(data);
  if (format === 2) {
    // RGB565
    for (let y = 0; y < height; y++) {
      const row = y * pitch;
      for (let x = 0; x < width; x++) {
        const o = row + x * 2;
        const v = heap[o] | (heap[o + 1] << 8);
        const r = (v >> 11) & 0x1f;
        const g = (v >> 5) & 0x3f;
        const b = v & 0x1f;
        const i = (y * width + x) * 3;
        out[i] = (r << 3) | (r >> 2);
        out[i + 1] = (g << 2) | (g >> 4);
        out[i + 2] = (b << 3) | (b >> 2);
      }
    }
  } else if (format === 1) {
    // XRGB8888
    for (let y = 0; y < height; y++) {
      const row = y * pitch;
      for (let x = 0; x < width; x++) {
        const o = row + x * 4;
        const i = (y * width + x) * 3;
        out[i] = heap[o + 2];
        out[i + 1] = heap[o + 1];
        out[i + 2] = heap[o];
      }
    }
  } else {
    // 0RGB1555
    for (let y = 0; y < height; y++) {
      const row = y * pitch;
      for (let x = 0; x < width; x++) {
        const o = row + x * 2;
        const v = heap[o] | (heap[o + 1] << 8);
        const r = (v >> 10) & 0x1f;
        const g = (v >> 5) & 0x1f;
        const b = v & 0x1f;
        const i = (y * width + x) * 3;
        out[i] = (r << 3) | (r >> 2);
        out[i + 1] = (g << 3) | (g >> 2);
        out[i + 2] = (b << 3) | (b >> 2);
      }
    }
  }
  return out;
}

/** Match Gym Retro's scenario crop [0, 16, 256, 200] and RGB565 expansion. */
export function decodePolicyFrame(data, width, height, pitch) {
  const w = Math.min(width, 256);
  const top = Math.min(16, height - 1);
  const h = Math.min(200, height - top);
  const rgb = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    let from = (y + top) * pitch;
    let to = y * w * 3;
    for (let x = 0; x < w; x++, from += 2) {
      const pixel = data[from] | (data[from + 1] << 8);
      rgb[to++] = (pixel >> 8) & 0xf8;
      rgb[to++] = (pixel >> 3) & 0xfc;
      rgb[to++] = (pixel << 3) & 0xf8;
    }
  }
  return { rgb, w, h };
}

export class Observer {
  constructor() {
    this.cw = 0;
    this.ch = 0;
    this.grayBuf = null;
    this.motion = [];          // deque of Uint8Array grays, max 5
    this.stack = [];           // deque of Uint8Array 84x84x3, max 5
    this.hist = new Array(HIST_LEN).fill(-1);
    this.mask = null;
    this.scratch = null;
  }

  reset(rgb, width, height) {
    this.cw = Math.min(256, width);
    this.ch = Math.min(200, height);
    if (!this.grayBuf) this.grayBuf = new Uint8Array(this.cw * this.ch);
    this.motion = [];
    this.stack = [];
    this.hist = new Array(HIST_LEN).fill(-1);
    this.mask = new Uint8Array(this.cw * this.ch);
    this.scratch = new Float32Array(HIST_LEN * N_ACTIONS);
    const first = this._process(rgb, width, height);
    this.stack = [first, first, first, first, first];
  }

  _gray(rgb, width, height) {
    const { cw, ch, grayBuf } = this;
    const left = (width - cw) >> 1;
    const top = (height - ch) >> 1;
    for (let y = 0; y < ch; y++) {
      let s = ((top + y) * width + left) * 3;
      let d = y * cw;
      for (let x = 0; x < cw; x++, s += 3, d++) {
        // cv2.COLOR_RGB2GRAY rounds; truncating here shifted the whole motion
        // histogram by a level.
        grayBuf[d] = Math.round(rgb[s] * 0.299 + rgb[s + 1] * 0.587 + rgb[s + 2] * 0.114);
      }
    }
    return grayBuf;
  }

  _process(rgb, width, height) {
    const { cw, ch } = this;
    const gray = this._gray(rgb, width, height);

    const g = new Uint8Array(gray);          // copy: grays are retained in the deque
    this.motion.push(g);
    if (this.motion.length > 5) this.motion.shift();

    let maskOn = false;
    if (this.motion.length >= 5) {
      const m = this.motion;
      const a = new Uint8Array(cw * ch);
      for (let i = 0; i < a.length; i++) {
        const d1 = Math.abs(m[0][i] - m[2][i]);
        const d2 = Math.abs(m[2][i] - m[4][i]);
        a[i] = d1 > d2 ? d1 : d2;
      }
      // Separable [1,4,6,4,1] blur with cv2's reflect-101 borders. Keep the
      // horizontal sum unrounded so the two passes equal the full 5x5 kernel.
      const horizontal = new Uint16Array(cw * ch);
      const blur = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        const row = y * cw;
        for (let x = 0; x < cw; x++) {
          horizontal[row + x] = a[row + reflect101(x - 2, cw)] + 4 * a[row + reflect101(x - 1, cw)]
            + 6 * a[row + x] + 4 * a[row + reflect101(x + 1, cw)] + a[row + reflect101(x + 2, cw)];
        }
      }
      for (let y = 0; y < ch; y++) {
        const row = y * cw;
        const above2 = reflect101(y - 2, ch) * cw, above = reflect101(y - 1, ch) * cw;
        const below2 = reflect101(y + 2, ch) * cw, below = reflect101(y + 1, ch) * cw;
        for (let x = 0; x < cw; x++) {
          blur[row + x] = (horizontal[above2 + x] + 4 * horizontal[above + x] + 6 * horizontal[row + x]
            + 4 * horizontal[below + x] + horizontal[below2 + x] + 128) >> 8;
        }
      }
      // Otsu
      const hist = new Float64Array(256);
      let total = 0;
      for (let i = 0; i < blur.length; i++) {
        hist[blur[i]]++;
        total++;
      }
      let sumAll = 0;
      for (let i = 0; i < 256; i++) sumAll += i * hist[i];
      let sumB = 0, wB = 0, best = -1, thr = 0;
      for (let t = 0; t < 256; t++) {
        wB += hist[t];
        if (wB === 0) continue;
        const wF = total - wB;
        if (wF === 0) break;
        sumB += t * hist[t];
        const mB = sumB / wB;
        const mF = (sumAll - sumB) / wF;
        const between = wB * wF * (mB - mF) * (mB - mF);
        if (between > best) { best = between; thr = t; }
      }
      const bin = this.mask;
      for (let i = 0; i < blur.length; i++) bin[i] = blur[i] > thr ? 255 : 0;
      // dilate 3x3
      const dil = new Uint8Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          let v = 0;
          for (let ky = -1; ky <= 1 && !v; ky++) {
            const yy = y + ky;
            if (yy < 0 || yy >= ch) continue;
            for (let kx = -1; kx <= 1; kx++) {
              const xx = x + kx;
              if (xx < 0 || xx >= cw) continue;
              if (bin[yy * cw + xx]) { v = 255; break; }
            }
          }
          dil[y * cw + x] = v;
        }
      }
      this.mask = dil;
      maskOn = true;
    }

    // crop the RGB frame, dim where static, then resize to 84x84
    const left = (width - cw) >> 1;
    const top = (height - ch) >> 1;
    const crop = new Uint8Array(cw * ch * 3);
    for (let y = 0; y < ch; y++) {
      let s = ((top + y) * width + left) * 3;
      let d = y * cw * 3;
      for (let x = 0; x < cw; x++, s += 3, d += 3) {
        // Python only dims static pixels inside the motion-mask branch, i.e. once
        // five grays exist.  Before that the frame is passed through untouched.
        const f = maskOn && this.mask[y * cw + x] === 0 ? 0.35 : 1;
        crop[d] = rgb[s] * f;
        crop[d + 1] = rgb[s + 1] * f;
        crop[d + 2] = rgb[s + 2] * f;
      }
    }
    return resize84(crop, cw, ch);
  }

  push(rgb, width, height, action = -1) {
    this.stack.push(this._process(rgb, width, height));
    if (this.stack.length > 5) this.stack.shift();
    this.hist.push(action);
    if (this.hist.length > HIST_LEN) this.hist.shift();
  }

  obs() {
    const s = this.stack;
    const image = new Float32Array(3 * IMG * IMG);
    for (let i = 0; i < IMG * IMG; i++) {
      image[i] = s[0][i * 3] / 255;
      image[IMG * IMG + i] = s[2][i * 3 + 1] / 255;
      image[2 * IMG * IMG + i] = s[4][i * 3 + 2] / 255;
    }
    const hist = this.scratch;
    hist.fill(0);
    for (let i = 0; i < HIST_LEN; i++) {
      const a = this.hist[i];
      if (a >= 0) hist[i * N_ACTIONS + a] = 1;
    }
    return { image, hist };
  }
}

/** Bilinear resize of an RGB888 crop to 84x84 with cv2's 11-bit coefficients. */
function resizeCoeffs(nIn) {
  const scale = nIn / IMG;
  const out = new Int32Array(IMG * 3);
  for (let i = 0; i < IMG; i++) {
    // cv2 computes the source coordinate in float32.
    const f = Math.fround((i + 0.5) * scale - 0.5);
    let i0 = Math.floor(f);
    let frac = f - i0;
    if (i0 < 0) { i0 = 0; frac = 0; }
    if (i0 >= nIn - 1) { i0 = nIn - 1; frac = 0; }
    out[i * 3] = i0;
    out[i * 3 + 1] = ((1 - frac) * RESIZE_SCALE) | 0;
    out[i * 3 + 2] = (frac * RESIZE_SCALE) | 0;
  }
  return out;
}

function resize84(src, sw, sh) {
  const xs = resizeCoeffs(sw);
  const ys = resizeCoeffs(sh);
  const out = new Uint8Array(IMG * IMG * 3);
  const h0 = new Int32Array(IMG * 3);
  const h1 = new Int32Array(IMG * 3);
  const filterRow = (row, dst) => {
    const base = row * sw * 3;
    for (let x = 0; x < IMG; x++) {
      const i0 = xs[x * 3];
      const c0 = xs[x * 3 + 1];
      const c1 = xs[x * 3 + 2];
      const i1 = Math.min(sw - 1, i0 + 1);
      const a = base + i0 * 3;
      const b = base + i1 * 3;
      const o = x * 3;
      for (let c = 0; c < 3; c++) dst[o + c] = src[a + c] * c0 + src[b + c] * c1;
    }
  };
  for (let y = 0; y < IMG; y++) {
    const y0 = ys[y * 3];
    const cy0 = ys[y * 3 + 1];
    const cy1 = ys[y * 3 + 2];
    const y1 = Math.min(sh - 1, y0 + 1);
    filterRow(y0, h0);
    filterRow(y1, h1);
    const o = y * IMG * 3;
    for (let i = 0; i < IMG * 3; i++) {
      const v = (h0[i] * cy0 + h1[i] * cy1 + RESIZE_ROUND) >> (2 * RESIZE_BITS);
      out[o + i] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return out;
}
