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

const GAUSS5 = (() => {
  // cv2.GaussianBlur(ksize=5, sigma=0) -> sigma = 0.3*((5-1)*0.5-1)+0.8 = 1.1
  const sigma = 1.1;
  const r = 2;
  const k = [];
  let sum = 0;
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      const v = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
      k.push(v);
      sum += v;
    }
  }
  return k.map((v) => v / sum);
})();

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
        grayBuf[d] = (rgb[s] * 0.299 + rgb[s + 1] * 0.587 + rgb[s + 2] * 0.114) | 0;
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
      // 5x5 gaussian, replicate edges
      const blur = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) {
          let acc = 0;
          let ki = 0;
          for (let ky = -2; ky <= 2; ky++) {
            const yy = Math.min(ch - 1, Math.max(0, y + ky));
            for (let kx = -2; kx <= 2; kx++, ki++) {
              const xx = Math.min(cw - 1, Math.max(0, x + kx));
              acc += a[yy * cw + xx] * GAUSS5[ki];
            }
          }
          blur[y * cw + x] = acc;
        }
      }
      // Otsu
      const hist = new Float64Array(256);
      let total = 0;
      for (let i = 0; i < blur.length; i++) {
        const v = blur[i] < 0 ? 0 : blur[i] > 255 ? 255 : blur[i] | 0;
        hist[v]++;
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

/** Bilinear resize of an RGB888 crop to 84x84 (matches cv2.INTER_LINEAR). */
function resize84(src, sw, sh) {
  const out = new Uint8Array(IMG * IMG * 3);
  const sx = sw / IMG;
  const sy = sh / IMG;
  for (let y = 0; y < IMG; y++) {
    const fy = (y + 0.5) * sy - 0.5;
    let y0 = Math.floor(fy);
    const wy = fy - y0;
    if (y0 < 0) { y0 = 0; }
    const y1 = Math.min(sh - 1, y0 + 1);
    for (let x = 0; x < IMG; x++) {
      const fx = (x + 0.5) * sx - 0.5;
      let x0 = Math.floor(fx);
      const wx = fx - x0;
      if (x0 < 0) { x0 = 0; }
      const x1 = Math.min(sw - 1, x0 + 1);
      const i00 = (y0 * sw + x0) * 3;
      const i01 = (y0 * sw + x1) * 3;
      const i10 = (y1 * sw + x0) * 3;
      const i11 = (y1 * sw + x1) * 3;
      const o = (y * IMG + x) * 3;
      for (let c = 0; c < 3; c++) {
        const top = src[i00 + c] * (1 - wx) + src[i01 + c] * wx;
        const bot = src[i10 + c] * (1 - wx) + src[i11 + c] * wx;
        out[o + c] = (top * (1 - wy) + bot * wy) | 0;
      }
    }
  }
  return out;
}
