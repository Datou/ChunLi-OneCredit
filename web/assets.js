import { t } from "./i18n.js?v=29";

/** Download an asset, timing out stalled connections and allowing slow transfers. */
export async function download(url, { signal, onProgress, idleMs = 25_000 } = {}) {
  const request = new AbortController();
  const cancel = () => request.abort(signal.reason);
  signal?.addEventListener("abort", cancel, { once: true });
  if (signal?.aborted) cancel();
  let timer;
  const keepAlive = () => {
    clearTimeout(timer);
    timer = setTimeout(() => request.abort(new Error(t("download.stalled"))), idleMs);
  };
  keepAlive();
  try {
    const response = await fetch(url, { signal: request.signal });
    if (!response.ok) throw new Error(t("download.failed", { status: response.status }));
    keepAlive();
    if (!response.body) return new Uint8Array(await response.arrayBuffer());

    // GitHub Pages can serve these assets compressed; fetch decodes the body,
    // so Content-Length would describe the encoded size and make the progress
    // bar finish early.  Only trust it for identity responses.
    const encoding = response.headers.get("content-encoding");
    const declared = Number(response.headers.get("content-length"));
    const total = !encoding || encoding === "identity"
      ? (Number.isFinite(declared) ? declared : 0)
      : 0;
    const reader = response.body.getReader();
    const chunks = [];
    let bytes = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      keepAlive();
      chunks.push(value);
      bytes += value.length;
      onProgress?.(bytes, total);
    }
    const buffer = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }
    return buffer;
  } catch (error) {
    throw request.signal.aborted ? request.signal.reason : error;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", cancel);
  }
}

export function withDeadline(promise, ms, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}
