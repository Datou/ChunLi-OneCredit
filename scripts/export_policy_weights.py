"""Dump the chunppo PPO policy weights for a pure-JS forward pass.

The model is small and feed-forward (3.40 M params), so the browser runs it
directly in JavaScript instead of pulling in ONNX Runtime.  That keeps the
deployed page dependency-free and works offline.

Outputs, all under web/models/:

  chunppo_p27.bin    raw little-endian float32, tensors concatenated in index order
  chunppo_p27.json   tensor index (name, shape, offset, count) + metadata
  chunppo_p27_ref.json   reference logits/value for fixed inputs, for the JS
                         parity test (scripts/test_policy_js_parity.mjs)

Run:  .venv\\Scripts\\python.exe scripts\\export_policy_weights.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

CKPT = ROOT / "runs" / "chunppo" / "E7_recovery_20M" / "final.zip"
OUT_DIR = ROOT / "web" / "models"

N_ACTIONS = 27
HIST_LEN = 24
HIST_DIM = HIST_LEN * N_ACTIONS  # 648
IMG = 84


class NatureCNN(nn.Module):
    def __init__(self, in_channels: int = 3, features_dim: int = 1024):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.linear = nn.Sequential(nn.Linear(64 * 7 * 7, features_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class Extractor(nn.Module):
    def __init__(self, cnn_dim: int = 1024, hist_dim: int = 128):
        super().__init__()
        self.cnn = NatureCNN(3, cnn_dim)
        self.hist = nn.Sequential(nn.Linear(HIST_DIM, hist_dim), nn.ReLU())

    def forward(self, image: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.cnn(image), self.hist(hist)], dim=1)


class PolicyNet(nn.Module):
    def __init__(self, n_actions: int = N_ACTIONS):
        super().__init__()
        self.features_extractor = Extractor()
        self.action_net = nn.Linear(1152, n_actions)
        self.value_net = nn.Linear(1152, 1)

    def forward(self, image: torch.Tensor, hist: torch.Tensor):
        f = self.features_extractor(image, hist)
        return self.action_net(f), self.value_net(f)


def load_policy_state(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        blob = z.read("policy.pth")
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=CKPT)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if not args.model.exists():
        print(f"missing checkpoint: {args.model}")
        return 1

    state = load_policy_state(args.model)

    # SB3 shares one feature extractor across the three roles; assert it, so the
    # exported copy really is the one the policy forwards through.
    shared = {
        k: v for k, v in state.items() if k.startswith("features_extractor.")
    }
    for prefix in ("pi_features_extractor.", "vf_features_extractor."):
        for name, tensor in shared.items():
            other = state[prefix + name[len("features_extractor."):]]
            if not torch.equal(tensor, other):
                print(f"ERROR: {prefix}{name} differs from the shared extractor")
                return 1

    own = {
        k: v
        for k, v in state.items()
        if not k.startswith(("pi_features_extractor.", "vf_features_extractor."))
    }
    net = PolicyNet().eval()
    missing, unexpected = net.load_state_dict(own, strict=False)
    real_missing = [m for m in missing if not m.startswith(("pi_", "vf_"))]
    if real_missing or unexpected:
        print(f"ERROR missing={real_missing} unexpected={unexpected}")
        return 1

    n_params = sum(p.numel() for p in net.parameters())
    print(f"policy: {n_params/1e6:.2f} M params, shared extractor verified")

    args.out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- weights
    tensors: list[tuple[str, torch.Tensor]] = []
    for name, tensor in net.state_dict().items():
        tensors.append((name, tensor.detach().to(torch.float32).contiguous()))

    blob = bytearray()
    index = []
    offset = 0
    for name, tensor in tensors:
        raw = tensor.numpy().astype("<f4", copy=False).tobytes()
        index.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "offset": offset,
                "count": int(tensor.numel()),
            }
        )
        blob.extend(raw)
        offset += len(raw)

    bin_path = args.out / "chunppo_p27.bin"
    bin_path.write_bytes(bytes(blob))
    print(f"wrote {bin_path.name}  ({len(blob)/1e6:.1f} MB, {len(index)} tensors)")

    meta = {
        "source": args.model.name,
        "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "params": int(n_params),
        "nActions": N_ACTIONS,
        "histLen": HIST_LEN,
        "histDim": HIST_DIM,
        "imageSize": IMG,
        "dtype": "float32",
        "endian": "little",
        "tensors": index,
    }
    (args.out / "chunppo_p27.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------- reference
    rng = np.random.default_rng(12345)
    cases = []
    for trial in range(8):
        img = rng.random((1, 3, IMG, IMG), dtype=np.float32)
        hist = np.zeros((1, HIST_DIM), np.float32)
        for step in range(HIST_LEN):
            if rng.random() < 0.6:
                hist[0, step * N_ACTIONS + int(rng.integers(0, N_ACTIONS))] = 1.0
        with torch.inference_mode():
            logits, value = net(torch.from_numpy(img), torch.from_numpy(hist))
        cases.append(
            {
                "image": img.reshape(-1).tolist(),
                "hist": hist.reshape(-1).tolist(),
                "logits": logits.numpy().reshape(-1).tolist(),
                "value": float(value.numpy().reshape(-1)[0]),
            }
        )

    ref_path = ROOT / "_web_verify" / "chunppo_p27_ref.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    ref_path.write_text(
        json.dumps({"imageSize": IMG, "histDim": HIST_DIM, "nActions": N_ACTIONS, "cases": cases}),
        encoding="utf-8",
    )
    print(f"wrote {ref_path.name}  ({len(cases)} reference cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
