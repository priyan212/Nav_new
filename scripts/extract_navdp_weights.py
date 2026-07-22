"""Extract NavDP (System-1) weights from the InternVLA-N1-w-NavDP checkpoint.

The full checkpoint is ~16 GB (Qwen2.5-VL 7B + NavDP). Only the
`model.navdp.*` tensors are needed to run NavDP standalone; this writes them
(with the prefix stripped) to checkpoints/navdp_extracted.pth so the policy
can be loaded without touching the VLM.

Run:  python scripts/extract_navdp_weights.py
"""

import json
import os

import torch
from safetensors import safe_open

CKPT_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "InternVLA-N1-w-NavDP")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "navdp_extracted.pth")
PREFIX = "model.navdp."


def main():
    index = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    navdp_keys = {k: v for k, v in weight_map.items() if k.startswith(PREFIX)}
    print(f"Found {len(navdp_keys)} NavDP tensors across {len(set(navdp_keys.values()))} shards")

    state = {}
    for shard in sorted(set(navdp_keys.values())):
        keys = [k for k, v in navdp_keys.items() if v == shard]
        with safe_open(os.path.join(CKPT_DIR, shard), framework="pt", device="cpu") as f:
            for k in keys:
                state[k[len(PREFIX):]] = f.get_tensor(k)

    total = sum(t.numel() * t.element_size() for t in state.values())
    print(f"Extracted {len(state)} tensors, {total / 1e6:.1f} MB")
    torch.save(state, OUT_PATH)
    print(f"Saved -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
