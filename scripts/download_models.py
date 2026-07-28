"""Fetch the model checkpoints this repo needs but excludes from GitHub
(see .gitignore: checkpoints/, *.ckpt, *.pth, *.pt).

Default (no flags) downloads only the two Depth Anything V2 checkpoints
(~200 MB total) used by nav_pipeline for monocular depth. The full
InternVLA-N1-w-NavDP checkpoint (~16 GB) is opt-in via --vlm since it's
only needed to reproduce the "extracted" NavDP fallback policy.

navdp-cross-modal.ckpt (the crossmodal policy nav_pipeline defaults to)
can't be automated: it's gated behind InternRobotics/NavDP's checkpoint
request form. This script just prints where it must go.

Usage:
  python scripts/download_models.py                # depth anything only
  python scripts/download_models.py --vlm           # + InternVLA-N1-w-NavDP,
                                                      #   then auto-extracts
                                                      #   navdp_extracted.pth
  python scripts/download_models.py --all           # everything automatable
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")

DEPTH_ANYTHING_FILES = [
    ("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
    ("depth-anything/Depth-Anything-V2-Metric-Hypersim-Small", "depth_anything_v2_metric_hypersim_vits.pth"),
]

VLM_REPO = "InternRobotics/InternVLA-N1-w-NavDP"
VLM_DIR = os.path.join(CKPT_DIR, "InternVLA-N1-w-NavDP")


def download_depth_anything():
    from huggingface_hub import hf_hub_download

    os.makedirs(CKPT_DIR, exist_ok=True)
    for repo_id, filename in DEPTH_ANYTHING_FILES:
        dest = os.path.join(CKPT_DIR, filename)
        if os.path.exists(dest):
            print(f"[skip] {filename} already present")
            continue
        print(f"Downloading {filename} from {repo_id} ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        if os.path.islink(dest) or os.path.exists(dest):
            os.remove(dest)
        os.symlink(os.path.abspath(path), dest)
        print(f"  -> {dest}")


def download_vlm():
    from huggingface_hub import snapshot_download

    if os.path.exists(os.path.join(VLM_DIR, "model.safetensors.index.json")):
        print(f"[skip] {VLM_REPO} already present at {VLM_DIR}")
        return
    print(f"Downloading {VLM_REPO} (~16 GB, this will take a while) ...")
    snapshot_download(repo_id=VLM_REPO, local_dir=VLM_DIR)
    print(f"  -> {VLM_DIR}")


def extract_navdp():
    out_path = os.path.join(CKPT_DIR, "navdp_extracted.pth")
    if os.path.exists(out_path):
        print("[skip] navdp_extracted.pth already present")
        return
    print("Extracting NavDP System-1 weights from InternVLA-N1-w-NavDP ...")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "extract_navdp_weights.py")], check=True)


def print_crossmodal_instructions():
    dest = os.path.join(ROOT, "navdp-cross-modal.ckpt")
    if os.path.exists(dest):
        print(f"[skip] navdp-cross-modal.ckpt already present")
        return
    print(
        "\nnavdp-cross-modal.ckpt (543 MB, the crossmodal policy nav_pipeline defaults to) "
        "is NOT downloadable automatically -- it's gated behind the official "
        "InternRobotics/NavDP checkpoint request form (https://github.com/InternRobotics/NavDP). "
        f"Once you have it, place it at:\n  {dest}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm", action="store_true", help="also download InternVLA-N1-w-NavDP (~16 GB) and extract navdp_extracted.pth")
    parser.add_argument("--all", action="store_true", help="shorthand for --vlm")
    args = parser.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface_hub")

    download_depth_anything()

    if args.vlm or args.all:
        download_vlm()
        extract_navdp()

    print_crossmodal_instructions()
    print("Done.")


if __name__ == "__main__":
    main()
