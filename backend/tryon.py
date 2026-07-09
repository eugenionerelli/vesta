"""Try-on CatVTON su Apple Silicon (MPS) - Vesta.

Prende foto-persona + capo (flat) -> genera la maschera dell'area capo -> esegue la
pipeline di diffusione CatVTON e salva il render. Misura i tempi reali.
"""
import os
import sys
import time

BACKEND = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BACKEND, ".hf-cache")
CATVTON_DIR = os.path.join(BACKEND, "CatVTON")
os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # per eventuali op non supportate su MPS

sys.path.insert(0, CATVTON_DIR)  # per `import model.pipeline` e `import utils` del repo
sys.path.insert(0, BACKEND)      # per `import mask_from_person`

import torch
from PIL import Image

from model.pipeline import CatVTONPipeline
from mask_from_person import garment_mask


def read_paths() -> dict:
    paths = {}
    with open(os.path.join(BACKEND, "weights_paths.txt")) as fh:
        for line in fh:
            key, val = line.strip().split("=", 1)
            paths[key] = val
    return paths


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--person", default=os.path.join(CATVTON_DIR, "resource/demo/example/person/women/049713_0.jpg"))
    ap.add_argument("--cloth", default=os.path.join(CATVTON_DIR, "resource/demo/example/condition/upper/24083449_54173465_2048.jpg"))
    ap.add_argument("--category", default="upper", choices=["upper", "lower", "overall"])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--out", default=os.path.join(BACKEND, "outputs/tryon_test.png"))
    args = ap.parse_args()

    paths = read_paths()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    person = Image.open(args.person).convert("RGB")
    cloth = Image.open(args.cloth).convert("RGB")

    t0 = time.perf_counter()
    mask = garment_mask(person, args.category)
    t_mask = time.perf_counter() - t0
    mask.save(args.out.replace(".png", "_mask.png"))

    t0 = time.perf_counter()
    pipe = CatVTONPipeline(
        base_ckpt=paths["BASE"],
        attn_ckpt=paths["MIX"],
        attn_ckpt_version="mix",
        weight_dtype=torch.bfloat16,
        device="mps",
        skip_safety_check=True,
        use_tf32=True,
    )
    t_load = time.perf_counter() - t0

    generator = torch.Generator(device="cpu").manual_seed(42)
    t0 = time.perf_counter()
    result = pipe(
        person,
        cloth,
        mask,
        num_inference_steps=args.steps,
        guidance_scale=2.5,
        height=args.height,
        width=args.width,
        generator=generator,
    )[0]
    t_inf = time.perf_counter() - t0

    result.save(args.out)
    print("------ TEMPI ------")
    print(f"maschera : {t_mask:.1f}s")
    print(f"caricam. : {t_load:.1f}s")
    print(f"inferenza: {t_inf:.1f}s  ({args.steps} step, {args.width}x{args.height})")
    print("salvato  :", args.out)
    try:
        print(f"RAM MPS  : {torch.mps.current_allocated_memory() / 1e9:.2f} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
