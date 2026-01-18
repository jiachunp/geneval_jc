import argparse
import json
import os
import math
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="Path to a directory containing many .txt prompt files (each file is one prompt), or a single .txt file",
    )

    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=1, help="num_images_per_prompt per call")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")

    opt = parser.parse_args()
    return opt


def get_dist_info():
    """torchrun sets these env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def load_prompts(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    files: List[Path]

    if p.is_dir():
        files = sorted(p.glob("*.txt"))
    elif p.is_file() and p.suffix.lower() == ".txt":
        files = [p]
    else:
        raise ValueError(
            f"metadata_file must be a directory of .txt prompts or a single .txt file, got: {path}"
        )

    prompts: List[Dict[str, Any]] = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        prompts.append({"prompt": text, "name": f.stem, "path": str(f)})

    if len(prompts) == 0:
        raise ValueError(f"No non-empty .txt prompts found under: {path}")

    return prompts


def main(opt):
    rank, world_size, local_rank = get_dist_info()

    # Each rank loads full list but processes only its shard
    metadatas = load_prompts(opt.metadata_file)

    # Shard indices: rank, rank+world_size, ...
    my_indices = list(range(rank, len(metadatas), world_size))

    if rank == 0:
        print(f"[world_size={world_size}] total prompts: {len(metadatas)}")

    print(f"[rank={rank}] processing {len(my_indices)} prompts")

    os.makedirs(opt.outdir, exist_ok=True)

    batch_size = opt.batch_size
    n_rows = batch_size*2

    for global_idx in my_indices:
        metadata = metadatas[global_idx]
        prompt = metadata["prompt"]
        prompt_name = metadata.get("name", f"{global_idx:0>5}")

        save_file = os.path.join(opt.outdir, prompt_name)
        final_image_path = os.path.join(opt.outdir, f"{prompt_name}.png")

        if not os.path.exists(save_file):
            print(f"[rank={rank}] Skip {prompt_name}: folder doesn't exist")
            continue

        print(f"[rank={rank}] Regenerating grid for ({global_idx: >5}/{len(metadatas)} | {prompt_name}): {prompt!r}")

        # Load all existing images from the folder
        image_files = sorted([f for f in os.listdir(save_file) if f.endswith('.png')])
        
        if len(image_files) == 0:
            print(f"[rank={rank}] Skip {prompt_name}: no images found in folder")
            continue

        all_samples = []
        for img_file in image_files:
            img_path = os.path.join(save_file, img_file)
            img = Image.open(img_path)
            all_samples.append(ToTensor()(img))

        if not opt.skip_grid and len(all_samples) > 0:
            grid = torch.stack(all_samples, dim=0)  # [N, C, H, W]
            grid = make_grid(grid, nrow=n_rows)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(final_image_path)
            print(f"[rank={rank}] Saved grid to {final_image_path}")

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)