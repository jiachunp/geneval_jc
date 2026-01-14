import argparse
import json
import os
import math
import random
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor

from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from transformer_sd3 import SD3Transformer2DModel

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="JSONL file containing lines of metadata for each prompt")

    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples per prompt")
    parser.add_argument("--steps", type=int, default=28, help="number of sampling steps")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="negative prompt for guidance",
    )
    parser.add_argument("--scale", type=float, default=4.0, help="guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="base seed (reproducible sampling)")

    parser.add_argument("--batch_size", type=int, default=1, help="num_images_per_prompt per call")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")
    parser.add_argument("--attention_slicing", action="store_true", help="enable attention slicing")

    # SD3 paths
    parser.add_argument(
        "--sd3_path",
        type=str,
        default="/root/autodl-tmp/stable-diffusion-3-medium-diffusers",
        help="path to SD3 diffusers folder",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="/root/autodl-tmp/stable-diffusion-3-medium-diffusers/transformer",
        help="path to transformer folder/ckpt",
    )

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


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_metadatas(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def build_pipeline(opt, device: torch.device):
    pipe = StableDiffusion3Pipeline.from_pretrained(opt.sd3_path, torch_dtype=torch.float16)

    new_transformer = SD3Transformer2DModel.from_pretrained(
        opt.transformer_path,
        torch_dtype=pipe.dtype,
    )
    pipe.transformer = new_transformer

    pipe = pipe.to(device)
    if opt.attention_slicing:
        pipe.enable_attention_slicing()
    return pipe


def main(opt):
    rank, world_size, local_rank = get_dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Each rank loads full list but processes only its shard (simple and robust).
    metadatas = load_metadatas(opt.metadata_file)

    # Build model on this GPU
    pipe = build_pipeline(opt, device)

    # Shard indices: rank, rank+world_size, ...
    my_indices = list(range(rank, len(metadatas), world_size))

    if rank == 0:
        print(f"[world_size={world_size}] total prompts: {len(metadatas)}")

    print(f"[rank={rank}] processing {len(my_indices)} prompts on {device}")

    os.makedirs(opt.outdir, exist_ok=True)

    for global_idx in my_indices:
        metadata = metadatas[global_idx]
        prompt = metadata["prompt"]

        # Deterministic: seed per prompt (and per rank) so results are stable
        # across different GPU counts / sharding patterns.
        prompt_seed = opt.seed + global_idx
        set_seeds(prompt_seed)
        generator = torch.Generator(device=device).manual_seed(prompt_seed)

        outpath = os.path.join(opt.outdir, f"{global_idx:0>5}")
        os.makedirs(outpath, exist_ok=True)

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        # Save metadata once
        with open(os.path.join(outpath, "metadata.json"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)

        batch_size = opt.batch_size
        n_rows = batch_size

        print(f"[rank={rank}] Prompt ({global_idx: >5}/{len(metadatas)}): {prompt!r}")

        sample_count = 0
        all_samples = []

        # total calls needed
        n_calls = math.ceil(opt.n_samples / batch_size)

        with torch.no_grad():
            for _ in trange(n_calls, desc=f"rank{rank} sampling", leave=False):
                cur_bs = min(batch_size, opt.n_samples - sample_count)
                if cur_bs <= 0:
                    break

                result = pipe(
                    prompt,
                    num_inference_steps=opt.steps,
                    guidance_scale=opt.scale,
                    num_images_per_prompt=cur_bs,
                    negative_prompt=opt.negative_prompt if opt.negative_prompt else None,
                    generator=generator,
                )
                images = result.images

                for img in images:
                    img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1

                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(img) for img in images], 0))

        if (not opt.skip_grid) and len(all_samples) > 0:
            grid = torch.cat(all_samples, dim=0)  # [N, C, H, W]
            grid = make_grid(grid, nrow=n_rows)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, "grid.png"))

    # Optional: make sure everything is done before exit
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)