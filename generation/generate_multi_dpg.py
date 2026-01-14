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

from pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from transformer_sd3 import SD3Transformer2DModel

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
    metadatas = load_prompts(opt.metadata_file)

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
        prompt_name = metadata.get("name", f"{global_idx:0>5}")

        # Deterministic: seed per prompt so results are stable across GPU counts / sharding patterns.
        prompt_seed = opt.seed + global_idx
        set_seeds(prompt_seed)
        generator = torch.Generator(device=device).manual_seed(prompt_seed)

        # Ensure output dir exists
        os.makedirs(opt.outdir, exist_ok=True)
        save_file = os.path.join(opt.outdir, prompt_name)
        os.makedirs(save_file, exist_ok=True)

        # Save prompt metadata next to outputs
        # meta_out = {
        #     "prompt": prompt,
        #     "prompt_file": metadata.get("path"),
        #     "name": prompt_name,
        #     "index": global_idx,
        # }
        # with open(os.path.join(opt.outdir, f"{prompt_name}.json"), "w", encoding="utf-8") as fp:
        #     json.dump(meta_out, fp, ensure_ascii=False, indent=2)

        batch_size = opt.batch_size
        n_rows = batch_size

        print(f"[rank={rank}] Prompt ({global_idx: >5}/{len(metadatas)} | {prompt_name}): {prompt!r}")

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
                    img.save(os.path.join(save_file, f"{sample_count:05}.png"))
                    sample_count += 1

                if not opt.skip_grid:
                    all_samples.append(torch.stack([ToTensor()(img) for img in images], 0))

        if (not opt.skip_grid) and len(all_samples) > 0:
            grid = torch.cat(all_samples, dim=0)  # [N, C, H, W]
            grid = make_grid(grid, nrow=n_rows)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(os.path.join(opt.outdir, f"{prompt_name}.png"))

    # Optional: make sure everything is done before exit
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)