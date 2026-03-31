# -*- coding: utf-8 -*-
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
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
#from logits_process_attnsave import BatchedClassifierFreeGuidanceLogitsProcessor
import sys
sys.path.append("./")
from emu3.mllm.modeling_emu3 import Emu3ForCausalLM
from emu3.mllm.processing_emu3 import Emu3Processor

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="JSONL file containing lines of metadata for each prompt")
    
    parser.add_argument("--outdir", type=str, default="outputs", help="dir to write results to")
    parser.add_argument("--n_samples", type=int, default=1, help="number of samples per prompt (always 1 for this script)")
    parser.add_argument("--seed", type=int, default=42, help="base seed (reproducible sampling)")
    parser.add_argument("--batch_size", type=int, default=1, help="num_images_per_prompt per call")
    
    # Model paths
    parser.add_argument(
        "--emu_hub",
        type=str,
        default="/home/aiops/zhangfz/pretrained_models/Emu3-Gen",
        help="path to Emu3-Gen model",
    )
    parser.add_argument(
        "--vq_hub",
        type=str,
        default="/home/aiops/zhangfz/pretrained_models/Emu3-VisionTokenizer",
        help="path to Emu3-VisionTokenizer model",
    )
    
    # Generation parameters
    parser.add_argument("--classifier_free_guidance", type=float, default=1.0, help="classifier free guidance scale")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--positive_prompt", type=str, default="", help="positive prompt suffix")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--ratio", type=str, default="1:1", help="image ratio (e.g., 1:1, 16:9)")
    
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


def build_model_and_processor(opt, device: torch.device):
    """Build model and processor on the specified device."""
    model = Emu3ForCausalLM.from_pretrained(
        opt.emu_hub,
        device_map=f"cuda:{device.index}" if device.type == "cuda" else "cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(opt.emu_hub, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(opt.vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        opt.vq_hub, 
        device_map=f"cuda:{device.index}" if device.type == "cuda" else "cpu", 
        trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    
    return model, processor


def generate_images_for_prompt(
    model,
    processor,
    prompt: str,
    opt,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    n_samples: int = 1,
    temperature: float = 1.0,
):
    """Generate images for a single prompt."""
    # Prepare prompts: always add a blank prompt "" after the formal prompt (as in original code)
    prompts = [prompt, ""]  # Always include blank prompt as second element
    prompts = [p + opt.positive_prompt for p in prompts]  # Add positive_prompt suffix to each
    
    # Prepare inputs
    kwargs = dict(
        mode='G',
        ratio=[opt.ratio] * len(prompts),  # Use len(prompts) instead of n_samples
        image_area=model.config.image_area,
        return_tensors="pt",
        padding="longest",
    )
    
    pos_inputs = processor(text=prompts, **kwargs)
    neg_inputs = processor(text=[opt.negative_prompt] * len(prompts), **kwargs)
    
    # Prepare generation config
    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
        temperature=temperature,
    )
    
    # Prepare logits processor
    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    logits_processor = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            opt.classifier_free_guidance,
            model,
            unconditional_ids=neg_inputs.input_ids.to(device),
        ),
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ])
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            pos_inputs.input_ids.to(device),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
        )
    
    # Decode images - only return the first n_samples images (even though we generated with blank prompt)
    images = []
    for idx, out in enumerate(outputs):
        if idx >= n_samples:  # Only take first n_samples
            break
        mm_list = processor.decode(out)
        for im in mm_list:
            if isinstance(im, Image.Image):
                images.append(im)
                if len(images) >= n_samples:  # Stop when we have enough
                    break
        if len(images) >= n_samples:
            break
    
    return images[:n_samples]  # Ensure we only return n_samples


def main(opt):
    rank, world_size, local_rank = get_dist_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Each rank loads full list but processes only its shard (simple and robust).
    metadatas = load_metadatas(opt.metadata_file)
    
    # Build model on this GPU
    model, processor = build_model_and_processor(opt, device)
    
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
        
        if os.path.isdir(sample_path):
            existing = [
                fn for fn in os.listdir(sample_path)
                if fn.endswith(".png") and fn[:-4].isdigit()   # only 00000.png style
            ]
            if len(existing) >= opt.n_samples:
                print(f"[rank={rank}] Skip {global_idx:0>5}: already has {len(existing)}/{opt.n_samples} samples")
                continue
        os.makedirs(sample_path, exist_ok=True)
        
        # Save metadata once
        with open(os.path.join(outpath, "metadata.jsonl"), "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False)
            fp.write("\n")
        
        batch_size = opt.batch_size
        print(f"[rank={rank}] Prompt ({global_idx: >5}/{len(metadatas)}): {prompt!r}")
        
        sample_count = 0
        
        # total calls needed
        n_calls = math.ceil(opt.n_samples / batch_size)
        
        for _ in trange(n_calls, desc=f"rank{rank} sampling", leave=False):
            cur_bs = min(batch_size, opt.n_samples - sample_count)
            if cur_bs <= 0:
                break
            
            images = generate_images_for_prompt(
                model,
                processor,
                prompt,
                opt,
                device,
                generator,
                n_samples=cur_bs,
                temperature=opt.temperature,
            )
            
            for img in images:
                if sample_count >= opt.n_samples:
                    break
                img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
                
<<<<<<< Updated upstream
            # Free memory after this prompt's samples so next prompt doesn't see 2x peak
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
=======
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free memory after this prompt's samples so next prompt doesn't see 2x peak
>>>>>>> Stashed changes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Optional: make sure everything is done before exit
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
