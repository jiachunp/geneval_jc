# python generation/generate_multi.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated"


CUDA_VISIBLE_DEVICES=0, \
torchrun --nproc_per_node=1 generation/generate_multi.py \
    "./prompts/evaluation_metadata.jsonl" \
    --outdir "generated"