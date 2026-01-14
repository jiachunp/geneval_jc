# python generation/generate_multi_base.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated_base"



CUDA_VISIBLE_DEVICES=0, \
torchrun --nproc_per_node=1 generation/generate_multi_base.py \
    "./prompts/evaluation_metadata.jsonl" \
    --scale 7 \
    --outdir "generated_base"
