# python generation/generate_multi.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated"


CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 generation/generate_multi.py \
    "./prompts/evaluation_metadata.jsonl" \
    --outdir "results/generated_try2"
