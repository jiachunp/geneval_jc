# python generation/generate_multi.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated"


CUDA_VISIBLE_DEVICES=0, \
torchrun --nproc_per_node=1 generation/generate_multi_base_dpg.py \
    "./dpg_prompts" \
    --scale 7.0 \
    --outdir "generated_dpg_base"