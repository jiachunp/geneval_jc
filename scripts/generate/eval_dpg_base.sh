# python generation/generate_multi.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated"


CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 generation/generate_multi_base_dpg.py \
    "./dpg_prompts" \
    --scale 7.0 \
    --outdir "results/generated_dpg_base_7"