# python gen_AR/image_generation_attnsave.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated_attnsave"


CUDA_VISIBLE_DEVICES=0, \
torchrun --nproc_per_node=1 gen_AR/image_generation_attnsave.py \
    "./prompts/evaluation_metadata.jsonl" \
    --classifier_free_guidance 10 \
    --outdir "generated_emu3_attnsave_cfg10"
