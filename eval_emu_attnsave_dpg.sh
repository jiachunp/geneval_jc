# python gen_AR/image_generation_attnsave_dpg.py \
#     "./dpg_prompts" \
#     --outdir "generated_emu3_attnsave_dpg"


CUDA_VISIBLE_DEVICES=0, \
torchrun --nproc_per_node=1 gen_AR/image_generation_attnsave_dpg.py \
    "./dpg_prompts" \
    --classifier_free_guidance 10.0 \
    --outdir "generated_emu3_attnsave_dpg"
