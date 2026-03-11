# python gen_AR/image_generation_attnsave.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated_attnsave"


# CUDA_VISIBLE_DEVICES=0, \
# torchrun --nproc_per_node=1 gen_AR/image_generation.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --classifier_free_guidance 10.0 \
#     --outdir "generated_emu3_base_cfg10"


CFG_SCALES=(16.0 18.0 20.0)
# CFG_SCALES=(10.0)

# Loop through each CFG scale
for scale in "${CFG_SCALES[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 gen_AR/image_generation.py \
        "./prompts/evaluation_metadata.jsonl" \
        --classifier_free_guidance ${scale} \
        --outdir "results/generated_emu3_base_cfg_${scale}"
done
