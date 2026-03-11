# python gen_AR/image_generation_attnsave.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated_attnsave"

CFG_SCALES=(14.0 16.0 18.0 20.0)

# Loop through each CFG scale
for scale in "${CFG_SCALES[@]}"; do
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=1 gen_AR/image_generation_attnsave.py \
        "./prompts/evaluation_metadata.jsonl" \
        --classifier_free_guidance ${scale} \
        --outdir "results/generated_emu3_attnsave_cfg_${scale}"
done
