# python generation/generate_multi.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated"




CFG_SCALES=(2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)

# Loop through each CFG scale
for scale in "${CFG_SCALES[@]}"; do
    echo "Running generation with CFG scale: $scale"
    
    # Format scale for filename (replace . with _)
    scale_str=$(echo $scale | tr '.' '_')
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    torchrun --nproc_per_node=4 generation/generate_multi_dpg.py \
        "./dpg_prompts" \
        --scale $scale \
        --outdir "results/generated_dpg_${scale_str}"
done