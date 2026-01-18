# python generation/generate_multi_base.py \
#     "./prompts/evaluation_metadata.jsonl" \
#     --outdir "generated_base"



CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 /home/aiops/zhangfz/geneval_jc/generation/generate_multi_base.py \
    "/home/aiops/zhangfz/geneval_jc/prompts/evaluation_metadata.jsonl" \
    --scale 7.0 \
    --outdir "/home/aiops/zhangfz/geneval_jc/results/generated_base_7"
