CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port 11111 /home/sort/ved/LaVIN_2/train.py \
    --llm_model 7B\
    --llama_model_path ../data/weights/ \
    --max_seq_len 128 \
    --batch_size 1 \
    --accum_iter 4 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 9e-4 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-7B/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router


