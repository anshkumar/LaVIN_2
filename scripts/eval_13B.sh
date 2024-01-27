CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port 11111 /home/sort/ved/LaVIN_2/eval.py \
    --ckpt_dir ../data/weights/ \
    --llm_model 13B\
    --tokenizer_path ../data/weights/tokenizer.model \
    --data_root ../data \
    --caption_file ../data/captions.json \
    --adapter_path ./LaVIN-13B/checkpoint-19.pth \
    --adapter_type attn \
    --adapter_dim 16 \
    --adapter_scale 1 \
    --prompt_format QCM-ALE \
    --max_batch_size 4\
    --max_seq_len 512 \
    --split test \
    --n_prompt 6 \
    --temperature 5.\
    --visual_adapter_type router\
    --bits 4bit \
    --cpu_load