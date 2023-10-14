CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port 11111 /home/sort/ved/LaVIN_2/demo.py \
    --llm_model 7B\
    --max_seq_len 512 \
    --ckpt_dir ../data/weights/ \
    --adapter_type attn\
    --adapter_path LaVIN-7B/checkpoint-19.pth
    --temperature 10.\
