# ip_list=("0.0.0.0") # e.g. one node ("1.1.1.1"); multi node ("1,1,1,1" "2,2,2,2")
# user_name=sort

# for((node_rank=0;node_rank<${#ip_list[*]};node_rank++));
# do
#   ssh $user_name@${ip_list[node_rank]} "cd `pwd`;PATH=$PATH \
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     NCCL_ALGO=Ring \
#     NCCL_SOCKET_IFNAME=eth0 \
#     NCCL_SOCKET_NTHREADS=8 \
#     NCCL_NSOCKS_PERTHREAD=2 \
#     NCCL_DEBUG=INFO
#     NCCL_DEBUG_SUBSYS=ALL
#     TORCH_DISTRIBUTED_DEBUG=INFO
#     torchrun --nproc_per_node 1 \
#     --nnodes 1 \
#     --node_rank 0 \
#     --node_rank ${node_rank} \
#     --master_addr=192.168.0.90 \
#     --master_port 11111 /home/sort/ved/LaVIN_2/train.py \
#         --llm_model 7B\
#         --llama_model_path ../data/weights/ \
#         --max_seq_len 512 \
#         --batch_size 1 \
#         --accum_iter 32 \
#         --epochs 20 \
#         --warmup_epochs 2 \
#         --blr 0.009 \
#         --weight_decay 0.02 \
#         --output_dir ./LaVIN-7B/\
#         --adapter_type attn\
#         --adapter_dim 8\
#         --adapter_scale 1\
#         --n_prompt 6 \
#         --prompt_format QCM-ALE \
#         --temperature 10.\
#         --visual_adapter_type router \
#         --gradient_checkpointing \
#         --bits 4bit \
#         --cpu_load
# done



TORCH_DISTRIBUTED_DEBUG=INFO
torchrun --nproc_per_node 1 \
--nnodes 1 \
--node_rank 0 \
--master_port 11111 /home/sort/ved/LaVIN_2/train.py \
    --llm_model 13B\
    --llama_model_path ../data/weights/ \
    --max_seq_len 512 \
    --batch_size 1 \
    --accum_iter 32 \
    --epochs 20 \
    --warmup_epochs 2 \
    --blr 0.009 \
    --weight_decay 0.02 \
    --output_dir ./LaVIN-13B/\
    --adapter_type attn\
    --adapter_dim 16\
    --adapter_scale 1\
    --n_prompt 6 \
    --prompt_format QCM-ALE \
    --temperature 5.\
    --visual_adapter_type router \
    --bits 4bit \
    --gradient_checkpointing \
    --cpu_load
