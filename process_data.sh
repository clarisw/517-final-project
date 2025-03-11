#!/usr/bin/env bash

model_name="meta-llama/Llama-2-7b-hf"
# model_name="meta-llama/Llama-3.1-8B-Instruct"

# total_idx=200

start_idx=0
end_idx=10

# start_idx=10
# end_idx=50

# start_idx=50
# end_idx=100

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python process_data.py \
    --model-name $model_name \
    --start-idx $start_idx \
    --end-idx $end_idx
