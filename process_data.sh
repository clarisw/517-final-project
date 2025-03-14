#!/usr/bin/env bash

model_name="meta-llama/Llama-2-7b-hf"
# model_name="meta-llama/Llama-3.1-8B-Instruct"

# total_idx=200

# start_idx=0
# end_idx=25

# start_idx=25
# end_idx=50

# start_idx=50
# end_idx=75

start_idx=75
end_idx=100

# race="Asian"
# race="Black"
# race="Hispanic"
race="White"

CUDA_VISIBLE_DEVICES=0 python process_data.py \
    --model-name $model_name \
    --start-idx $start_idx \
    --end-idx $end_idx \
    --race $race
