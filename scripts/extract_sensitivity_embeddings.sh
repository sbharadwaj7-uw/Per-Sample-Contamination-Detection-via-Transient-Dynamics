#!/bin/bash

# Script to run sensitivity-based embedding extraction

CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7,8 python src/new_approach/extract_sensitivity.py \
    --model_path "models/mistralai--Mistral-7B-Instruct-v0.1/final_model" \
    --data_path \
        data/math/probe/train.parquet \
        data/math/probe/eval.parquet \
        data/arc/probe/train.parquet \
        data/arc/probe/eval.parquet \
        data/mmlu/probe/train.parquet \
        data/mmlu/probe/eval.parquet \
    --output_base_dir "embeddings_sensitivity_3" \
    --lr "1e-4"
