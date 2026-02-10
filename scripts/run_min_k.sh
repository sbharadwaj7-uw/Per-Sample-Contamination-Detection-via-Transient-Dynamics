#!/bin/bash
# Run Min-K% analysis for all datasets

python src/new_approach/min_k_prob.py \
    --model_path models/mistralai--Mistral-7B-Instruct-v0.1/final_model \
    --data_path data/math/probe/eval.parquet data/arc/probe/eval.parquet data/mmlu/probe/eval.parquet \
    --output_base_dir output-min-k
