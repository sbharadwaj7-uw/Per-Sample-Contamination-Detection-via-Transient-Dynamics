"""
Min-K% Probability for Data Contamination Detection.

This script implements the Min-K% method:
1. Load the contaminated model (LoRA adapter on base model)
2. For each sample:
   a. Compute token-level log probabilities
   b. Select the k% of tokens with minimum probabilities
   c. Average their log-likelihoods
3. Use train.parquet to find optimal threshold (epsilon) for each k
4. Save results with predictions and thresholds to parquet

The hypothesis: text seen during training will have higher min-k% probabilities
(the model "learned" even the outlier tokens), while unseen text will have
lower min-k% probabilities.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple, Set, Dict
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto"
):
    """Load base model with LoRA adapter (same as extract_sensitivity.py)."""
    print(f"Loading base model: {base_model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # Load PEFT adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        is_trainable=False  # We don't need training for Min-K%
    )
    
    model.eval()
    return model, tokenizer


def calculate_token_log_probs(
    text: str,
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int = 2048
) -> Tuple[float, List[float], float]:
    """
    Calculate perplexity and per-token log probabilities for a text.
    
    Returns:
        - perplexity: exp(avg_negative_log_likelihood)
        - all_prob: list of log probabilities for each token (excluding first)
        - avg_log_likelihood: mean of all_prob
    """
    # Tokenize
    input_ids = tokenizer.encode(text, truncation=True, max_length=max_length)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        logits = outputs.logits
    
    # Apply log softmax to get log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Extract log probability for each actual token (shifted by 1 for causal LM)
    all_prob = []
    input_ids_processed = input_ids[0][1:]  # Skip first token (no prediction for it)
    
    for i, token_id in enumerate(input_ids_processed):
        log_probability = log_probs[0, i, token_id].item()
        all_prob.append(log_probability)
    
    perplexity = torch.exp(loss).item()
    avg_log_likelihood = loss.item()  # This is the average negative log likelihood
    
    return perplexity, all_prob, avg_log_likelihood


def compute_min_k_metrics(
    all_prob: List[float],
    k_ratios: List[float] = [0.1, 0.2, 0.3, 0.5]
) -> Dict[str, float]:
    """
    Compute Min-K% metrics for various K values.
    
    For each ratio k, select the k% of tokens with minimum log probabilities
    and compute the negative mean (so higher = more likely contaminated).
    
    Args:
        all_prob: list of log probabilities (negative values)
        k_ratios: list of k values as ratios (0.1 = 10%)
    
    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}
    
    if len(all_prob) == 0:
        for ratio in k_ratios:
            metrics[f"min_k_{int(ratio*100)}"] = 0.0
        return metrics
    
    sorted_probs = np.sort(all_prob)  # Sort ascending (most negative first)
    
    for ratio in k_ratios:
        k_length = max(1, int(len(all_prob) * ratio))
        topk_prob = sorted_probs[:k_length]  # Take k% lowest probs
        # Negative mean so higher values indicate more likely contaminated
        metrics[f"min_k_{int(ratio*100)}"] = -np.mean(topk_prob).item()
    
    return metrics


def process_samples(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    k_ratios: List[float] = [0.1, 0.2, 0.3, 0.5],
    max_length: int = 2048
) -> pd.DataFrame:
    """
    Process all samples and compute Min-K% metrics.
    
    Returns DataFrame with original columns plus Min-K% metrics.
    """
    if 'instruction_text' not in df.columns:
        raise ValueError("DataFrame must contain 'instruction_text' column")
    if 'id_num' not in df.columns:
        raise ValueError("DataFrame must contain 'id_num' column")
    
    device = next(model.parameters()).device
    
    results = []
    
    print(f"Processing {len(df)} samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Min-K% Analysis"):
        text = row['instruction_text']
        
        try:
            perplexity, all_prob, avg_log_likelihood = calculate_token_log_probs(
                text, model, tokenizer, device, max_length
            )
            
            # Compute Min-K% metrics
            min_k_metrics = compute_min_k_metrics(all_prob, k_ratios)
            
            # Build result row
            result = {
                'id_num': row['id_num'],
                'perplexity': perplexity,
                'avg_log_likelihood': avg_log_likelihood,
                'num_tokens': len(all_prob),
            }
            result.update(min_k_metrics)
            
            # Preserve the 'seen' label if it exists
            if 'seen' in row:
                result['seen'] = row['seen']
            
            results.append(result)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  OOM for sample {row['id_num']}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return pd.DataFrame(results)


def find_optimal_thresholds(
    train_df: pd.DataFrame,
    k_ratios: List[float] = [0.1, 0.2, 0.3, 0.5]
) -> Dict[str, Tuple[float, float, float]]:
    """
    Find optimal threshold (epsilon) for each k ratio that maximizes accuracy.
    Also computes ROC-AUC for each k ratio.
    
    For Min-K%, lower values indicate contaminated (seen=True).
    We search over possible thresholds and pick the one with best accuracy.
    
    Args:
        train_df: DataFrame with 'seen' column and min_k_X columns
        k_ratios: list of k ratios
        
    Returns:
        Dictionary mapping metric name to (optimal_threshold, accuracy, auc)
    """
    if 'seen' not in train_df.columns:
        raise ValueError("train_df must contain 'seen' column for threshold learning")
    
    thresholds = {}
    
    for ratio in k_ratios:
        col_name = f"min_k_{int(ratio*100)}"
        if col_name not in train_df.columns:
            continue
            
        values = train_df[col_name].values
        labels = train_df['seen'].values.astype(bool)
        
        # For ROC-AUC: seen text should have LOWER min_k scores
        # So we negate values to make seen=higher score for roc_auc_score
        # (roc_auc_score expects higher scores for positive class)
        try:
            auc = roc_auc_score(labels, -values)
        except ValueError:
            auc = 0.5  # If only one class present
        
        # Search over all unique values as potential thresholds
        unique_vals = np.unique(values)
        
        best_acc = 0.0
        best_threshold = np.median(values)
        
        for thresh in unique_vals:
            # Predict seen=True if min_k < threshold (lower = more likely contaminated)
            # Reasoning: seen text has higher probs for outliers → less negative log probs → lower -mean
            predictions = values < thresh
            accuracy = np.mean(predictions == labels)
            
            if accuracy > best_acc:
                best_acc = accuracy
                best_threshold = thresh
        
        thresholds[col_name] = (best_threshold, best_acc, auc)
        print(f"  {col_name}: best_threshold={best_threshold:.4f}, accuracy={best_acc:.4f}, AUC={auc:.4f}")
    
    return thresholds


def apply_thresholds(
    df: pd.DataFrame,
    thresholds: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Add epsilon (threshold) and prediction columns to the dataframe.
    
    For each min_k_X column, adds:
    - epsilon_k_X: the learned threshold
    - pred_k_X: boolean prediction (True if min_k > threshold, i.e., predicted contaminated)
    """
    df = df.copy()
    
    for col_name, (threshold, _, _) in thresholds.items():
        # Extract k value from column name (e.g., "min_k_10" -> "10")
        k_val = col_name.replace("min_k_", "")
        epsilon_col = f"epsilon_k_{k_val}"
        pred_col = f"pred_k_{k_val}"
        auc_col = f"auc_k_{k_val}"
        
        df[epsilon_col] = threshold
        # Predict seen=True if min_k < threshold (lower = more likely contaminated)
        df[pred_col] = df[col_name] < threshold
        # Add AUC from training
        df[auc_col] = thresholds[col_name][2]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Min-K% Probability Analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--data_path", type=str, nargs='+', required=True, help="Path(s) to eval parquet files")
    parser.add_argument("--output_base_dir", type=str, default="output-min-k", help="Output base directory")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map")
    parser.add_argument(
        "--k_ratios", 
        type=float, 
        nargs='+', 
        default=[0.1, 0.2, 0.3, 0.5],
        help="K ratio values for Min-K%% (e.g., 0.1 = 10%%)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    data_paths = [Path(p) for p in args.data_path]
    
    # Determine model name - use parent/checkpoint structure
    # e.g., models/mistralai--Mistral-7B-Instruct-v0.1/final_model
    #       -> mistralai--Mistral-7B-Instruct-v0.1/final_model
    checkpoint_name = model_path.name  # e.g., "final_model"
    model_name_clean = model_path.parent.name  # e.g., "mistralai--Mistral-7B-Instruct-v0.1"
    model_output_path = f"{model_name_clean}/{checkpoint_name}"
        
    # Load config
    adapter_config_path = model_path / "adapter_config.json"
    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)
    base_model_name = adapter_config['base_model_name_or_path']
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(
        base_model_name,
        str(model_path),
        device_map=args.device_map
    )
    
    for data_path in data_paths:
        print(f"\n{'='*60}")
        print(f"Processing {data_path.name}...")
        
        # Parse path info
        # Expected: data/<dataset>/probe/eval.parquet
        try:
            parts = data_path.parts
            data_idx = parts.index('data')
            dataset_name = parts[data_idx + 1]
        except ValueError:
            print(f"Skipping {data_path}: Invalid path structure")
            continue
        
        # Derive train.parquet path from eval.parquet path
        train_path = data_path.parent / "train.parquet"
        if not train_path.exists():
            print(f"⚠️  Warning: train.parquet not found at {train_path}")
            print("   Cannot learn optimal thresholds. Skipping...")
            continue
            
        # Load data
        eval_df = pd.read_parquet(data_path)
        train_df = pd.read_parquet(train_path)
        
        print(f"Train samples: {len(train_df)}, Eval samples: {len(eval_df)}")
        
        # Process train samples first to learn thresholds
        print("\n--- Processing TRAIN set to learn thresholds ---")
        train_results_df = process_samples(
            model, 
            tokenizer, 
            train_df, 
            k_ratios=args.k_ratios,
            max_length=args.max_length
        )
        
        # Find optimal thresholds on train set
        print("\n--- Finding optimal thresholds ---")
        thresholds = find_optimal_thresholds(train_results_df, k_ratios=args.k_ratios)
        
        # Process eval samples
        print("\n--- Processing EVAL set ---")
        eval_results_df = process_samples(
            model, 
            tokenizer, 
            eval_df, 
            k_ratios=args.k_ratios,
            max_length=args.max_length
        )
        
        # Apply thresholds (add epsilon columns)
        eval_results_df = apply_thresholds(eval_results_df, thresholds)
        
        # Save results
        output_dir = Path(args.output_base_dir) / model_output_path / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "eval.parquet"
        
        eval_results_df.to_parquet(output_file)
        print(f"\n✓ Saved {len(eval_results_df)} results to {output_file}")
        
        # Print summary statistics and save metrics
        metrics_summary = {}
        if 'seen' in eval_results_df.columns:
            print(f"\n--- Eval Summary by seen/unseen ---")
            for k_col in [c for c in eval_results_df.columns if c.startswith('min_k_')]:
                seen_mean = eval_results_df[eval_results_df['seen'] == True][k_col].mean()
                unseen_mean = eval_results_df[eval_results_df['seen'] == False][k_col].mean()
                
                # Get threshold for this k
                if k_col in thresholds:
                    thresh, train_acc, train_auc = thresholds[k_col]
                    # Calculate eval accuracy (seen = min_k < threshold)
                    pred = eval_results_df[k_col] < thresh
                    eval_acc = (pred == eval_results_df['seen']).mean()
                    
                    # Calculate eval AUC
                    eval_labels = eval_results_df['seen'].values.astype(bool)
                    eval_values = eval_results_df[k_col].values
                    try:
                        eval_auc = roc_auc_score(eval_labels, -eval_values)
                    except ValueError:
                        eval_auc = 0.5
                    
                    print(f"  {k_col}: seen={seen_mean:.4f}, unseen={unseen_mean:.4f}, "
                          f"epsilon={thresh:.4f}, train_acc={train_acc:.4f}, eval_acc={eval_acc:.4f}, "
                          f"train_auc={train_auc:.4f}, eval_auc={eval_auc:.4f}")
                    
                    # Store metrics for JSON output
                    metrics_summary[k_col] = {
                        'threshold': float(thresh),
                        'train_accuracy': float(train_acc),
                        'eval_accuracy': float(eval_acc),
                        'train_auc': float(train_auc),
                        'eval_auc': float(eval_auc),
                        'seen_mean': float(seen_mean),
                        'unseen_mean': float(unseen_mean)
                    }
                else:
                    print(f"  {k_col}: seen={seen_mean:.4f}, unseen={unseen_mean:.4f}")
        
        # Save metrics summary as JSON
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"✓ Saved metrics to {metrics_file}")


if __name__ == "__main__":
    main()
