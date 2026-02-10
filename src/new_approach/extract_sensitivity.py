"""
Extract sensitivity embeddings (trajectory) from LoRA-tuned models.

This script implements the "sensitivity" approach:
1. Load the contaminated model
2. For each sample:
   a. Extract embedding (step 0)
   b. Perform N gradient steps on that sample
   c. Extract embedding after each step
   d. Reset model to original state
3. Save the full embedding trajectory (N+1 embeddings per sample)

This allows detecting contamination by measuring how much the model "moves" 
when trained on a specific sample.
"""

import os
import argparse
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple, Set
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm import tqdm


def get_last_non_special_token_embedding(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    input_ids: torch.Tensor,
    special_token_ids: Set[int]
) -> torch.Tensor:
    """
    Extract the embedding of the last non-special token from hidden states.
    """
    batch_size = hidden_states.shape[0]
    embeddings = []
    
    for i in range(batch_size):
        # Find the last non-padding position
        valid_length = attention_mask[i].sum().item()
        
        # Work backwards from the last valid position to find last non-special token
        last_non_special_idx = valid_length - 1
        for idx in range(valid_length - 1, -1, -1):
            token_id = input_ids[i, idx].item()
            if token_id not in special_token_ids:
                last_non_special_idx = idx
                break
        
        # Get the embedding at the last non-special token position
        last_embedding = hidden_states[i, last_non_special_idx, :]
        embeddings.append(last_embedding)
    
    return torch.stack(embeddings)


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: str,
    device_map: str = "auto"
):
    """Load base model with LoRA adapter."""
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
        is_trainable=True # Important: we need to train it
    )
    
    # Ensure only adapter parameters are trainable
    model.print_trainable_parameters()
    
    return model, tokenizer


def process_single_sample(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    special_token_ids: Set[int],
    optimizer_cls,
    lr: float,
    num_steps: int,
    device: torch.device,
    max_length: int = 2048
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Process a single sample: extract embeddings at each step.
    Returns: (embeddings_trajectory, losses, grad_norms)
    embeddings_trajectory shape: (num_steps + 1, embedding_dim)
    """
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Labels for training (causal LM)
    inputs['labels'] = inputs['input_ids'].clone()
    
    embeddings_list = []

    # 1. Get pre_ft embedding (Step 0)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embedding = get_last_non_special_token_embedding(
            hidden_states,
            inputs['attention_mask'],
            inputs['input_ids'],
            special_token_ids
        )
        embeddings_list.append(embedding.cpu().float().numpy())
    
    # 2. Train steps
    model.train()
    # Create a fresh optimizer for this sample
    optimizer = optimizer_cls(model.parameters(), lr=lr)
    
    losses = []
    grad_norms = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        outputs_train = model(**inputs)
        loss = outputs_train.loss
        loss.backward()
        
        losses.append(loss.item())
            
        # Compute gradient norm for trainable parameters
        total_norm = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norms.append(total_norm ** 0.5)
        
        optimizer.step()

        # Extract embedding after step
        model.eval()
        with torch.no_grad():
            outputs_step = model(**inputs, output_hidden_states=True)
            hidden_states_step = outputs_step.hidden_states[-1]
            embedding_step = get_last_non_special_token_embedding(
                hidden_states_step,
                inputs['attention_mask'],
                inputs['input_ids'],
                special_token_ids
            )
            embeddings_list.append(embedding_step.cpu().float().numpy())
        model.train()
        
    return (
        np.vstack(embeddings_list),
        losses,
        grad_norms
    )


def extract_sensitivity_embeddings(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    lr: float = 1e-4,
    num_steps: int = 1,
    max_length: int = 2048
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings trajectory for all samples in dataframe.
    Returns: (all_embeddings, id_nums, losses, grad_norms)
    """
    if 'instruction_text' not in df.columns:
        raise ValueError("DataFrame must contain 'instruction_text' column")
    if 'id_num' not in df.columns:
        raise ValueError("DataFrame must contain 'id_num' column")
    
    special_token_ids = set(tokenizer.all_special_ids)
    device = next(model.parameters()).device
    
    all_embeddings_list = []
    id_nums_list = []
    losses_list = []
    grad_norms_list = []
    
    # Save initial state of the adapter
    # We only need to save/restore the adapter weights since base model is frozen
    initial_adapter_state = {k: v.cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
    
    # Optimizer class (AdamW for consistency with training)
    optimizer_cls = optim.AdamW
    
    print(f"Processing {len(df)} samples (steps={num_steps})...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Sensitivity Analysis"):
        text = row['instruction_text']
        id_num = row['id_num']
        
        # Ensure model is in initial state
        set_peft_model_state_dict(model, initial_adapter_state)
        
        try:
            sample_embeddings, sample_losses, sample_grad_norms = process_single_sample(
                model,
                tokenizer,
                text,
                special_token_ids,
                optimizer_cls,
                lr,
                num_steps,
                device,
                max_length
            )
            
            all_embeddings_list.append(sample_embeddings)
            id_nums_list.append(id_num)
            losses_list.append(sample_losses)
            grad_norms_list.append(sample_grad_norms)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  OOM for sample {id_num}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                
    # Restore model to initial state one last time to be safe
    set_peft_model_state_dict(model, initial_adapter_state)
    
    # Stack results
    # sample_embeddings is (steps+1, dim)
    # all_embeddings will be (num_samples, steps+1, dim)
    all_embeddings = np.stack(all_embeddings_list)
    id_nums = np.array(id_nums_list)
    losses = np.array(losses_list)
    grad_norms = np.array(grad_norms_list)
    
    return all_embeddings, id_nums, losses, grad_norms


def save_embeddings(
    embeddings: np.ndarray,
    id_nums: np.ndarray,
    output_dir: Path,
    losses: Optional[np.ndarray] = None,
    grad_norms: Optional[np.ndarray] = None
):
    """Save embeddings, id_nums, and optional metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "id_nums.npy", id_nums)
    
    if losses is not None:
        np.save(output_dir / "losses.npy", losses)
        
    if grad_norms is not None:
        np.save(output_dir / "grad_norms.npy", grad_norms)
    
    metadata = {
        'num_samples': len(embeddings),
        'embedding_shape': str(embeddings.shape),
        'id_nums_min': int(id_nums.min()) if len(id_nums) > 0 else 0,
        'id_nums_max': int(id_nums.max()) if len(id_nums) > 0 else 0,
        'has_losses': losses is not None,
        'has_grad_norms': grad_norms is not None
    }
    
    with open(output_dir / "metadata.txt", 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def main():
    parser = argparse.ArgumentParser(description="Extract sensitivity embeddings")
    parser.add_argument("--model_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--data_path", type=str, nargs='+', required=True, help="Path(s) to parquet files")
    parser.add_argument("--output_base_dir", type=str, default="embeddings_sensitivity", help="Output base directory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for single step update")
    parser.add_argument("--num_steps", type=int, default=5, help="Number of gradient steps per sample")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map")
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    data_paths = [Path(p) for p in args.data_path]
    
    # Determine model name
    if model_path.name in ['final_model', 'checkpoint-*']:
        model_name = model_path.parent.name
    else:
        model_name = model_path.name
        
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
        print(f"\nProcessing {data_path.name}...")
        
        # Parse path info
        # Expected: data/<dataset>/probe/<split>.parquet
        try:
            parts = data_path.parts
            data_idx = parts.index('data')
            dataset_name = parts[data_idx + 1]
            probe_level = parts[data_idx + 2]
            split = data_path.stem
        except ValueError:
            print(f"Skipping {data_path}: Invalid path structure")
            continue
            
        # Load data
        df = pd.read_parquet(data_path)
        
        # Extract
        all_emb, id_nums, losses, grad_norms = extract_sensitivity_embeddings(
            model, tokenizer, df, lr=args.lr, num_steps=args.num_steps, max_length=args.max_length
        )
        
        # Save Trajectory
        output_dir = Path(args.output_base_dir) / model_name / dataset_name / "trajectory" / probe_level / split
        save_embeddings(all_emb, id_nums, output_dir, losses=losses, grad_norms=grad_norms)
        print(f"Saved trajectory to {output_dir}")

if __name__ == "__main__":
    main()
