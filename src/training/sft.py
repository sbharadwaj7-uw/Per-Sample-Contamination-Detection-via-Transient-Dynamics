"""
Fine-tuning script for multi-task dataset using TRL and LoRA.
"""

import os
import sys
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.multi_task_dataset import MultiTaskDataset
os.environ["WANDB_PROJECT"] = "contamination"  # or your desired project name

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(config: dict):
    """Create train and validation datasets from separate files."""
    # Load training dataset (contamination)
    train_dataset_pytorch = MultiTaskDataset(
        math_path=config['data']['math_train_path'],
        mmlu_path=config['data']['mmlu_train_path'],
        arc_path=config['data']['arc_train_path'],
        keep_seen=config['data'].get('keep_seen', False)
    )
    
    # Print training dataset info
    train_info = train_dataset_pytorch.get_dataset_info()
    
    print(f"\nTraining dataset composition:")
    print(f"  MATH: {train_info['math_count']}")
    print(f"  MMLU: {train_info['mmlu_count']}")
    print(f"  ARC: {train_info['arc_count']}")
    print(f"  Total: {train_info['total_count']}")
    
    # Check if validation paths are provided
    has_val_paths = all([
        config['data'].get('math_val_path'),
        config['data'].get('mmlu_val_path'),
        config['data'].get('arc_val_path')
    ])
    
    val_dataset_pytorch = None
    if has_val_paths:
        # Load validation dataset
        val_dataset_pytorch = MultiTaskDataset(
            math_path=config['data']['math_val_path'],
            mmlu_path=config['data']['mmlu_val_path'],
            arc_path=config['data']['arc_val_path'],
            keep_seen=config['data'].get('keep_seen', False)
        )
        
        val_info = val_dataset_pytorch.get_dataset_info()
        print(f"\nValidation dataset composition:")
        print(f"  MATH: {val_info['math_count']}")
        print(f"  MMLU: {val_info['mmlu_count']}")
        print(f"  ARC: {val_info['arc_count']}")
        print(f"  Total: {val_info['total_count']}")
    else:
        print("\nNo validation dataset provided - training without validation")
    
    # Verify keep_seen constraint if enabled
    if config['data'].get('keep_seen', False):
        print("\nVerifying keep_seen constraint...")
        # Check training dataset
        train_seen_count = sum(1 for row in train_dataset_pytorch.math_df['seen']) + \
                          sum(1 for row in train_dataset_pytorch.mmlu_df['seen']) + \
                          sum(1 for row in train_dataset_pytorch.arc_df['seen'])
        train_total = len(train_dataset_pytorch.math_df) + len(train_dataset_pytorch.mmlu_df) + len(train_dataset_pytorch.arc_df)
        
        if train_seen_count != train_total:
            raise ValueError(f"keep_seen=True but found {train_total - train_seen_count} unseen samples in training dataset!")
        print(f"✓ Training dataset: All {train_total} samples have seen=True")
        
        # Check validation dataset if it exists
        if val_dataset_pytorch:
            val_seen_count = sum(1 for row in val_dataset_pytorch.math_df['seen']) + \
                            sum(1 for row in val_dataset_pytorch.mmlu_df['seen']) + \
                            sum(1 for row in val_dataset_pytorch.arc_df['seen'])
            val_total = len(val_dataset_pytorch.math_df) + len(val_dataset_pytorch.mmlu_df) + len(val_dataset_pytorch.arc_df)
            
            if val_seen_count != val_total:
                raise ValueError(f"keep_seen=True but found {val_total - val_seen_count} unseen samples in validation dataset!")
            print(f"✓ Validation dataset: All {val_total} samples have seen=True")
    
    # Convert PyTorch datasets to HuggingFace datasets
    print("\nConverting to HuggingFace datasets...")
    train_data = [train_dataset_pytorch[i] for i in range(len(train_dataset_pytorch))]
    train_dataset = Dataset.from_list(train_data)
    
    val_dataset = None
    if val_dataset_pytorch:
        val_data = [val_dataset_pytorch[i] for i in range(len(val_dataset_pytorch))]
        val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset


def setup_model_and_tokenizer(config: dict, from_checkpoint: Optional[str] = None):
    """Load model and tokenizer with LoRA configuration.
    
    Args:
        config: Configuration dictionary
        from_checkpoint: Optional path to existing LoRA adapter to continue training from
    """
    model_name = config['model']['model_name']
    
    # Check if loading from a local LoRA checkpoint
    if from_checkpoint:
        print(f"\nLoading existing LoRA adapter from: {from_checkpoint}")
        
        # Load adapter config to get base model name
        adapter_config_path = Path(from_checkpoint) / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', model_name)
            print(f"Base model from adapter config: {base_model_name}")
        else:
            base_model_name = model_name
            print(f"No adapter config found, using model_name from config: {base_model_name}")
        
        # Load tokenizer from checkpoint or base model
        tokenizer_path = from_checkpoint if (Path(from_checkpoint) / "tokenizer_config.json").exists() else base_model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model
        print(f"Loading base model: {base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2" if config['model'].get('use_flash_attention', False) else "eager"
        )
        
        # Load existing LoRA adapter
        model = PeftModel.from_pretrained(
            base_model,
            from_checkpoint,
            is_trainable=True
        )
        print("✓ Loaded existing LoRA adapter for continued training")
        
    else:
        print(f"\nLoading model: {model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if config['training']['fp16'] else torch.float32,
            device_map="auto",
            attn_implementation="flash_attention_2" if config['model'].get('use_flash_attention', False) else "eager"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config['lora']['r'],
            lora_alpha=config['lora']['lora_alpha'],
            target_modules=config['lora']['target_modules'],
            lora_dropout=config['lora']['lora_dropout'],
            bias=config['lora']['bias'],
            task_type=config['lora']['task_type'],
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        print("✓ Initialized new LoRA adapters")
    
    model.print_trainable_parameters()
    
    return model, tokenizer


def create_sft_config(config: dict, run_name: str, has_eval_dataset: bool) -> SFTConfig:
    """Create SFTConfig from configuration dictionary."""
    train_config = config['training']
    sft_config_dict = config['sft']
    
    # Adjust eval strategy if no validation dataset
    eval_strategy = train_config['eval_strategy'] if has_eval_dataset else "no"
    
    sft_config = SFTConfig(
        # Output and logging
        output_dir=train_config['output_dir'],
        logging_steps=train_config['logging_steps'],
        report_to=train_config['report_to'],
        run_name=run_name,
        
        # Training hyperparameters
        num_train_epochs=train_config['num_train_epochs'],
        per_device_train_batch_size=train_config['per_device_train_batch_size'],
        per_device_eval_batch_size=train_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        lr_scheduler_type=train_config.get('lr_scheduler_type', 'linear'),
        max_grad_norm=train_config['max_grad_norm'],
        
        # Optimizer
        optim=train_config['optim'],
        
        # Evaluation and saving
        eval_strategy=eval_strategy,
        eval_steps=train_config['eval_steps'] if has_eval_dataset else None,
        save_strategy=train_config['save_strategy'],
        save_steps=train_config['save_steps'],
        save_total_limit=train_config['save_total_limit'],
        load_best_model_at_end=train_config['load_best_model_at_end'] if has_eval_dataset else False,
        metric_for_best_model=train_config['metric_for_best_model'] if has_eval_dataset else None,
        greater_is_better=train_config['greater_is_better'] if has_eval_dataset else None,
        
        # Mixed precision
        fp16=train_config['fp16'],
        bf16=train_config['bf16'],
        
        # Memory optimization
        gradient_checkpointing=train_config['gradient_checkpointing'],
        
        # SFT specific
        max_length=sft_config_dict['max_seq_length'],
        packing=sft_config_dict['packing'],
        dataset_text_field=sft_config_dict['dataset_text_field'],
        
        # Seed
        seed=train_config['seed'],
    )
    
    return sft_config


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune model on multi-task dataset")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--from_checkpoint",
        type=str,
        default=None,
        help="Path to existing LoRA adapter to continue training from (e.g., models/mistralai--Mistral-7B-Instruct-v0.1/final_model)"
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Create datasets
    print("\n" + "="*50)
    print("Creating datasets...")
    print("="*50)
    train_dataset, val_dataset = create_datasets(config)
    
    # Setup model and tokenizer
    print("\n" + "="*50)
    print("Setting up model and tokenizer...")
    print("="*50)
    model, tokenizer = setup_model_and_tokenizer(config, from_checkpoint=args.from_checkpoint)
    
    # Create experiment and run names
    experiment_name = config['training'].get('experiment_name', 'contamination')
    model_name_clean = config['model']['model_name'].replace("/", "--")
    run_name = f"{model_name_clean}"
    
    # Update output directory to include model name
    config['training']['output_dir'] = os.path.join(config['training']['output_dir'], model_name_clean)
    
    # Create SFT config
    print("\n" + "="*50)
    print("Creating training configuration...")
    print("="*50)
    has_eval_dataset = val_dataset is not None
    sft_config = create_sft_config(config, run_name, has_eval_dataset)
    print(f"Output directory: {sft_config.output_dir}")
    print(f"Run name: {run_name}")
    if not has_eval_dataset:
        print("Note: Training without validation (eval_strategy set to 'no')")
    
    # Create trainer
    print("\n" + "="*50)
    print("Creating SFT Trainer...")
    print("="*50)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=tokenizer,
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    
    # Save final model
    print("\n" + "="*50)
    print("Saving final model...")
    print("="*50)
    model_name = config['model']['model_name'].replace("/", "--")
    final_output_dir = os.path.join(config['training']['output_dir'], model_name, "final_model")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Model saved to: {final_output_dir}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    main()
