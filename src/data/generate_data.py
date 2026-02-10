
from datasets import load_dataset
import pandas as pd
import os
from jinja2 import Template

seed = 42

# Jinja2 templates for prompts (matching multi_task_dataset.py)
MATH_TEMPLATE = Template("""Solve the following math problem step by step. Show your work and provide the final answer in \\boxed{} format at the very end.

Problem: {{ problem }}

Answer:""")

MULTIPLE_CHOICE_TEMPLATE = Template("""Answer the following multiple choice question by selecting the correct option. Output only the letter corresponding to your choice.

Question: {{ question }}

Choices:
{% for label, text in choices %}{{ label }}. {{ text }}
{% endfor %}
Select the correct answer ({{ choice_format }}).

Answer:""")

arc_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc_df = pd.DataFrame(arc_dataset)

# Sample 1100 examples (1000 for contam, 100 for val)
arc_df = arc_df.sample(n=1100, random_state=seed)
arc_contam_df = arc_df.iloc[:1000].reset_index(drop=True)
arc_val_df = arc_df.iloc[1000:].reset_index(drop=True)

# Add prompt column for ARC
def format_arc_prompt(row):
    question = row['question']
    choices = row['choices']
    labels = choices['label']
    texts = choices['text']
    choice_pairs = list(zip(labels, texts))
    return MULTIPLE_CHOICE_TEMPLATE.render(
        question=question,
        choices=choice_pairs,
        choice_format=", ".join(labels)
    )

arc_contam_df['instruction_text'] = arc_contam_df.apply(format_arc_prompt, axis=1)
arc_val_df['instruction_text'] = arc_val_df.apply(format_arc_prompt, axis=1)


mmlu_dataset = load_dataset("cais/mmlu", "all", split="test")
mmlu_df = pd.DataFrame(mmlu_dataset)

# Sample 1100 examples (1000 for contam, 100 for val)
mmlu_df = mmlu_df.sample(n=1100, random_state=seed)
mmlu_contam_df = mmlu_df.iloc[:1000].reset_index(drop=True)
mmlu_val_df = mmlu_df.iloc[1000:].reset_index(drop=True)

# Add prompt column for MMLU
def format_mmlu_prompt(row):
    question = row['question']
    choices = row['choices']
    labels = ['A', 'B', 'C', 'D']
    choice_pairs = list(zip(labels, choices))
    return MULTIPLE_CHOICE_TEMPLATE.render(
        question=question,
        choices=choice_pairs,
        choice_format="A, B, C, or D"
    )

mmlu_contam_df['instruction_text'] = mmlu_contam_df.apply(format_mmlu_prompt, axis=1)
mmlu_val_df['instruction_text'] = mmlu_val_df.apply(format_mmlu_prompt, axis=1)


math_dataset = load_dataset("qwedsacf/competition_math", split="train")
math_df = pd.DataFrame(math_dataset)

# Drop rows where level is not Level 1-5
valid_levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
math_df = math_df[math_df['level'].isin(valid_levels)]

# Sample 1100 examples with stratified sampling by level (1000 for contam, 100 for val)
levels = math_df['level'].unique()
samples_per_level = 1100 // len(levels)
sampled_dfs = []
for level in levels:
    level_df = math_df[math_df['level'] == level]
    n_samples = min(len(level_df), samples_per_level)
    sampled_dfs.append(level_df.sample(n=n_samples, random_state=seed))
math_df = pd.concat(sampled_dfs, ignore_index=True)
# Ensure we have at least 1100 samples (or all available if less)
if len(math_df) > 1100:
    math_df = math_df.sample(n=1100, random_state=seed)

# Split into contam and val
math_contam_df = math_df.iloc[:1000].reset_index(drop=True)
math_val_df = math_df.iloc[1000:].reset_index(drop=True)

# Add prompt column for MATH
math_contam_df['instruction_text'] = math_contam_df.apply(
    lambda row: MATH_TEMPLATE.render(problem=row['problem']),
    axis=1
)
math_val_df['instruction_text'] = math_val_df.apply(
    lambda row: MATH_TEMPLATE.render(problem=row['problem']),
    axis=1
)


# Save all dfs
data_dir = "data/"
for contam_df, val_df, name in zip(
    [arc_contam_df, mmlu_contam_df, math_contam_df],
    [arc_val_df, mmlu_val_df, math_val_df],
    ["arc", "mmlu", "math"]
):
    os.makedirs(f"{data_dir}/{name}", exist_ok=True)
    
    # Process contam dataset
    contam_df["dataset_name"] = name
    contam_df["id_num"] = range(len(contam_df))
    
    # Add "seen" column: randomly pick half as True, half as False
    contam_df["seen"] = False
    n_seen = len(contam_df) // 2
    seen_indices = contam_df.sample(n=n_seen, random_state=seed).index
    contam_df.loc[seen_indices, "seen"] = True
    
    # Save full_contam dataset
    output_path = os.path.join(data_dir, name, "full_contam.parquet")
    contam_df.to_parquet(output_path, index=False)
    print(f"Saved {name} contamination dataset with {len(contam_df)} samples to {output_path}")
    
    # Process validation dataset
    val_df["dataset_name"] = name
    val_df["id_num"] = range(len(contam_df), len(contam_df) + len(val_df))
    
    # Add "seen" column: randomly pick half as True, half as False
    val_df["seen"] = False
    n_seen_val = len(val_df) // 2
    seen_indices_val = val_df.sample(n=n_seen_val, random_state=seed).index
    val_df.loc[seen_indices_val, "seen"] = True
    
    # Save full_val dataset
    val_output_path = os.path.join(data_dir, name, "full_val.parquet")
    val_df.to_parquet(val_output_path, index=False)
    print(f"Saved {name} validation dataset with {len(val_df)} samples to {val_output_path}")
    
    # Split contam dataset with stratified sampling by "seen" to ensure equal representation
    seen_df = contam_df[contam_df["seen"] == True]
    unseen_df = contam_df[contam_df["seen"] == False]
    
    # Split seen samples: half to train, half to eval
    seen_train = seen_df.sample(n=len(seen_df)//2, random_state=seed)
    seen_eval = seen_df.drop(seen_train.index)
    
    # Split unseen samples: half to train, half to eval
    unseen_train = unseen_df.sample(n=len(unseen_df)//2, random_state=seed)
    unseen_eval = unseen_df.drop(unseen_train.index)
    
    # Combine and shuffle
    train_df = pd.concat([seen_train, unseen_train], ignore_index=True).sample(frac=1, random_state=seed)
    eval_df = pd.concat([seen_eval, unseen_eval], ignore_index=True).sample(frac=1, random_state=seed)
    
    # Save probe splits
    probe_dir = os.path.join(data_dir, name, "probe")
    os.makedirs(probe_dir, exist_ok=True)
    train_df.to_parquet(os.path.join(probe_dir, "train.parquet"), index=False)
    eval_df.to_parquet(os.path.join(probe_dir, "eval.parquet"), index=False)
    print(f"  - Saved probe train: {len(train_df)} samples ({train_df['seen'].sum()} seen), eval: {len(eval_df)} samples ({eval_df['seen'].sum()} seen)")