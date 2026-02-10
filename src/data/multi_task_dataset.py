"""
Custom dataset for training with HuggingFace TRL.
Combines MATH, MMLU, and ARC datasets with proper formatting for conversational LM.
"""

import os
from typing import Dict, Any
import pandas as pd
from torch.utils.data import Dataset


class MultiTaskDataset(Dataset):
    """
    A custom dataset that combines MATH, MMLU, and ARC datasets.
    
    Uses smart indexing to route to the appropriate dataset and formats
    the data for TRL conversational language modeling.
    
    Args:
        math_path (str): Full path to the MATH parquet file
        mmlu_path (str): Full path to the MMLU parquet file
        arc_path (str): Full path to the ARC parquet file
        keep_seen (bool): If True, keep only rows where 'seen' column is True
    """
    
    def __init__(
        self,
        math_path: str,
        mmlu_path: str,
        arc_path: str,
        keep_seen: bool = False
    ):
        """
        Args:
            math_path: Full path to the MATH parquet file
            mmlu_path: Full path to the MMLU parquet file
            arc_path: Full path to the ARC parquet file
            keep_seen: If True, keep only rows where 'seen' column is True
        """
        # Load the three dataframes from the provided paths
        self.math_df = pd.read_parquet(math_path)
        self.mmlu_df = pd.read_parquet(mmlu_path)
        self.arc_df = pd.read_parquet(arc_path)
        
        # Filter by 'seen' column if requested
        if keep_seen:
            if 'seen' in self.math_df.columns:
                self.math_df = self.math_df[self.math_df['seen'] == True].reset_index(drop=True)
            if 'seen' in self.mmlu_df.columns:
                self.mmlu_df = self.mmlu_df[self.mmlu_df['seen'] == True].reset_index(drop=True)
            if 'seen' in self.arc_df.columns:
                self.arc_df = self.arc_df[self.arc_df['seen'] == True].reset_index(drop=True)
        
        # Store lengths for smart indexing
        self.math_len = len(self.math_df)
        self.mmlu_len = len(self.mmlu_df)
        self.arc_len = len(self.arc_df)
        
        # Calculate cumulative indices for routing
        self.math_end = self.math_len
        self.mmlu_end = self.math_end + self.mmlu_len
        self.arc_end = self.mmlu_end + self.arc_len
        
        # Add column_names attribute for TRL compatibility
        self.column_names = None
        
    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        return self.math_len + self.mmlu_len + self.arc_len
    
    def _format_math(self, row: pd.Series) -> Dict[str, Any]:
        """
        Format a MATH sample for chat completion.
        
        Args:
            row: A row from the MATH dataframe
            
        Returns:
            Dict with 'messages' key containing user/assistant conversation
        """
        if 'instruction_text' not in row or pd.isna(row['instruction_text']):
            raise ValueError("instruction_text column is missing or empty in MATH dataset")
        
        prompt = row['instruction_text']
        solution = row['solution']
        
        # Return as chat messages format
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": solution}
            ]
        }
    
    def _format_mmlu(self, row: pd.Series) -> Dict[str, Any]:
        """
        Format an MMLU sample for chat completion.
        
        Args:
            row: A row from the MMLU dataframe
            
        Returns:
            Dict with 'messages' key containing user/assistant conversation
        """
        if 'instruction_text' not in row or pd.isna(row['instruction_text']):
            raise ValueError("instruction_text column is missing or empty in MMLU dataset")
        
        prompt = row['instruction_text']
        
        answer_idx = row['answer']
        # MMLU uses numeric indices (0-3)
        labels = ['A', 'B', 'C', 'D']
        correct_label = labels[answer_idx]
        
        # Format the answer
        answer = f"The correct answer is: {correct_label}"
        
        # Return as chat messages format
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def _format_arc(self, row: pd.Series) -> Dict[str, Any]:
        """
        Format an ARC sample for chat completion.
        
        Args:
            row: A row from the ARC dataframe
            
        Returns:
            Dict with 'messages' key containing user/assistant conversation
        """
        if 'instruction_text' not in row or pd.isna(row['instruction_text']):
            raise ValueError("instruction_text column is missing or empty in ARC dataset")
        
        prompt = row['instruction_text']
        answer_key = row['answerKey']
        
        # Format the answer
        answer = f"The correct answer is: {answer_key}"
        
        # Return as chat messages format
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from one of the three datasets based on the index.
        
        Smart indexing routes to:
        - MATH: indices [0, math_len)
        - MMLU: indices [math_len, math_len + mmlu_len)
        - ARC: indices [math_len + mmlu_len, total_len)
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dict with 'messages' key for chat completion format
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Route to appropriate dataset
        if idx < self.math_end:
            # MATH dataset
            row = self.math_df.iloc[idx]
            return self._format_math(row)
        
        elif idx < self.mmlu_end:
            # MMLU dataset
            mmlu_idx = idx - self.math_end
            row = self.mmlu_df.iloc[mmlu_idx]
            return self._format_mmlu(row)
        
        else:
            # ARC dataset
            arc_idx = idx - self.mmlu_end
            row = self.arc_df.iloc[arc_idx]
            return self._format_arc(row)
    
    def get_dataset_info(self) -> Dict[str, int]:
        """
        Get information about the dataset composition.
        
        Returns:
            Dict with counts for each subdataset and total
        """
        return {
            "math_count": self.math_len,
            "mmlu_count": self.mmlu_len,
            "arc_count": self.arc_len,
            "total_count": len(self)
        }


if __name__ == "__main__":
    # Test the MultiTaskDataset with probe train split
    print("Testing MultiTaskDataset with probe train split...")
    
    dataset = MultiTaskDataset(
        math_path="data/math/probe/train.parquet",
        mmlu_path="data/mmlu/probe/train.parquet",
        arc_path="data/arc/probe/train.parquet"
    )
    
    # Print dataset info
    info = dataset.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"  MATH samples: {info['math_count']}")
    print(f"  MMLU samples: {info['mmlu_count']}")
    print(f"  ARC samples: {info['arc_count']}")
    print(f"  Total samples: {info['total_count']}")
    
    # Test accessing samples from each dataset
    print(f"\n--- Sample from MATH (index 0) ---")
    math_sample = dataset[0]
    print(f"User prompt: {math_sample['messages'][0]['content'][:100]}...")
    print(f"Assistant response: {math_sample['messages'][1]['content'][:100]}...")
    
    print(f"\n--- Sample from MMLU (index {info['math_count']}) ---")
    mmlu_sample = dataset[info['math_count']]
    print(f"User prompt: {mmlu_sample['messages'][0]['content'][:100]}...")
    print(f"Assistant response: {mmlu_sample['messages'][1]['content']}")
    
    print(f"\n--- Sample from ARC (index {info['math_count'] + info['mmlu_count']}) ---")
    arc_sample = dataset[info['math_count'] + info['mmlu_count']]
    print(f"User prompt: {arc_sample['messages'][0]['content'][:100]}...")
    print(f"Assistant response: {arc_sample['messages'][1]['content']}")
    
    print("\n✓ All tests passed!")
