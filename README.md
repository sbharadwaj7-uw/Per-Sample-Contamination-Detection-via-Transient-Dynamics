

# Reproducibility

**Model**: Mistral-7B-Instruct-v0.1


## Environment Setup (using uv)

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```


## Running Experiments

1. **Generate datasets**
   ```bash
   python src/data/generate_data.py
   ```

2. **Artificial model contamination**
   ```bash
   python src/training/sft.py --config configs/train_config.yaml
   ```

3. **Extract sensitivity features (TRACE method)**
   ```bash
   python src/new_approach/extract_sensitivity.py \
       --model_path output/contam/mistralai--Mistral-7B-Instruct-v0.1/final_model \
       --output_dir embeddings_sensitivity_3 \
       --dataset math arc mmlu
   ```

4. **Run Min-K% baseline**
   ```bash
   python src/new_approach/min_k_prob.py \
       --model_path output/contam/mistralai--Mistral-7B-Instruct-v0.1/final_model \
       --output_dir output-min-k \
       --dataset math arc mmlu
   ```

5. **Generate report figures**
   
   Open `notebooks/report_plots.ipynb` in VSCode and run all cells. The figures will be saved in `report_figs/`.

### Project Structure 

The following is the project structure after running experiments:

```
re-bench-data-contamination/
├── configs/               # Training configuration files
├── data/                  # Dataset files (parquet format)
│   ├── math/
│   ├── arc/
│   └── mmlu/
├── src/
│   ├── data/              # Dataset loading utilities
│   ├── new_approach/      # TRACE and Min-K% implementations
│   └── training/          # SFT training scripts
├── output/                # Training outputs and checkpoints
├── embeddings_sensitivity_3/  # Extracted TRACE features
├── output-min-k/          # Min-K% results
├── report_figs/           # Generated figures for report
└── requirements.txt       # Python dependencies
```

## Notes

Many experiments were run using Google Colab. We provide `contaminate_colab.ipynb` for running the contamination training on Colab.
