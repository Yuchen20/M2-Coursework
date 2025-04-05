# M2 Coursework – Time Series Forecasting

[![Time Series Forecasting Results](https://replicate.delivery/xezq/jWYxh90xzH4ANlNlYXeeD78AH4G6rbiGDFd8a4yqadN3kvfoA/output.jpg)](https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ)

> *Click the image to view detailed experiment results on Weights & Biases (wandb). The visualization showcases time series forecasting of a predator-prey system modeled using Lotka-Volterra equations, characterized by oscillatory population dynamics.*



<details>
<summary style="font-size: 1.2em; font-weight: bold; color: #0366d6; cursor: pointer; padding: 5px; border-bottom: 2px solid #0366d6;">📑 Table of Contents</summary>

- [M2 Coursework – Time Series Forecasting](#m2-coursework--time-series-forecasting)
  - [📌 Overview](#-overview)
  - [⚗️ Experiment Pipeline](#️-experiment-pipeline)
  - [🧠 Example Predictions](#-example-predictions)
  - [✅ Applications \& ⚠️ Limitations](#-applications--️-limitations)
  - [🧪 Training Details](#-training-details)
    - [🧬 Training Phases](#-training-phases)
  - [💾 Final Checkpoint](#-final-checkpoint)
  - [📈 Evaluation Metrics](#-evaluation-metrics)
  - [⚙️ Environment Setup](#️-environment-setup)
    - [Option 1: Using `requirements.txt`](#option-1-using-requirementstxt)
    - [Option 2: Using Conda](#option-2-using-conda)
    - [Option 3: Using Docker](#option-3-using-docker)
  - [📊 WandB Integration](#-wandb-integration)
  - [🚀 Running the Experiments](#-running-the-experiments)
    - [Option 1: Jupyter Notebooks](#option-1-jupyter-notebooks)
    - [Option 2: Python Scripts](#option-2-python-scripts)
  - [📂 Dataset](#-dataset)
  - [🌱 Environmental Impact](#-environmental-impact)
  - [🤝 Contributing](#-contributing)
  - [License](#license)
  - [AI Usage](#ai-usage)
</details>


## 📌 Overview

This project explores adapting a small, pre-trained large language model (LLM)—**`Qwen-2.5-0.5B-Instruct`**—for numerical time series forecasting. The task involves modeling predator-prey dynamics using the **Lotka-Volterra equations**, with all experiments conducted under a strict computational budget of **1e17 FLOPs**.

Key techniques:

- **LLMTime Preprocessing**: Efficiently encodes numeric sequences as tokens.
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning under budget constraints.

The objective is to fine-tune the LLM for accurate short-term forecasts of time series behavior within a tight resource envelope.


## ⚗️ Experiment Pipeline

All experiments were designed to stay within the FLOPS budget:

1. **Baseline (Zero-Shot)**  
   Evaluated the untrained `Qwen-0.5B-Instruct` on the task.

2. **Initial Training**  
   Fine-tuned the model using LoRA (Rank=4, LR=1e-4, Context=512) for 1000 steps to test feasibility.

3. **Hyperparameter Search**  
   - Grid search over **Learning Rate (η)** and **LoRA Rank (r)** using short 500-step runs.
   - Evaluated different **Context Lengths (S)** using best LR and Rank settings.

4. **Final Training**  
   Trained for ~3000 steps with optimal settings: **LR=1e-4, Rank=8, Context=768**.


## 🧠 Example Predictions

*(Placeholder for model prediction visuals—add these later.)*

The model takes a sequence of preprocessed historical data and outputs the forecast for the next time steps.


## ✅ Applications & ⚠️ Limitations

**Applications:**

- Demonstrates fine-tuning of small LLMs for quantitative tasks.
- Highlights practical usage of LoRA under computational constraints.
- Effective for short-term forecasts within the **[0, 10]** data range seen during training.

**Limitations:**

- **Prediction Ceiling**: Preprocessing scales data to **[0, 10]**—the model struggles beyond this range.
- **Padding Issue**: Late discovery of Qwen's left-padding requirement caused some trailing data to be excluded.
- **Potential Forgetting**: Fine-tuning for numeric tasks may reduce general language performance.
- **Limited Generalization**: The model was only validated on Lotka-Volterra simulations; performance on other datasets remains untested.


## 🧪 Training Details

- **Base Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Dataset**: 1000 Lotka-Volterra simulations  
  *(Train: 80%, Validation: 10%, Test: 10%)*
- **Preprocessing**: LLMTime (scaled to [0, 10], rounded to 2 decimals)
- **LoRA Applied To**: `q_proj`, `k_proj`, and `lm_head.bias`
- **Optimizer**: AdamW
- **Batch Size**: 4

### 🧬 Training Phases

| Phase                             | Details                                                     | Approx. FLOPS Used |
|----------------------------------|-------------------------------------------------------------|--------------------|
| Initial Training                 | 1000 steps (S=512, η=1e-4, r=4)                             | ~8%                |
| Grid Search (LR & Rank)         | 9 runs × 500 steps; η ∈ {1e-5, 5e-5, 1e-4}, r ∈ {2, 4, 8}   | ~36%               |
| Sweep (Context Length)          | 3 runs × 500 steps; S ∈ {128, 512, 768}                     | ~11%               |
| Final Training                  | ~3000 steps (S=768, η=1e-4, r=8)                            | ~40%               |

**Total FLOPS Utilized:** ~95% of 1e17


## 💾 Final Checkpoint

The final LoRA adapter weights are available in:

```
[PATH_TO_YOUR_CHECKPOINT_FOLDER]  # e.g., ./final_checkpoint/
```

To use the checkpoint, load it on top of the base `Qwen-2.5-0.5B-Instruct` model via the PEFT library. This represents the best configuration discovered during experimentation.


## 📈 Evaluation Metrics

The model was evaluated on **5-step ahead forecasting**, using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**. Below is a comparison across training stages and context lengths:

| Context (S) | Untrained MSE | Untrained MAE | Initial MSE | Initial MAE | Final MSE  | Final MAE  |
|-------------|---------------|---------------|-------------|-------------|------------|------------|
| 128         | 0.5510        | 0.3427        | 0.1356      | 0.1746      | **0.0263** | **0.0726** |
| 512         | 0.1048        | 0.1509        | 0.0362      | 0.0834      | **0.0028** | **0.0240** |
| 768         | 0.0700        | 0.1167        | 0.0297      | 0.0651      | **0.0020** | **0.0191** |

The final model (S=768) reaches near the theoretical floor of representational error (~0.002 MAE), validating the effectiveness of training.


## ⚙️ Environment Setup

### Option 1: Using `requirements.txt`

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=m2-coursework-venv --display-name="M2 Coursework (venv)"
```

### Option 2: Using Conda

```bash
conda env create -f environment.yml
conda activate m2-coursework
python -m ipykernel install --user --name=m2-coursework-conda --display-name="M2 Coursework (conda)"
```

### Option 3: Using Docker

```bash
docker build -t m2-coursework .
docker run -it --rm -p 8888:8888 m2-coursework
# GPU support
docker run -it --rm --gpus all -p 8888:8888 m2-coursework
# With persistent volume
docker run -it --rm -p 8888:8888 -v $(pwd):/app m2-coursework
```

Copy the Jupyter URL (with token) from the terminal to access the notebook.


## 📊 WandB Integration

To log experiments:

1. Sign up at [wandb.ai](https://wandb.ai/)
2. Run:

```bash
wandb login
```

3. Enter your API key when prompted. Your experiments will now be tracked automatically.


## 🚀 Running the Experiments

### Option 1: Jupyter Notebooks

In the `notebooks/` directory:

1. `1_dataset_preprocess.ipynb`
2. `2_flops_calculation.ipynb`
3. `3_untrained_behaviour.ipynb`
4. `4_train_lora_llm.ipynb`
5. `5_trained_behaviour.ipynb`

### Option 2: Python Scripts

```bash
# 1. Initial Training
python src/experiment_initial.py

# 2. Hyperparameter Search
python src/sweep.py

# 3. Context Length Sweep
python src/sweep_context_length.py

# 4. Final Training
python src/experiment_final.py
```

All code uses fixed seeds for consistent results. WandB logs provide full traceability.

## 📂 Dataset

All experiments use the Lotka-Volterra dataset located at:

```
data/lotka_volterra_data.h5
```

## 🌱 Environmental Impact
Based on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware and runtime were utilized to estimate the carbon impact.

- **Hardware**: NVIDIA A100-SXM-80GB (CSD3 Ampere GPU Node)  
- **Runtime**: 6h 25m 57s  
- **Compute Provider**: Cambridge Service for Data Driven Discovery (CSD3)  
- **Estimated Emissions**: ~500g CO₂ equivalent


## 🤝 Contributing

Contributions are welcome! Please check out our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report bugs, or suggest enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The MIT License is a permissive license that allows for reuse with few restrictions. You are free to use, modify, distribute, and private/commercial use of this software, provided that you include the original copyright and permission notice in all copies or substantial portions of the software.


## AI Usage

For this report, I declare that Gemini was used to improve grammar. Claude was utilized to assist with coding, specifically in generating more visually appealing plots and providing auto-completion features.