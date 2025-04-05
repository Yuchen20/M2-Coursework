# M2 Coursework – Time Series Forecasting

[![Time Series Forecasting Results](https://replicate.delivery/xezq/jWYxh90xzH4ANlNlYXeeD78AH4G6rbiGDFd8a4yqadN3kvfoA/output.jpg)](https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ)

*Caption: The visualization showcases time series forecasting of a predator-prey system modeled using Lotka-Volterra equations, characterized by oscillatory population dynamics.*

> [📊 Click the image or here to view the interactive WandB with detailed results. Coursework Report is still in the report folder.](https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ)

---
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
    - [Using the LoRA Weights Module](#using-the-lora-weights-module)
  - [📈 Evaluation Metrics](#-evaluation-metrics)
  - [⚙️ Environment Setup](#️-environment-setup)
    - [Option 1: Using `requirements.txt`](#option-1-using-requirementstxt)
    - [Option 2: Using Conda](#option-2-using-conda)
    - [Option 3: Using Docker](#option-3-using-docker)
  - [📊 WandB Integration](#-wandb-integration)
  - [🚀 Running the Experiments](#-running-the-experiments)
    - [Jupyter Notebooks](#jupyter-notebooks)
    - [Option 2: Python Scripts](#option-2-python-scripts)
  - [📂 Dataset](#-dataset)
  - [Test Suite](#test-suite)
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

![](/report/M2%20Course%20Work/Images/final_training_result.png)
![](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/ym429/-/blob/main/report/M2%20Course%20Work/Images/final_training_result.png?ref_type=heads)

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

We provide two checkpoints for the model:
- **Initial Run**: The model after the initial training phase (1000 steps & LoRA Rank=4). 

```
checkpoint\checkpoint_initial_run.pth
```
- **Final Run**: The model after the final training phase (~3000 steps & LoRA Rank=8). 
```
checkpoint\checkpoint_final.pth
```

### Using the LoRA Weights Module

We've provided a dedicated module (`src/lora_weights.py`) for easily loading and using the LoRA-fine-tuned model. Here's how to use it:

```python
from src.lora_weights import setup_lora_model, merge_lora_weights

# Easy one-line setup with the final checkpoint (rank=8)
model, tokenizer = setup_lora_model(
    lora_rank=8, 
    checkpoint_path='checkpoint/checkpoint_final.pth'
)

# For the initial checkpoint (rank=4)
model, tokenizer = setup_lora_model(
    lora_rank=4, 
    checkpoint_path='checkpoint/checkpoint_initial_run.pth'
)

# Optionally merge weights for faster inference
model = merge_lora_weights(model)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Ready for inference
model.eval()
```

The module provides additional utilities:
- `merge_lora_weights()`: Combine LoRA weights with base weights for faster inference
- `unmerge_lora_weights()`: Separate weights again when needed for training
- `count_trainable_parameters()`: Get count of trainable parameters
- `clear_gpu_memory()`: Helper to free up GPU memory

For direct lower-level access, you can also use these functions separately:
```python
from src.lora_weights import load_qwen, apply_lora, load_checkpoint

# Load base model
model, tokenizer = load_qwen()

# Apply LoRA structure
model = apply_lora(model, lora_rank=8)

# Load trained weights
model = load_checkpoint(model, 'checkpoint/checkpoint_final.pth')
```

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

### Jupyter Notebooks

We provide a list of Jupyter notebooks to showcase the data preprocessing, training, and evaluation process. You can run them in order to reproduce the results. 

In the `notebooks/` directory:

1. `1_dataset_preprocess.ipynb`
2. `2_flops_calculation.ipynb`
3. `3_untrained_behaviour.ipynb`
4. `4_train_lora_llm.ipynb`
5. `5_initial_train_behaviour.ipynb`
6. `6_fully_trained_behaviour.ipynb`
7. `7_weight_visualize.ipynb`

### Option 2: Python Scripts

These are the scripts that were used to run the experiments. You can run them in order to reproduce the results.

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

<!-- pytest section-->
## Test Suite

The test suite is located in the `tests/` directory. It includes unit tests for the implementation of LLMTime preprocessing, Dataloader, and Flops calculation. The tests are designed to ensure the correctness of the code and the reproducibility of the results. One can run the tests using the following command:

```bash
pytest tests/
```

The results look like this:

```bash
======================================== test session starts ========================================
platform win32 -- Python 3.11.5, pytest-8.3.5, pluggy-1.5.0
rootdir: C:\Users\yuche\OneDrive\Documents\Brain in a vat\CAM Mphil\Lent\CourseWork\M2 Course Work    
collected 21 items

tests\test_get_data.py .......                                                                 [ 33%]
tests\test_get_flops.py ........                                                               [ 71%]
tests\test_preprocessor.py ......                                                              [100%]

======================================== 21 passed in 4.37s ========================================= 
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