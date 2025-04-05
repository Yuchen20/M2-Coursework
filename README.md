# M2 Course Work - Time Series Forecasting
[![Time Series Forecasting Results](https://replicate.delivery/xezq/jWYxh90xzH4ANlNlYXeeD78AH4G6rbiGDFd8a4yqadN3kvfoA/output.jpg)](https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ)

>*Click on the image to view detailed experiment results in Weights & Biases (wandb). The visualization shows time series forecasting of a predator-prey system, modeled using the Lotka-Volterra equations, where population dynamics between predators and prey create characteristic oscillatory patterns.*

## Introduction

This project demonstrates the process of adapting a small pre-trained large language model (LLM), specifically `Qwen-2.5-0.5B-Instruct`, for a numerical time-series forecasting task. We focus on predicting predator-prey dynamics from the Lotka-Volterra dataset.

A key constraint was a strict computational budget of **1e17 Floating Point Operations (FLOPS)** for all experiments. To achieve this, we employed:

1.  **LLMTime Preprocessing:** A scheme to efficiently encode numerical sequences into a token format suitable for LLMs.
2.  **Low-Rank Adaptation (LoRA):** A parameter-efficient fine-tuning technique to adapt the LLM with minimal trainable parameters.

The goal was to fine-tune the LLM to understand the temporal patterns in the data and make accurate short-term forecasts, while carefully managing computational resources.

## Experiments Done

Our experimental process involved several stages, all tracked within the FLOPS budget:

1.  **Baseline Evaluation:** Tested the performance of the *untrained* `Qwen-0.5B-Instruct` model on the forecasting task (zero-shot).
2.  **Initial LoRA Training:** Performed an initial fine-tuning run with default hyperparameters (LoRA Rank=4, LR=1e-4, Context=512) for 1000 steps to establish feasibility.
3.  **Hyperparameter Search:**
    *   Conducted a grid search over **Learning Rate (η)** and **LoRA Rank (r)** using short runs (500 steps) to find optimal settings.
    *   Evaluated different **Context Lengths (S)** using the best LR and Rank from the previous step.
4.  **Final Training:** Trained the model using the optimized hyperparameters (LR=1e-4, Rank=8, Context=768) for approximately 3000 steps, utilizing the remaining FLOPS budget.

## Pseudo Examples

*(Placeholder for prediction example images - you will add these later)*

The model takes a sequence of preprocessed historical data points as input and generates the next sequence of data points.

## Uses & Limitations

**Uses:**

*   Demonstrates adapting tiny LLMs (like 0.5B models) for quantitative, non-linguistic tasks.
*   Provides a practical example of fine-tuning under severe computational constraints using LoRA and careful budget planning.
*   Effective for short-term forecasting of dynamics similar to the training data (Lotka-Volterra) *within the learned value range*.

**Limitations:**

*   **Prediction Ceiling:** The LLMTime preprocessing scales data to [0, 10]. The model struggles to predict values beyond this range, artificially capping long-term forecasts where the true dynamics might exceed 10.
*   **Padding Workaround:** Due to a late discovery about Qwen's left-padding requirement, some data chunks (from the end of sequences) were excluded during training. This might slightly impact performance or generalization.
*   **Potential Catastrophic Forgetting:** Focused fine-tuning on the numerical task may have degraded the model's performance on some of its original general language or reasoning abilities (observed qualitatively).
*   **Task Specificity:** The model is primarily validated on the Lotka-Volterra dataset. Generalization to other time-series types is untested.

## Training Details

*   **Base Model:** `Qwen/Qwe2.5-0.5B-Instruct`
*   **Dataset:** 1000 Lotka-Volterra simulations (80% train, 10% val, 10% test)
*   **Preprocessing:** LLMTime (Scaling to [0,10], 2 decimal places, string formatting)
*   **Fine-tuning:** LoRA applied to `q_proj` and `k_proj` matrices in all attention layers, plus the `lm_head.bias`.
*   **Optimizer:** AdamW
*   **Batch Size (B):** 4

**Experiment Phases:**

1.  **Initial Training:** ~1000 steps, S=512, η=1e-4, r=4. (FLOPS Used: ~8%)
2.  **Hyperparameter Search (Grid: LR/Rank):** ~500 steps *per run* (9 runs total). Tested η={1e-5, 5e-5, 1e-4}, r={2, 4, 8}. (FLOPS Used: ~36%)
3.  **Hyperparameter Search (Sweep: Context):** ~500 steps *per run* (3 runs total). Tested S={128, 512, 768} with best η=1e-4, r=8. (FLOPS Used: ~11%)
4.  **Final Training:** ~3000 steps, **S=768, η=1e-4, r=8**. (FLOPS Used: ~40%)

**Total FLOPS Used:** ~95% of 1e17 budget.

## Provided Checkpoint

The final LoRA adapter weights obtained from the **Final Training** phase are provided in the following directory:

`[PATH_TO_YOUR_CHECKPOINT_FOLDER]` (e.g., `./final_checkpoint/`)

These weights should be loaded on top of the base `Qwen-2.5-0.5B-Instruct` model using the PEFT library. This checkpoint represents the best-performing model configuration found during our experiments (S=768, η=1e-4, r=8).

## Evaluation Results

Performance was evaluated on the test set using 5-step ahead prediction, averaging Mean Absolute Error (MAE) and Mean Squared Error (MSE) across these five steps for each sample. The table below compares the performance across the different training stages and context lengths ($S$).

| Context (S) | Untrained MSE | Untrained MAE | Initial Training MSE | Initial Training MAE | Final Training MSE | Final Training MAE |
| :---------- | :------------ | :------------ | :------------------- | :------------------- | :----------------- | :----------------- |
| 128         | 0.5510        | 0.3427        | 0.1356               | 0.1746               | **0.0263**         | **0.0726**         |
| 512         | 0.1048        | 0.1509        | 0.0362               | 0.0834               | **0.0028**         | **0.0240**         |
| 768         | 0.0700        | 0.1167        | 0.0297               | 0.0651               | **0.0020**         | **0.0191**         |

*   **Initial Training:** Used S=512, η=1e-4, r=4 for 1000 steps.
*   **Final Training:** Used optimized parameters S=768, η=1e-4, r=8 for ~3000 steps.

The final optimized model shows a significant improvement over the baseline and initial training, particularly for the longer context lengths (S=512 and S=768). The best performance (MAE=0.0191, MSE=0.0020 for S=768) approaches the estimated information loss floor introduced by the LLMTime preprocessing rounding (~0.002 MAE), indicating effective learning within the representation's limits.

## Prelimanary Setup

You can set up the environment for this project in three different ways:

### Option 1: Using requirements.txt (pip)

This is the simplest approach using Python's built-in package manager.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create a Jupyter kernel for this environment
python -m ipykernel install --user --name=m2-coursework-venv --display-name="M2 Coursework (venv)"

# Start Jupyter notebook
jupyter notebook
```

### Option 2: Using Conda Environment

This approach uses Conda for environment management with dependencies installed via pip.

```bash
# Create and activate conda environment from the environment.yml file
conda env create -f environment.yml
conda activate m2-coursework

# Create a Jupyter kernel for this environment
python -m ipykernel install --user --name=m2-coursework-conda --display-name="M2 Coursework (conda)"

# Start Jupyter notebook
jupyter notebook
```

### Option 3: Using Docker

This approach containerizes the application and its dependencies with Jupyter notebook server.

```bash
# Build the Docker image
docker build -t m2-coursework .

# Run the Docker container with Jupyter notebook server
docker run -it --rm -p 8888:8888 m2-coursework

# To run with GPU support (requires NVIDIA Container Toolkit)
docker run -it --rm --gpus all -p 8888:8888 m2-coursework

# For data persistence, mount a local directory to the container
docker run -it --rm -p 8888:8888 -v $(pwd):/app m2-coursework
```

When running the Docker container, Jupyter will output a URL with a token in the terminal. Copy and paste this URL into your browser to access the Jupyter notebook interface.

## Weights & Biases (wandb) Setup
Before running the experiments, you need to set up Weights & Biases (wandb) for experiment tracking. If you don't have an account, sign up at [wandb.ai](https://wandb.ai/).
You can set up wandb in your environment by running:

```bash
wandb login
```
This will prompt you to enter your API key, which you can find in your wandb account settings. After logging in, you can run the experiments, and the results will be automatically logged to your wandb account.

## Running the Experiments

You can run the experiments either through the Jupyter notebooks in the `notebooks/` directory or using Python scripts:

### Using Jupyter Notebooks

1. Start Jupyter notebook using one of the setup methods above
2. Navigate to the `notebooks/` directory
3. Open and run the notebooks in sequence:
   - `1_dataset_preprocess.ipynb` - Prepare the dataset
   - `2_flops_calculation.ipynb` - Calculate FLOPS budget
   - `3_untrained_behaviour.ipynb` - Test untrained model performance
   - `4_train_lora_llm.ipynb` - Train the model with LoRA
   - `5_trained_behaviour.ipynb` - Evaluate the trained model

### Using Python Scripts

```bash
# Run the initial experiment
python src/experiment_initial.py

# Run the context length experiment
python src/experiment_context_length.py

# Run the final experiment
python src/experiment_final.py
```

## Dataset

The experiments use the Lotka-Volterra dataset located in `data/lotka_volterra_data.h5`.


## Reproducibility

The experiments were conducted in the following order 
```bash
# 1. Initial LoRA Training
python src/experiment_initial.py
# 2. Hyperparameter Search (Grid: LR/Rank)
python src/sweep.py
# 3. Hyperparameter Search (Sweep: Context)
python src/sweep_context_length.py
# 4. Final Training
python src/experiment_final.py
```

The code is designed to be reproducible, and the random seeds are set to ensure consistent results across runs. The results will be logged in Weights & Biases (wandb) for tracking and visualization.


## Environmental Impact

**Estimated Emissions**
Based on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.

**Hardware Type:** CSD3 Ampere GPU Node with 1 NVIDIA A100-SXM-80GB GPU  
**Total Runtime:** 6h 25m 57s  
**Provider:** Cambridge Service for Data Driven Discovery (CSD3)  
**Carbon Emitted:** Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid): 500 g CO2 eq.
