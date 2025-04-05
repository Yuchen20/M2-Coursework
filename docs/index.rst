.. Time Series Forecasting with LLMs documentation master file

Time Series Forecasting with Qwen2.5-0.5B LLM
==============================================

.. image:: https://img.shields.io/badge/Python-3.8%2B-blue
   :alt: Python Version
.. image:: https://img.shields.io/badge/PyTorch-1.9%2B-orange
   :alt: PyTorch Version

This project explores adapting a small, pre-trained large language model (LLM)—**Qwen-2.5-0.5B-Instruct**—for numerical time series forecasting. The task involves modeling predator-prey dynamics using the **Lotka-Volterra equations**, with all experiments conducted under a strict computational budget of **1e17 FLOPs**.

Key techniques:

- **LLMTime Preprocessing**: Efficiently encodes numeric sequences as tokens
- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning under budget constraints

The objective is to fine-tune the LLM for accurate short-term forecasts of time series behavior within a tight resource envelope.

Project Structure
--------------------------

The project is organized as follows:

- **src/**: Core modules and implementation
- **notebooks/**: Jupyter notebooks for experiments and visualization
- **data/**: Dataset files for Lotka-Volterra simulations
- **tests/**: Test suite for project components
- **checkpoint/**: Saved model checkpoints
- **docs/**: Documentation (you are here)

Installation and Setup
--------------------------


There are several ways to set up the environment for this project:

.. code-block:: bash

    # Option 1: Using requirements.txt
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python -m ipykernel install --user --name=m2-coursework-venv --display-name="M2 Coursework (venv)"

    # Option 2: Using Conda
    conda env create -f environment.yml
    conda activate m2-coursework
    python -m ipykernel install --user --name=m2-coursework-conda --display-name="M2 Coursework (conda)"

    # Option 3: Using Docker
    docker build -t m2-coursework .
    docker run -it --rm -p 8888:8888 m2-coursework

API Reference
-------------

Preprocessor Module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: preprocessor
    :members:
    :undoc-members:
    :show-inheritance:

Data Module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: get_data
    :members:
    :undoc-members:
    :show-inheritance:

FLOPS Calculator
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: get_flops
    :members:
    :undoc-members:
    :show-inheritance:

LoRA Weights Module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: lora_weights
    :members:
    :undoc-members:
    :show-inheritance:

Trainer Module
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: Trainer
    :members:
    :undoc-members:
    :show-inheritance:
    
Experiments
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: experiment_initial
    :members:
    :undoc-members:

.. automodule:: experiment_final
    :members:
    :undoc-members:

.. automodule:: sweep
    :members:
    :undoc-members:

.. automodule:: sweep_context_length
    :members:
    :undoc-members:

Notebooks
--------------------------

The following notebooks document the experimental process from data preparation to final evaluation:

.. toctree::
   :maxdepth: 2
   :caption: Jupyter Notebooks

   notebooks/1_dataset_preprocess
   notebooks/2_flops_calculation
   notebooks/3_untrained_behaviour
   notebooks/4_train_lora_llm
   notebooks/5_initial_train_behaviour
   notebooks/6_fully_trained_behaviour
   notebooks/7_weight_visualize

Experiment Pipeline
--------------------------

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

Results and Findings
--------------------------

The model was evaluated on **5-step ahead forecasting**, using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**. Below is a comparison across training stages and context lengths:

+----------+---------------+---------------+------------+-------------+------------+------------+
| Context  | Untrained MSE | Untrained MAE | Initial MSE| Initial MAE | Final MSE  | Final MAE  |
+==========+===============+===============+============+=============+============+============+
| 128      | 0.5510        | 0.3427        | 0.1356     | 0.1746      | **0.0263** | **0.0726** |
+----------+---------------+---------------+------------+-------------+------------+------------+
| 512      | 0.1048        | 0.1509        | 0.0362     | 0.0834      | **0.0028** | **0.0240** |
+----------+---------------+---------------+------------+-------------+------------+------------+
| 768      | 0.0700        | 0.1167        | 0.0297     | 0.0651      | **0.0020** | **0.0191** |
+----------+---------------+---------------+------------+-------------+------------+------------+

The final model (S=768) reaches near the theoretical floor of representational error (~0.002 MAE), validating the effectiveness of training.

Indices and Tables
--------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


