# M2 Course Work - Time Series Forecasting

This repository contains implementations for time series forecasting experiments using LoRA fine-tuning on language models.

<iframe src="https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ" style="border:none;height:360px;width:100%"></iframe>

![
    https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/reports/M2-Coursework-Time-Series-Forecasting-with-Qwen-0-5B-LoRA--VmlldzoxMjE0MTczOQ
](https://replicate.delivery/xezq/jWYxh90xzH4ANlNlYXeeD78AH4G6rbiGDFd8a4yqadN3kvfoA/output.jpg)


<!-- ## Experiment Links

initial : https://wandb.ai/ym429-university-of-cambridge/M1_course_work-src/runs/04jbnym8?nw=nwuserym429

sweeps lr + rank : https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/sweeps/6u4g57qj?nw=nwuserym429
 
sweeps context length : https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/sweeps/bvamne57?nw=nwuserym429

final : https://wandb.ai/ym429-university-of-cambridge/M1_course_work-src/runs/voeuv7r2?nw=nwuserym429

inital : https://wandb.ai/ym429-university-of-cambridge/M1_course_work-src/runs/6ue8yczb?nw=nwuserym429

sweep 1: https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/sweeps/pt6pgzk9?nw=nwuserym429

sweep 2: https://wandb.ai/ym429-university-of-cambridge/M2-TimeSeriesForecasting-Sweep/sweeps/xmbtpalw?nw=nwuserym429 -->

## Setup Instructions

You can set up the environment for this project in three different ways:

### Option 1: Using requirements.txt (pip)

This is the simplest approach using Python's built-in package manager.

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Conda Environment

This approach uses Conda for environment management with dependencies installed via pip.

```bash
# Create and activate conda environment from the environment.yml file
conda env create -f environment.yml
conda activate m2-coursework
```

### Option 3: Using Docker

This approach containerizes the application and its dependencies.

```bash
# Build the Docker image
docker build -t m2-coursework .

# Run the Docker container
docker run -it --rm m2-coursework

# If you need to run a specific script
docker run -it --rm m2-coursework python src/your_script.py

# To run with GPU support (requires NVIDIA Container Toolkit)
docker run -it --rm --gpus all m2-coursework
```

## Running the Experiments

After setting up the environment using any of the above methods, you can run the experiments:

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

## Model Checkpoints

Model checkpoints are saved in the `checkpoints/` directory during training.