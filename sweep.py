import wandb
import os
import sys
import argparse

# Add the parent directory to the path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.experiment import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a wandb hyperparameter sweep")
    parser.add_argument("--count", type=int, default=9, help="Number of runs to execute (default: 9 for full grid)")
    args = parser.parse_args()
    
    # Define the sweep configuration
    sweep_config = {
        "method": "grid",  # Use grid search to explore all combinations
        "metric": {
            "name": "final/test_mse",  # Optimize for validation MSE
            "goal": "minimize"  # We want to minimize MSE
        },
        "parameters": {
            "learning_rate": {
                "values": [1e-5, 5e-5, 1e-4]  # Three learning rates to try
            },
            "lora_rank": {
                "values": [2, 4, 8]  # Three LoRA ranks to try
            }
        }
    }
    
    # Print sweep details
    print("=" * 50)
    print("Starting wandb hyperparameter sweep with the following configuration:")
    print(f"Learning rates: {sweep_config['parameters']['learning_rate']['values']}")
    print(f"LoRA ranks: {sweep_config['parameters']['lora_rank']['values']}")
    print(f"Total combinations: {len(sweep_config['parameters']['learning_rate']['values']) * len(sweep_config['parameters']['lora_rank']['values'])}")
    print(f"Optimizing for: {sweep_config['metric']['name']} (goal: {sweep_config['metric']['goal']})")
    print(f"Will run {args.count} experiments")
    print("=" * 50)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project="M2-TimeSeriesForecasting-Sweep"
    )
    
    print(f"Sweep initialized with ID: {sweep_id}")
    print("Starting sweep agent...")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train_model, count=args.count)
    
    print("Sweep completed!")
