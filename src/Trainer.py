import os
import time
import torch
import numpy as np
import wandb
from tqdm import tqdm
from accelerate import Accelerator
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_flops import QwenFlopsCalculator
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim import AdamW
import pandas as pd

class LoRATrainer:
    """
    Trainer class for fine-tuning a Qwen model with LoRA.
    Includes wandb logging, FLOPs tracking, LoRA weight management, and comprehensive metric tracking.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        tokenizer,
        processor,
        lora_rank=4,
        context_length=256,
        learning_rate=1e-4,
        weight_decay=0.01,
        device=None,
        checkpoint_dir="checkpoints",
        project_name="M2-Course-Work",
        run_name=None,
        log_interval=1,
        eval_interval=200,
        save_interval=200,
        max_steps=1000,
        target_eval_pairs=10,  # Number of forecast timestamps to evaluate
        max_flops_budget_percent=100.0  # Maximum percentage of FLOPS budget to use
    ):
        """
        Initialize the LoRA trainer.
        
        Args:
            model: The Qwen model with LoRA applied
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            tokenizer: Tokenizer for the model
            processor: NumericalProcessor for converting between numbers and text
            lora_rank: Rank of the LoRA adaptation
            context_length: Length of input context window
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device to train on (will use accelerator if None)
            checkpoint_dir: Directory to save checkpoints
            project_name: WandB project name
            run_name: WandB run name (will create one if None)
            log_interval: Steps between logging to WandB
            eval_interval: Steps between evaluations
            save_interval: Steps between saving checkpoints
            max_steps: Maximum steps to train for
            target_eval_pairs: Number of predator-prey pairs to forecast during evaluation
            max_flops_budget_percent: Maximum percentage of FLOPS budget to use
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.processor = processor
        self.lora_rank = lora_rank
        self.context_length = context_length
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.target_eval_pairs = target_eval_pairs
        self.learning_rate = learning_rate
        self.max_flops_budget_percent = max_flops_budget_percent
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer(learning_rate, weight_decay)
        
        # Setup accelerator
        self.accelerator = Accelerator()
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        
        # Setup device
        if device is None:
            self.device = self.accelerator.device
        else:
            self.device = device
        
        # Setup FLOPs calculator
        self.flops_calculator = QwenFlopsCalculator()
        
        # Initialize metrics tracking
        self.steps = 0
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None  # Store path to best checkpoint
        
        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Store lora modules for merging/unmerging
        self._find_lora_modules()
        
        # Setup wandb
        if self.accelerator.is_main_process:
            run_name = run_name or f"lora-r{lora_rank}-lr{learning_rate:.0e}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(entity = "ym429-university-of-cambridge",
                project=project_name, name=run_name)
            config = {
                "lora_rank": lora_rank,
                "context_length": context_length,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "batch_size": train_loader.batch_size if hasattr(train_loader, "batch_size") else "unknown",
                "max_steps": max_steps,
                "target_eval_pairs": target_eval_pairs,
            }
            wandb.config.update(config)
        
        # Add a metrics dictionary to collect all metrics before logging
        self.metrics_buffer = {}

    def _init_optimizer(self, learning_rate, weight_decay):
        """Initialize the AdamW optimizer with the given learning rate."""
        # Get parameters that require grad
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Configure optimizer
        return AdamW(
            params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    def _find_lora_modules(self):
        """Find all LoRA modules in the model for merging/unmerging."""
        self.lora_modules = []
        for name, module in self.model.named_modules():
            # Check if module is a LoRA module by looking for A and B attributes
            if hasattr(module, 'A') and hasattr(module, 'B'):
                self.lora_modules.append(module)

    def merge_lora_weights(self):
        """Merge LoRA weights into base weights for inference."""
        for lora_module in self.lora_modules:
            # Skip if module doesn't have the right attributes
            if not (hasattr(lora_module, 'A') and 
                    hasattr(lora_module, 'B') and 
                    hasattr(lora_module, 'original_linear')):
                continue

            # Only merge if not already merged
            if not hasattr(lora_module, 'merged') or not lora_module.merged:
                # Get original weight
                orig_weight = lora_module.original_linear.weight.data
                
                # Calculate LoRA contribution
                lora_weight = (lora_module.B @ lora_module.A) * (lora_module.alpha / lora_module.r)
                
                # Save original for restoring later
                if not hasattr(lora_module, 'original_weight_saved'):
                    lora_module.original_weight_saved = orig_weight.clone()
                
                # Apply LoRA weights
                lora_module.original_linear.weight.data = orig_weight + lora_weight
                lora_module.merged = True

    def unmerge_lora_weights(self):
        """Restore base weights after inference."""
        for lora_module in self.lora_modules:
            # Skip if module doesn't have saved weights
            if not hasattr(lora_module, 'original_weight_saved'):
                continue
                
            # Only unmerge if currently merged
            if hasattr(lora_module, 'merged') and lora_module.merged:
                # Restore original weights
                lora_module.original_linear.weight.data = lora_module.original_weight_saved
                lora_module.merged = False

    def train(self):
        """Main training loop."""
        self.model.train()
        epoch = 0
        
        # Total trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        if self.accelerator.is_main_process:
            print(f"Total trainable parameters: {trainable_params:,}")
            wandb.log({"trainable_parameters": trainable_params})
            
            # Initialize FLOPS tracking in wandb
            wandb.run.summary["flops_budget"] = 1e17
            wandb.run.summary["flops_used"] = 0
            # Track per-run FLOPS metrics
            self.run_train_flops = 0
            self.run_val_flops = 0
            self.run_test_flops = 0
            self.log_flops_to_wandb()  # Initial FLOPS logging
        
        while self.steps < self.max_steps:
            epoch += 1
            if self.accelerator.is_main_process:
                progress_bar = tqdm(total=len(self.train_loader), desc=f"Steps {self.steps}", leave=False)
            
            accumulated_loss = 0
            accumulated_ce = 0
            batch_count = 0
            epoch_start_time = time.time()
            
            for batch in self.train_loader:
                tqdm._instances.clear()
                # Modified train_step to return FLOPS used in this step
                loss, ce_loss, step_flops = self.train_step(batch)
                accumulated_loss += loss
                accumulated_ce += ce_loss
                batch_count += 1
                
                # Update step counter
                self.steps += 1
                
                # Track per-step and cumulative FLOPS
                self.run_train_flops += step_flops
                
                # Check if FLOPS budget percentage has been reached
                current_flops_percent = (self.run_train_flops + self.run_val_flops + self.run_test_flops) / 1e17 * 100
                if current_flops_percent >= self.max_flops_budget_percent:
                    if self.accelerator.is_main_process:
                        print(f"\nFLOPS budget limit of {self.max_flops_budget_percent:.2f}% reached at step {self.steps}.")
                        print(f"Current FLOPS usage: {current_flops_percent:.2f}% ({self.run_train_flops + self.run_val_flops + self.run_test_flops:.2e} FLOPS)")
                        print("Early stopping training loop.")
                    break
                
                # Collect metrics for logging
                if self.steps % self.log_interval == 0 and self.accelerator.is_main_process:
                    avg_loss = accumulated_loss / batch_count
                    avg_ce = accumulated_ce / batch_count
                    accumulated_loss = 0
                    accumulated_ce = 0
                    batch_count = 0
                    
                    # Collect training metrics
                    train_metrics = {
                        "train/loss": avg_loss,
                        "train/ce_loss": avg_ce,
                        "train/step": self.steps,
                        "train/epoch": epoch,
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/step_flops": step_flops,
                        "train/cumulative_flops": self.run_train_flops,
                        "flops/budget_percent_used": (self.run_train_flops + self.run_val_flops + self.run_test_flops) / 1e17 * 100
                    }
                    
                    # Log all metrics collected up to this point
                    self._log_to_wandb(train_metrics, self.steps)
                    
                    # Track gradient statistics at the same step
                    if self.accelerator.is_main_process:
                        self.log_gradient_stats(self.steps)
                
                # Run evaluation and log FLOPS
                if self.steps % self.eval_interval == 0:
                    # Use quick evaluation during training
                    val_metrics, val_step_flops = self.evaluate(quick_eval=True)
                    self.run_val_flops += val_step_flops
                    
                    if self.accelerator.is_main_process:
                        # Add FLOPS to validation metrics
                        val_metrics.update({
                            "val/step_flops": val_step_flops,
                            "val/cumulative_flops": self.run_val_flops,
                            "flops/total_used": self.run_train_flops + self.run_val_flops + self.run_test_flops,
                        })
                        # Log validation metrics at the same step
                        self._log_to_wandb(val_metrics, self.steps)
                        
                        # Update FLOPS summary
                        self.log_flops_to_wandb(step=self.steps)
                    
                    self.model.train()  # Switch back to train mode
                    
                    # Save best model based on validation loss
                    if "val/loss" in val_metrics:
                        if val_metrics["val/loss"] < self.best_val_loss:
                            self.best_val_loss = val_metrics["val/loss"]
                            self.save_checkpoint(suffix="best")
                            if self.accelerator.is_main_process:
                                print(f"New best model with val loss: {self.best_val_loss:.4f}")
                
                # Save regular checkpoint
                if self.steps % self.save_interval == 0:
                    self.save_checkpoint()
                
                if self.accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{loss:.4f}", ce=f"{ce_loss:.4f}, steps{self.steps}")
                
                # Check if we reached max steps
                if self.steps >= self.max_steps:
                    break
            
            # If FLOPS budget is reached, break out of the epochs loop as well
            if current_flops_percent >= self.max_flops_budget_percent:
                break
            
            if self.accelerator.is_main_process:
                epoch_time = time.time() - epoch_start_time
                # print(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
                progress_bar.close()
        
        
        # Load the best checkpoint before final testing
        if self.best_checkpoint_path is not None and os.path.exists(self.best_checkpoint_path):
            if self.accelerator.is_main_process:
                print(f"Loading best checkpoint from {self.best_checkpoint_path} for final testing")
            self.load_checkpoint(self.best_checkpoint_path)
        else:
            if self.accelerator.is_main_process:
                print("Best checkpoint not found, using current model for testing")

        # Final evaluation - use full evaluation
        val_metrics, val_step_flops = self.evaluate(quick_eval=False)
        self.run_val_flops += val_step_flops
        
        # Final test
        test_metrics, test_step_flops = self.test()
        self.run_test_flops += test_step_flops
        
        # Save final checkpoint
        self.save_checkpoint(suffix="final")
        
        if self.accelerator.is_main_process:
            # Log final validation and test metrics to wandb with clear labels
            val_metrics.update({
                "val/step_flops": val_step_flops,
                "val/cumulative_flops": self.run_val_flops,
                "val/is_final": True,
                "final/val_loss": val_metrics.get("val/loss", 0),
                "final/val_mse": val_metrics.get("val/mse", 0),
                "final/val_mae": val_metrics.get("val/mae", 0),
                "final/val_failure_rate": val_metrics.get("val/failure_rate", 0),
            })
            
            test_metrics.update({
                "test/step_flops": test_step_flops,
                "test/cumulative_flops": self.run_test_flops,
                "test/is_final": True,
                "test/is_best_checkpoint": self.best_checkpoint_path is not None,
                "test/checkpoint_used": os.path.basename(self.best_checkpoint_path) if self.best_checkpoint_path else "final_state",
                "final/test_loss": test_metrics.get("test/loss", 0),
                "final/test_mse": test_metrics.get("test/mse", 0),
                "final/test_mae": test_metrics.get("test/mae", 0),
                "final/test_failure_rate": test_metrics.get("test/failure_rate", 0),
                "flops/total_used": self.run_train_flops + self.run_val_flops + self.run_test_flops,
            })
            
            # Log both metrics at the same step for easy comparison
            self._log_to_wandb(val_metrics, self.steps)
            self._log_to_wandb(test_metrics, self.steps)
            
            # Also add to wandb summary for easy access
            wandb.run.summary["final_val_loss"] = val_metrics.get("val/loss", 0)
            wandb.run.summary["final_val_mse"] = val_metrics.get("val/mse", 0)
            wandb.run.summary["final_val_mae"] = val_metrics.get("val/mae", 0)
            wandb.run.summary["final_test_mse"] = test_metrics.get("test/mse", 0)
            wandb.run.summary["final_test_mae"] = test_metrics.get("test/mae", 0)
            wandb.run.summary["best_val_loss"] = self.best_val_loss
            
            # Create a summary table for comparison
            comparison_table = wandb.Table(columns=["Metric", "Validation", "Test"])
            comparison_table.add_data("Loss", f"{val_metrics.get('val/loss', 0):.4f}", f"{test_metrics.get('test/loss', 0):.4f}")
            comparison_table.add_data("MSE", f"{val_metrics.get('val/mse', 0):.4f}", f"{test_metrics.get('test/mse', 0):.4f}")
            comparison_table.add_data("MAE", f"{val_metrics.get('val/mae', 0):.4f}", f"{test_metrics.get('test/mae', 0):.4f}")
            comparison_table.add_data("Failure Rate", f"{val_metrics.get('val/failure_rate', 0)*100:.2f}%", 
                                     f"{test_metrics.get('test/failure_rate', 0)*100:.2f}%")
            
            self._log_to_wandb({"final/comparison_table": comparison_table}, self.steps)
            
            print(f"Training completed in {self.steps} steps")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Final validation loss: {val_metrics['val/loss']:.4f}")
            print(f"Test MSE: {test_metrics['test/mse']:.4f}, Test MAE: {test_metrics['test/mae']:.4f}")
        
        # Final FLOPS logging
        if self.accelerator.is_main_process:
            self.log_flops_to_wandb(final=True)  # Final FLOPS accounting
    
    def train_step(self, batch):
        """Perform a single training step."""
        self.optimizer.zero_grad()
        
        # Get batch size and sequence length for FLOPS calculation
        batch_size = batch["input_ids"].size(0)
        seq_len = batch["input_ids"].size(1)
        
        # Track FLOPS before logging to get actual per-step FLOPS
        flops_before = self._get_current_flops()
        
        # Log FLOPs for this training step
        self.flops_calculator.log_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            rank=self.lora_rank,
            verbose=False,
            inference=False,
            description="training"
        )
        
        # Calculate FLOPS used in this step
        flops_after = self._get_current_flops()
        step_flops = flops_after - flops_before
        
        # Forward pass
        outputs = self.model(
            batch["input_ids"],
            labels=batch["input_ids"]
        )
        # print(outputs)

        loss = outputs.loss
        
        # Calculate CE loss separately for logging (already included in outputs.loss)
        # We need this for detailed tracking
        # logits = outputs.logits[:, :-1, :]  # remove last position
        # targets = batch["target"][:, 1:]    # shift right for next token prediction
        
        # # Reshape for cross-entropy calculation
        # logits_flat = logits.reshape(-1, logits.size(-1))
        # targets_flat = targets.reshape(-1)
        
        # # Calculate CE loss
        # ce_loss = torch.nn.functional.cross_entropy(
        #     logits_flat, 
        #     targets_flat, 
        #     ignore_index=self.tokenizer.pad_token_id
        # )
        
        # Backward pass and optimization
        self.accelerator.backward(loss)
        
        # Apply gradient clipping with max norm of 1.0
        self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item(), loss.item(), step_flops
    
    def log_gradient_stats(self, step=None):
        """Log gradient statistics for trainable parameters to wandb."""
        if not self.accelerator.is_main_process:
            return
        
        # Initialize statistics containers
        grad_stats = {
            "mean": [],
            "max": [],
            "min": [],
            "norm": [],
            "is_nan": [],
        }
        
        # Track layer-specific stats
        lora_a_stats = {"mean": [], "norm": []}
        lora_b_stats = {"mean": [], "norm": []}
        
        # For tracking LoRA parameter magnitudes
        lora_a_values = []
        lora_b_values = []
        
        # Collect statistics
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.detach().cpu()
                
                # Skip if NaN or Inf
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    grad_stats["is_nan"].append(1.0)
                    continue
                
                grad_stats["is_nan"].append(0.0)
                grad_stats["mean"].append(grad.abs().mean().item())
                grad_stats["max"].append(grad.abs().max().item())
                grad_stats["min"].append(grad.abs().min().item())
                grad_stats["norm"].append(grad.norm().item())
                
                # Track LoRA specific gradients
                if ".A" in name:
                    lora_a_stats["mean"].append(grad.abs().mean().item())
                    lora_a_stats["norm"].append(grad.norm().item())
                    # Also collect the parameter values themselves
                    lora_a_values.append(param.detach().cpu().flatten())
                elif ".B" in name:
                    lora_b_stats["mean"].append(grad.abs().mean().item())
                    lora_b_stats["norm"].append(grad.norm().item())
                    # Also collect the parameter values themselves
                    lora_b_values.append(param.detach().cpu().flatten())
        
        # Calculate aggregate statistics
        if grad_stats["mean"]:
            grad_metrics = {
                "gradients/mean": np.mean(grad_stats["mean"]),
                "gradients/max": np.max(grad_stats["max"]),
                "gradients/min": np.min(grad_stats["min"]),
                "gradients/norm": np.mean(grad_stats["norm"]),
                "gradients/nan_ratio": np.mean(grad_stats["is_nan"]),
            }
            
            # Log LoRA specific stats if available
            if lora_a_stats["mean"]:
                grad_metrics.update({
                    "gradients/lora_a_mean": np.mean(lora_a_stats["mean"]),
                    "gradients/lora_a_norm": np.mean(lora_a_stats["norm"]),
                })
            if lora_b_stats["mean"]:
                grad_metrics.update({
                    "gradients/lora_b_mean": np.mean(lora_b_stats["mean"]),
                    "gradients/lora_b_norm": np.mean(lora_b_stats["norm"]),
                })
            
            # Add histogram as a separate log to avoid mixing scalar metrics with histograms
            hist_metrics = {
                "gradients/norm_histogram": wandb.Histogram(np.array(grad_stats["norm"]))
            }
            
            # Add LoRA parameter histograms if available
            lora_param_metrics = {}
            if lora_a_values:
                # Concatenate all A matrices for histogram
                all_a_values = torch.cat(lora_a_values).numpy()
                lora_param_metrics["parameters/lora_a_histogram"] = wandb.Histogram(all_a_values)
                lora_param_metrics["parameters/lora_a_magnitude"] = np.mean(np.abs(all_a_values))
            
            if lora_b_values:
                # Concatenate all B matrices for histogram
                all_b_values = torch.cat(lora_b_values).numpy()
                lora_param_metrics["parameters/lora_b_histogram"] = wandb.Histogram(all_b_values)
                lora_param_metrics["parameters/lora_b_magnitude"] = np.mean(np.abs(all_b_values))
            
            # Log all metrics with the same step
            self._log_to_wandb(grad_metrics, step)
            self._log_to_wandb(hist_metrics, step)
            
            # Log LoRA parameter metrics
            if lora_param_metrics:
                self._log_to_wandb(lora_param_metrics, step)
    
    def evaluate(self, quick_eval=True):
        """
        Evaluate the model on the validation set.
        
        Args:
            quick_eval: If True, only calculate loss without generation (faster)
                       If False, perform full evaluation with generation and metrics
        """
        # Clean tqdm instance
        tqdm._instances.clear()
        self.model.eval()
        
        # Merge LoRA weights for evaluation
        self.merge_lora_weights()
        
        # Track FLOPS before validation
        flops_before = self._get_current_flops()
        
        # Create a dataloader from validation dataset
        dataloader = self.val_loader
        
        # Use different evaluation methods based on mode
        if quick_eval:
            # Fast evaluation - just calculate loss like during training
            metrics = self._quick_evaluation_loop(
                dataloader=dataloader,
                description="Quick Validation",
                metric_key_prefix="val"
            )
        else:
            # Full evaluation with generation and detailed metrics
            metrics = self.evaluation_loop(
                dataloader=dataloader,
                description="Validation",
                metric_key_prefix="val"
            )
        
        # Add step to metrics
        metrics["val/step"] = self.steps
        
        # Calculate FLOPS used during validation
        flops_after = self._get_current_flops()
        val_step_flops = flops_after - flops_before
        metrics["val/step_flops"] = val_step_flops
        
        # Unmerge LoRA weights after evaluation
        self.unmerge_lora_weights()
        
        return metrics, val_step_flops

    def _quick_evaluation_loop(self, dataloader, description, metric_key_prefix="eval"):
        """
        Fast evaluation loop that only calculates loss without generation.
        
        Args:
            dataloader: DataLoader for evaluation data
            description: Description for progress bar
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            dict: Dictionary of metrics
        """
        self.model.eval()
        
        # Track metrics
        losses = []
        
        # Process batches
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc=description, leave=False)):
            inputs = self._prepare_inputs(inputs)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    inputs["input_ids"],
                    labels=inputs["input_ids"]
                )
                loss = outputs.loss.item()
                losses.append(loss)
        
        # Calculate metrics
        metrics = {}
        if losses:
            metrics[f"{metric_key_prefix}/loss"] = np.mean(losses)
        
        return metrics

    def test(self):
        """Evaluate the model on the test set."""
        self.model.eval()
        
        # Merge LoRA weights for testing
        self.merge_lora_weights()
        
        # Track FLOPS before testing
        flops_before = self._get_current_flops()
        
        # Create a dataloader from test dataset
        dataloader = self.test_loader
        
        # Use the evaluation_loop function to perform evaluation
        metrics = self.evaluation_loop(
            dataloader=dataloader,
            description="Test",
            metric_key_prefix="test"
        )
        
        # Add step to metrics
        metrics["test/step"] = self.steps
        
        # Calculate FLOPS used during testing
        flops_after = self._get_current_flops()
        test_step_flops = flops_after - flops_before
        metrics["test/step_flops"] = test_step_flops
        
        # Let the calling code handle logging with _log_to_wandb
        # removing direct wandb.log call here
        
        # Unmerge LoRA weights after testing
        self.unmerge_lora_weights()
        
        return metrics, test_step_flops

    def save_checkpoint(self, suffix=None):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        # Create checkpoint name
        if suffix:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{suffix}.pth")
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.steps}.pth")
        
        # Create unwrapped model copy to save
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save all parameters that have gradients enabled
        trainable_state_dict = {}
        for name, param in unwrapped_model.named_parameters():
            if param.requires_grad:
                trainable_state_dict[name] = param
        
        # Save checkpoint
        checkpoint = {
            "model": trainable_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "best_val_loss": self.best_val_loss,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Store the path to the best checkpoint
        if suffix == "best":
            self.best_checkpoint_path = checkpoint_path
        
        # Log checkpoint to wandb
        artifact = wandb.Artifact(
            name=f"{wandb.run.name}-checkpoint-step-{self.steps}", 
            type="model",
            description=f"Model checkpoint at step {self.steps}"
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    def load_checkpoint(self, checkpoint_path):
        """Load a model checkpoint."""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            # First try to load with weights_only=True (safer)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            print(f"Warning: Loading checkpoint with weights_only=False due to: {str(e)}")
            try:
                # If that fails, try with weights_only=False
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e2:
                # If both attempts fail, continue with original model
                print(f"Error: Failed to load checkpoint even with weights_only=False: {str(e2)}")
                print(f"Continuing with original model without loading checkpoint.")
                return False
        
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # Load model weights
            model_state_dict = unwrapped_model.state_dict()
            for key, value in checkpoint["model"].items():
                if key in model_state_dict:
                    model_state_dict[key] = value
            
            unwrapped_model.load_state_dict(model_state_dict, strict=False)
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            
            # Load training state
            self.steps = checkpoint["steps"]
            self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            self.learning_rate = checkpoint.get("learning_rate", self.learning_rate)
            
            print(f"Loaded checkpoint from {checkpoint_path} (step {self.steps})")
            return True
        except Exception as e:
            print(f"Error while applying checkpoint: {str(e)}")
            print(f"Continuing with original model without loading checkpoint.")
            return False

    def _get_semicolon_token_id(self):
        """Helper method to get the semicolon token ID."""
        try:
            # First attempt: direct token conversion
            return self.tokenizer.convert_tokens_to_ids([';'])[0]
        except (AttributeError, IndexError):
            try:
                # Second attempt: encoding
                return self.tokenizer.encode(';', add_special_tokens=False)[0]
            except (AttributeError, IndexError):
                try:
                    # Third attempt: through vocabulary
                    semicolon_token_id = self.tokenizer.vocab.get(';')
                    if (semicolon_token_id is None):
                        raise ValueError()
                    return semicolon_token_id
                except (AttributeError, ValueError):
                    raise ValueError("Could not determine token ID for ';'. Check tokenizer implementation.")
    

    def log_flops_to_wandb(self, final=False, step=None):
        """Log FLOPS usage to wandb with detailed breakdown."""
        if not self.accelerator.is_main_process:
            return

        flops_df = pd.read_csv(self.flops_calculator.log_file)
        current_exp_flops = flops_df[flops_df['name'] == self.flops_calculator.log_name]
        
        # Total FLOPS used
        total_flops = current_exp_flops.flops.sum()
        
        # Per-step metrics
        train_flops = current_exp_flops[current_exp_flops['train_or_inference'] == 'training'].flops.sum()
        val_flops = current_exp_flops[current_exp_flops['train_or_inference'] == 'inference'].flops.sum()
        
        # Update FLOPS in wandb summary (these are not step-specific)
        wandb.run.summary["flops_used"] = total_flops
        wandb.run.summary["flops_budget_percent"] = total_flops / 1e17 * 100
        wandb.run.summary["flops_budget_percent_of_limit"] = (total_flops / 1e17 * 100) / self.max_flops_budget_percent * 100
        
        # Log detailed FLOPS breakdown
        flops_by_type = current_exp_flops.groupby('train_or_inference').agg({'flops': 'sum'})
        flops_by_desc = current_exp_flops.groupby('description').agg({'flops': 'sum'})
        
        # Create metrics dict
        flops_metrics = {
            "flops/total": total_flops,
            "flops/budget_percent": total_flops / 1e17 * 100,
            "flops/train_total": train_flops,
            "flops/val_total": val_flops,
            "flops/per_train_step_avg": train_flops / max(1, self.steps),
            "flops/per_eval_step_avg": val_flops / max(1, self.steps // self.eval_interval),
        }
        
        # Add training vs inference breakdown
        # for op_type, flops in flops_by_type.iterrows():
        #     flops_metrics[f"flops/{op_type}_total"] = flops['flops']
        #     flops_metrics[f"flops/{op_type}_percent"] = flops['flops'] / total_flops * 100
        
        # # Add detailed operation breakdown
        # for desc, flops in flops_by_desc.iterrows():
        #     flops_metrics[f"flops/by_operation/{desc}"] = flops['flops']
        #     flops_metrics[f"flops/by_operation/{desc}_percent"] = flops['flops'] / total_flops * 100
        
        # Log metrics with the same step
        self._log_to_wandb(flops_metrics, step)
        
        # If this is the final summary, create a pie chart
        # if final:
        #     # Create pie chart of FLOPS usage
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
        #     # Train vs Inference pie chart
        #     train_infer_data = flops_by_type.reset_index()
        #     ax1.pie(train_infer_data['flops'], labels=train_infer_data['train_or_inference'], 
        #            autopct='%1.1f%%', startangle=90)
        #     ax1.set_title('FLOPS Usage: Training vs Inference')
            
        #     # Operation breakdown pie chart
        #     op_data = flops_by_desc.reset_index()
        #     ax2.pie(op_data['flops'], labels=op_data['description'], 
        #            autopct='%1.1f%%', startangle=90)
        #     ax2.set_title('FLOPS Usage by Operation')
            
        #     plt.tight_layout()
        #     chart_metrics = {"flops/summary_charts": wandb.Image(fig)}
        #     self._log_to_wandb(chart_metrics, step)
        #     plt.close(fig)
            
        #     # Also log as a table
        #     table_metrics = {
        #         "flops/summary_table": wandb.Table(
        #             columns=["Operation", "FLOPS", "Percent"],
        #             data=[[op, f"{flops:,.0f}", f"{flops/total_flops*100:.2f}%"] 
        #                   for op, flops in zip(op_data['description'], op_data['flops'])]
        #         )
        #     }
        #     self._log_to_wandb(table_metrics, step)

    def _get_current_flops(self):
        """Helper method to get current total FLOPS."""
        try:
            flops_df = pd.read_csv(self.flops_calculator.log_file)
            current_exp_flops = flops_df[flops_df['name'] == self.flops_calculator.log_name]
            return current_exp_flops.flops.sum()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return 0

    def decode_to_string(self, ids):
        """
        Decode token IDs to string representation (similar to processor.decode_to_string).
        
        Args:
            ids: Token IDs to decode
            
        Returns:
            Raw decoded string
        """
        # Delegate to processor's decode_to_string if available
        if hasattr(self.processor, 'decode_to_string'):
            return self.processor.decode_to_string(ids)
        else:
            # Fallback implementation using the tokenizer
            return self.tokenizer.decode(ids, skip_special_tokens=True)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval"
    ):
        """Evaluation loop for time series metrics using batched inference with semicolon tracking."""
        self.model.eval()
        
        # Track metrics
        losses = []
        mse_values = []
        mae_values = []
        mse_per_time = []
        mae_per_time = []
        first_pred = None
        first_target = None
        
        # Track inference speed metrics
        token_counts = []
        inference_times = []
        
        # Track generation completion metrics
        failed_generations = 0
        total_generations = 0
        
        # Store text examples from last batch for visualization
        last_batch_examples = []
        
        # Get semicolon token ID for sequence parsing
        semi_colon_id = self._get_semicolon_token_id()
        semi_colon_max = self.target_eval_pairs + 1
        
        # Process batches
        for batch_idx, inputs in enumerate(tqdm(dataloader, desc=description, leave=False, position=0)):
            inputs = self._prepare_inputs(inputs)
            batch_size = inputs["input_ids"].size(0)
            total_generations += batch_size
            
            # Track text examples from this batch
            batch_examples = []
            
            # Store original inputs for each sample in the batch
            original_inputs = inputs["input_ids"].clone()
            original_length = original_inputs.shape[1]
            original_targets = inputs["target"].clone()
            
            # Track each sample's status
            active_samples = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            semicolon_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            generated_tokens = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            
            # Batch generation
            max_new_tokens = semi_colon_max * 12  # Conservative estimate
            start_time = time.time()
            
            # Initial input_ids for generation - initialize as list of tensors for easier append operations
            input_ids_list = [inputs["input_ids"][i].clone() for i in range(batch_size)]
            
            # Store generations for each sample
            all_generations = [[] for _ in range(batch_size)]
            per_sample_ce_losses = [[] for _ in range(batch_size)]
            
            # Start batched generation

            for step in tqdm(range(max_new_tokens), desc="Generating", leave=False, position=1):
                # If no active samples left, we're done
                if not active_samples.any():
                    break
                    
                # Get current active batch
                active_indices = active_samples.nonzero().squeeze(-1)
                
                # Handle case where active_indices is a scalar (only one active sample)
                if (active_indices.dim() == 0):
                    active_indices = active_indices.unsqueeze(0)
                
                # Create active batch from individual tensors in list
                active_input_ids = torch.stack([input_ids_list[i][-self.context_length:] for i in active_indices])
                
                # Log FLOPs for this inference step
                # self.flops_calculator.log_flops(
                #     batch_size=len(active_indices),
                #     seq_len=min(active_input_ids.size(1), self.context_length),
                #     rank=self.lora_rank,
                #     verbose=False,
                #     inference=True,
                #     description=f"{description}_batch"
                # )
                
                # Forward pass with context window for active samples
                outputs = self.model(active_input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append next tokens
                for i, sample_idx in enumerate(active_indices):
                    # Get original sample index and append token
                    token_id = next_tokens[i].item()
                    all_generations[sample_idx].append(token_id)
                    input_ids_list[sample_idx] = torch.cat([
                        input_ids_list[sample_idx], 
                        torch.tensor([token_id], device=self.device)
                    ])
                    generated_tokens[sample_idx] += 1
                    
                    # Calculate CE loss if applicable
                    target_pos = len(all_generations[sample_idx]) - 1

                    if target_pos < original_targets[sample_idx].shape[0]:
                        target_token = original_targets[sample_idx, target_pos]
                        loss = torch.nn.functional.cross_entropy(
                            next_token_logits[i].unsqueeze(0), 
                            target_token.unsqueeze(0)
                        )
                        
                
                        losses.append(loss.item())
                        per_sample_ce_losses[sample_idx].append(loss.item())

    
                    
                    # Check if this is a semicolon
                    if token_id == semi_colon_id:
                        semicolon_counts[sample_idx] += 1
                        
                        # Mark sample as done if reached max semicolons
                        if semicolon_counts[sample_idx] >= semi_colon_max:
                            active_samples[sample_idx] = False
                
                # No longer need to update input_ids for next iteration since we're using a list of tensors
            
            # Record inference times for entire batch
            batch_inference_time = time.time() - start_time
            for i in range(batch_size):
                token_counts.append(generated_tokens[i].item())
                inference_times.append(batch_inference_time / batch_size)  # Approximate per-sample time
                
                # Count failed generations (didn't reach enough semicolons)
                if semicolon_counts[i] < semi_colon_max:
                    failed_generations += 1
            
            # Process results for evaluation
            sample_tqdm = tqdm(range(batch_size), desc="Processing", leave=False, position=2)
            for sample_idx in range(batch_size):
                try:
                    # Get the original input and generated tokens
                    original_input = original_inputs[sample_idx]
                    generated_sequence = all_generations[sample_idx]
                    
                    # Convert to full sequence of IDs - use the stored tensor directly
                    full_sequence = input_ids_list[sample_idx]
                    
                    # Decode to string format
                    pred_ids = full_sequence.cpu().numpy()[original_length:]
                    target_ids = original_targets[sample_idx].cpu().numpy()

                    
                    # For the last batch, save examples for visualization
                    if batch_idx == len(dataloader) - 1:
                        predicted_text = self.processor.decode_to_string(pred_ids)
                        target_text = self.processor.decode_to_string(target_ids)
                        batch_examples.append({
                            "sample_idx": sample_idx,
                            "predicted_text": predicted_text,
                            "target_text": target_text,
                            "semicolon_count": semicolon_counts[sample_idx].item(),
                            "success": semicolon_counts[sample_idx].item() >= semi_colon_max
                        })
                    
                    
                    # Decode to values using processor
                    predicted_text = self.processor.decode_to_string(pred_ids)
                    target_text = self.processor.decode_to_string(target_ids)
                    
                    # Parse values for metrics calculation
                    pred_values = self._parse_values(predicted_text, semicolon_counts[sample_idx].item())
                    target_values = self._parse_values(target_text, semicolon_counts[sample_idx].item())
                    

                    # Calculate metrics if we have enough values
                    if (len(pred_values) >= self.target_eval_pairs and 
                        len(target_values) >= self.target_eval_pairs):
                        
                        # Limit to target pairs
                        pred_values = pred_values[:self.target_eval_pairs]
                        target_values = target_values[:self.target_eval_pairs]
                        
                        # Save first prediction for visualization
                        if batch_idx == 0 and sample_idx == 0:
                            first_pred = pred_values.copy()
                            first_target = target_values.copy()
                        
                        # Calculate metrics
                        mse = np.mean((pred_values - target_values) ** 2)
                        mae = np.mean(np.abs(pred_values - target_values))
                        
                        mse_per_timestep = np.mean((pred_values - target_values) ** 2, axis=1)
                        mae_per_timestep = np.mean(np.abs(pred_values - target_values), axis=1)
                        
                        mse_values.append(mse)
                        mae_values.append(mae)
                        mse_per_time.append(mse_per_timestep)
                        mae_per_time.append(mae_per_timestep)

                except Exception as e:
                    if self.is_world_process_zero():
                        print(f"Error processing sequence {batch_idx}/{sample_idx}: {e}")
            
            # Save examples from the last batch
            if batch_idx == len(dataloader) - 1:
                last_batch_examples = batch_examples
        
        # Calculate final metrics
        metrics = {}
        
        # Average loss
        if losses:
            metrics[f"{metric_key_prefix}/loss"] = np.nanmean(losses)
        
        # MSE and MAE metrics
        if mse_values:
            metrics[f"{metric_key_prefix}/mse"] = np.mean(mse_values)
            metrics[f"{metric_key_prefix}/mae"] = np.mean(mae_values)
            metrics[f"{metric_key_prefix}/samples_evaluated"] = len(mse_values)
            
            # Per-time metrics if available
            if mse_per_time:
                mse_per_time_array = np.stack(mse_per_time)
                mae_per_time_array = np.stack(mae_per_time)
                
                metrics[f"{metric_key_prefix}/mse_per_time"] = np.mean(mse_per_time_array, axis=0).tolist()
                metrics[f"{metric_key_prefix}/mae_per_time"] = np.mean(mae_per_time_array, axis=0).tolist()
        
        # Generation completion metrics
        metrics[f"{metric_key_prefix}/failed_generations"] = failed_generations
        metrics[f"{metric_key_prefix}/total_generations"] = total_generations
        metrics[f"{metric_key_prefix}/failure_rate"] = failed_generations / total_generations if total_generations > 0 else 0
        
        # Store visualization data
        if first_pred is not None:
            metrics["pred_values"] = first_pred
            metrics["target_values"] = first_target
        
        # Calculate inference speed metrics
        if token_counts and inference_times:
            total_tokens = sum(token_counts)
            total_time = sum(inference_times)
            tokens_per_second = total_tokens / total_time if total_time > 0 else 0
            ms_per_token = (total_time / total_tokens) * 1000 if total_tokens > 0 else 0
            
            metrics[f"{metric_key_prefix}/tokens_per_second"] = tokens_per_second
            metrics[f"{metric_key_prefix}/ms_per_token"] = ms_per_token
            metrics[f"{metric_key_prefix}/total_tokens_generated"] = total_tokens
            metrics[f"{metric_key_prefix}/total_inference_time"] = total_time
        
        # Log example generations to wandb
        if self.is_world_process_zero() and last_batch_examples:
            example_table = wandb.Table(columns=["Sample", "Success", "Prediction", "Target"])
            for example in last_batch_examples[:5]:  # Log up to 5 examples to avoid clutter
                example_table.add_data(
                    example["sample_idx"],
                    "yes" if example["success"] else "no",
                    example["predicted_text"][:500],  # Limit text length
                    example["target_text"][:500]      # Limit text length
                )
            metrics[f"{metric_key_prefix}/generation_examples"] = example_table
        
        # Log metrics to console
        if self.is_world_process_zero():
            mse_str = f"{metrics.get(f'{metric_key_prefix}/mse', 'N/A')}"
            mae_str = f"{metrics.get(f'{metric_key_prefix}/mae', 'N/A')}"
            samples = metrics.get(f"{metric_key_prefix}/samples_evaluated", "unknown")
            failure_rate = metrics.get(f"{metric_key_prefix}/failure_rate", 0) * 100
            
            inference_speed = metrics.get(f"{metric_key_prefix}/tokens_per_second", None)
            if inference_speed is not None:
                print(f"\n{description} on {samples} samples - "
                      f"MSE: {mse_str}, MAE: {mae_str}, "
                      f"Failure rate: {failure_rate:.1f}%, "
                      f"Speed: {inference_speed:.2f} tokens/sec")
            else:
                print(f"\n{description} on {samples} samples - "
                      f"MSE: {mse_str}, MAE: {mae_str}, "
                      f"Failure rate: {failure_rate:.1f}%")
        
        # Return metrics
        return metrics

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for model by moving them to the appropriate device.
        
        Args:
            inputs: The input dictionary containing tensors
            
        Returns:
            Dictionary with tensors moved to the appropriate device
        """
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        elif isinstance(inputs, (list, tuple)):
            return [v.to(self.device) if hasattr(v, 'to') else v for v in inputs]
        else:
            return inputs.to(self.device) if hasattr(inputs, 'to') else inputs
        
    def is_world_process_zero(self):
        """Helper method to check if this is the main process."""
        return self.accelerator.is_main_process

    def _parse_values(self, text, count):
        """Parse numerical values from semicolon-separated text."""
        # Extract all value pairs 
        values = text.split(";")[1:count]
        values = [pair.split(',') for pair in values]
        
        # Filter valid pairs and convert to float
        values = [[float(v) for v in pair] for pair in values if len(pair) == 2 
                                                            and all(self._is_valid_float(v) for v in pair)]
        
        if not values:
            return np.array([], dtype=np.float32)
        
        # Convert to numpy and scale
        values = np.array(values, dtype=np.float32)
        values = values / self.processor.scaler
        
        return values

    def _is_valid_float(self, value):
        """Check if a string can be converted to a valid float."""
        try:
            float(value)
            return True
        except:
            return False
    
    def _log_to_wandb(self, metrics, step=None):
        """Centralized method to log metrics to wandb.
        
        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step to associate with these metrics
        """
        if self.accelerator.is_main_process:
            wandb.log(metrics, step=step)
