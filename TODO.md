# TODO for TimeSeriesTrainer

- [x] Add FLOPS tracking integration with wandb
- [x] Add LoRA weight merging/unmerging
- [x] Implement basic time series metrics (MSE/MAE)
<!-- - [ ] Improve prediction visualization with more detailed plots -->
- [ ] Add support for different prediction strategies (greedy, sampling, etc.)
- [ ] Implement cross-entropy per token tracking
- [ ] Add comprehensive documentation and examples
- [ ] Implement hypothesis testing for model comparisons
- [ ] Add support for custom preprocessing pipelines
- [ ] Create a benchmark suite for model comparison
- [ ] Add early stopping based on validation metrics
- [ ] Implement learning rate scheduling optimization
- [ ] Add support for multi-GPU training with DDP
- [ ] Implement automatic hyperparameter tuning
- [ ] Create model artifacts storage and retrieval system

## Sanity check

### Rationale
Before committing to long training runs, we need efficient verification of the Trainer's functionality. Proper sanity checks save computation budget and help identify bugs early.

### Pre-Training Sanity Checks

1. **Model Input/Output Verification**
   - [ ] Confirm inputs properly flow through model (shape consistency)
   - [ ] Verify outputs have expected dimensions
   - [ ] Check token generation behavior with small sequence

2. **Parameter Gradients**
   - [ ] Confirm LoRA parameters receive gradients
   - [ ] Verify frozen parameters remain unchanged
   - [ ] Validate gradient magnitudes are reasonable (not exploding/vanishing)
   - [ ] Test backward pass works correctly

3. **LoRA Functionality**
   - [ ] Test weight merging produces expected results
   - [ ] Verify unmerging restores original weights exactly
   - [ ] Confirm merged/unmerged behavior difference in predictions

4. **FLOPS Accounting**
   - [ ] Validate per-step FLOPS calculations are accurate
   - [ ] Check FLOPS are logged correctly for both training and inference
   - [ ] Verify batch-level FLOPS accounting with small batches

5. **Data Pipeline**
   - [ ] Confirm batches maintain consistent dimensions
   - [ ] Test preprocessing and target creation
   - [ ] Verify padding and masking behavior

