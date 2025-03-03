# one_peace/utils/setup_training.py
import os
import sys
import torch
import argparse
from .fairseq_logger_callback import FairseqLoggerCallback

def setup_training_environment(args=None):
    """
    Setup training environment with enhanced logging and checkpointing.
    
    Args:
        args: Argparse namespace or dict with training configuration
    
    Returns:
        Updated args with logging and checkpointing configuration
    """
    if args is None:
        # If no args provided, try to get them from sys.argv
        parser = argparse.ArgumentParser()
        parser.add_argument('--config-dir', type=str, required=True)
        parser.add_argument('--config-name', type=str, required=True)
        parser.add_argument('--save-dir', type=str, default='checkpoints/anuraset')
        parser.add_argument('--logging-dir', type=str, default=None)
        parser.add_argument('--use-wandb', action='store_true')
        parser.add_argument('--wandb-project', type=str, default='one_peace')
        parser.add_argument('--wandb-entity', type=str, default=None)
        parser.add_argument('--log-interval', type=int, default=10)
        parser.add_argument('--checkpoint-activations', action='store_true')
        parser.add_argument('--offload-activations', action='store_true')
        
        args, _ = parser.parse_known_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set logging directory
    if args.logging_dir is None:
        args.logging_dir = os.path.join(args.save_dir, 'logs')
    os.makedirs(args.logging_dir, exist_ok=True)
    
    # Setup enhanced checkpoint configuration
    args.save_interval = getattr(args, 'save_interval', 1)
    args.save_interval_updates = getattr(args, 'save_interval_updates', 1000)
    args.keep_interval_updates = getattr(args, 'keep_interval_updates', 5)
    args.keep_last_epochs = getattr(args, 'keep_last_epochs', 5)
    args.keep_best_checkpoints = getattr(args, 'keep_best_checkpoints', 5)
    args.maximize_best_checkpoint_metric = False  # We're tracking loss (minimize)
    args.best_checkpoint_metric = 'loss'
    args.patience = getattr(args, 'patience', 5)
    args.no_epoch_checkpoints = False
    args.no_last_checkpoints = False
    
    # Memory efficiency settings for long training runs
    if torch.cuda.is_available():
        # Use checkpoint activations to save memory
        if getattr(args, 'checkpoint_activations', False):
            args.checkpoint_activations = True
            
        # Offload activations to CPU to save more GPU memory
        if getattr(args, 'offload_activations', False):
            args.offload_activations = True
    
    # Create logger
    logger = FairseqLoggerCallback(args)
    
    # Add logger to args for access in training script
    args.custom_logger = logger
    
    return args

def main():
    """Main function to run before the actual training script."""
    args = setup_training_environment()
    
    # Print configuration summary
    print("\nTraining Environment Setup:")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Logging directory: {args.logging_dir}")
    print(f"  Checkpoint interval: {args.save_interval} epochs, {args.save_interval_updates} updates")
    print(f"  Keep checkpoints: {args.keep_interval_updates} by updates, {args.keep_last_epochs} by epochs")
    print(f"  Best metric: {args.best_checkpoint_metric} (minimize={not args.maximize_best_checkpoint_metric})")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Memory optimizations: checkpoint_activations={args.checkpoint_activations}, offload_activations={args.offload_activations}")
    print("\nContinuing to main training script...\n")
    
    # Return args for use in actual training script
    return args

if __name__ == "__main__":
    args = main()
    # Now the actual training script would be executed with these args