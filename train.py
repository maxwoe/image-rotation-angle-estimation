#!/usr/bin/env python3
"""
Flexible Training Script for Orientation Estimation
=================================================

This script allows easy switching between two different approaches:
1. Direct Angle Prediction (single output neuron)
2. Unit Vector Approach (two output neurons for cos/sin)

Usage:
    # Train with unit vector approach (default, more stable)
    python train_flexible.py --approach=unit_vector --loss-type=mse
    
    # Train with direct angle approach (simpler, traditional)
    python train_flexible.py --approach=direct_angle --loss-type=mae
"""

import json
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.tuner import Tuner
from loguru import logger
# Removed config.py imports - using inline defaults
import matplotlib.pyplot as plt
from metrics import compute_test_metrics, CircularMetrics
import torch
from architectures import get_default_learning_rate, get_scaled_learning_rate

# Import all model approaches
from model_unit_vector import UnitVectorAngleEstimation
from model_direct_angle import DirectAngleEstimation
from model_classification import ClassificationAngleEstimation
from model_cgd import CGDAngleEstimation
from model_psc import PSCAngleEstimation
from model_multibin import MultiBinAngleEstimation

torch.set_float32_matmul_precision('high')

def get_model_class(approach):
    """Get the appropriate model class based on approach"""
    if approach == "unit_vector":
        return UnitVectorAngleEstimation
    elif approach == "direct_angle":
        return DirectAngleEstimation
    elif approach == "classification":
        return ClassificationAngleEstimation
    elif approach == "cgd":
        return CGDAngleEstimation
    elif approach == "psc":
        return PSCAngleEstimation
    elif approach == "multibin":
        return MultiBinAngleEstimation
    else:
        raise ValueError(
            f"Unknown approach: {approach}. Use 'unit_vector', 'direct_angle', 'classification', 'cgd', 'psc', 'multibin'")


def find_learning_rate(model, trainer, args):
    """
    Find optimal learning rate using PyTorch Lightning's built-in LR finder
    
    Args:
        model: Lightning module
        trainer: Lightning trainer
        args: Command line arguments
        
    Returns:
        suggested_lr: Suggested optimal learning rate
    """
    logger.info("Starting learning rate finder...")
    
    # Create tuner
    tuner = Tuner(trainer)
    
    # Run learning rate finder
    lr_finder = tuner.lr_find(
        model,
        min_lr=args.lr_min,
        max_lr=args.lr_max,
        num_training=100,  # Number of learning rate steps
    )

    # Get suggested learning rate
    suggested_lr = lr_finder.suggestion()
    
    # Plot results
    fig = lr_finder.plot(suggest=True, show=False)
    
    # Save plot if requested
    if args.lr_plot:
        plot_path = os.path.join(args.save_dir, f"lr_finder_{args.approach}_{args.loss_type}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Learning rate plot saved to: {plot_path}")
    
    # Show results
    logger.info(f"Learning rate finder results:")
    logger.info(f"   Suggested LR: {suggested_lr}")
    
    plt.close(fig)  # Clean up
    
    return suggested_lr


def train_model(args):
    """Train model with specified approach and parameters"""
    logger.info(f"Starting training with {args.approach} approach")

    # Determine learning rate with priority: explicit > scaled > default
    if args.learning_rate:
        learning_rate = args.learning_rate
        lr_source = "explicit"
    else:
        learning_rate = get_scaled_learning_rate(args.model_name, args.batch_size)
        lr_source = f"scaled for {args.model_name} (batch_size={args.batch_size})"
    
    logger.info(f"Learning rate: {learning_rate} ({lr_source})")
    logger.info(f"Model: {args.model_name}")

    # Get the appropriate model class
    ModelClass = get_model_class(args.approach)

    # Handle checkpoint resuming
    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume_ckpt}")
        
        logger.info(f"Resuming training from checkpoint: {args.resume_ckpt}")
        
        # Load model from checkpoint - this automatically restores model state
        checkpoint_kwargs = {
            "batch_size": args.batch_size,
            "train_dir": args.train_dir,
            "model_name": args.model_name,
            "learning_rate": learning_rate,
            "validation_split": args.validation_split,
            "random_seed": args.random_seed,
            "image_size": args.image_size,
            "loss_type": args.loss_type,
            "test_rotation_range": args.test_rotation_range,
            "test_random_seed": args.test_random_seed,
        }
        
        # Add test directory if provided
        if args.test_dirs:
            checkpoint_kwargs["test_dir"] = args.test_dirs[0]
        
                
        model = ModelClass.load_from_checkpoint(args.resume_ckpt, **checkpoint_kwargs)
        logger.info("Model state loaded from checkpoint")
    else:
        # Create new model instance
        model_kwargs = {
            "batch_size": args.batch_size,
            "train_dir": args.train_dir,
            "model_name": args.model_name,
            "learning_rate": learning_rate,
            "validation_split": args.validation_split,
            "random_seed": args.random_seed,
            "image_size": args.image_size,
            "test_rotation_range": args.test_rotation_range,
            "test_random_seed": args.test_random_seed,
        }
        
        if args.loss_type:
            model_kwargs["loss_type"] = args.loss_type
            
        # Add test directory if provided
        if args.test_dirs:
            model_kwargs["test_dir"] = args.test_dirs[0]
            

        model = ModelClass(**model_kwargs)
        logger.info("Created new model instance")
        
        # Load pretrained weights if specified
        if args.pretrained_ckpt:
            model.load_pretrained_weights(args.pretrained_ckpt)

    # Setup logger
    logger_tb = TensorBoardLogger(
        args.save_dir, name=f"{args.approach}-{model.loss_type}-{args.model_name}-{16 if args.mixed_precision else 32}")
    
    # Log hyperparameters
    logger_tb.log_hyperparams(vars(args))
    if rank_zero_only.rank == 0:
        with open(os.path.join(logger_tb.log_dir, "args.json"), "w") as fp:
            json.dump(vars(args), fp, indent=4)

    # Setup callbacks
    cp_callback = ModelCheckpoint(monitor='val_loss',
                                  mode="min",
                                  save_top_k=1,  # Save the best checkpoint
                                  save_last=True,  # Also save the last checkpoint
                                  auto_insert_metric_name=True,
                                  filename="{epoch:04d}-{step:07d}-{train_loss:.4f}-{val_loss:.4f}-{train_mae_deg_epoch:.4f}-{val_mae_deg:.4f}")
    # add model name and version to checkpoint files (monkey-patch the callback’s filename)
    cp_callback.filename = (f"{logger_tb.name}-version={logger_tb.version}-" + cp_callback.filename)

    callbacks = [
        cp_callback,
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Add EarlyStopping if not disabled
    if not args.disable_early_stopping:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stopping)
        logger.info(f"EarlyStopping enabled with patience={args.early_stopping_patience}")

    # Configure trainer
    trainer_kwargs = {
        "max_epochs": args.max_epochs,
        "callbacks": callbacks,
        "logger": logger_tb,
        "accelerator": "auto",
        "devices": "auto",
        "precision": "16-mixed" if args.mixed_precision else "32",
        "check_val_every_n_epoch": args.val_epoch
    }

    # Add gradient accumulation if specified
    if args.accumulate_grad_batches > 1:
        trainer_kwargs["accumulate_grad_batches"] = args.accumulate_grad_batches

    # Add development/testing options
    if args.dev_mode:
        trainer_kwargs.update({
            "overfit_batches": 1,
            "max_epochs": 10,
            "log_every_n_steps": 1,
            "limit_train_batches": 5,
            "limit_val_batches": 2
        })
        logger.info("Development mode enabled - quick training for testing")

    if args.fast_dev_run:
        trainer_kwargs["fast_dev_run"] = True
        logger.info(
            "Fast dev run enabled - single batch through train/val/test")

    if args.overfit_batches:
        trainer_kwargs["overfit_batches"] = args.overfit_batches
        trainer_kwargs["log_every_n_steps"] = 1
        trainer_kwargs["val_check_interval"] = 1.0
        logger.info(f"Overfitting on {args.overfit_batches} batch(es)")

    # Create trainer and start training
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Now that model is attached to trainer, save optimizer configuration
    if rank_zero_only.rank == 0:
        try:
            # Temporarily attach trainer to model for optimizer config extraction
            model.trainer = trainer
            optimizer_config = model.configure_optimizers()
            with open(os.path.join(logger_tb.log_dir, "optimizer_config.txt"), "w") as fp:
                fp.write(str(optimizer_config))
            logger.info(str(optimizer_config))
        except Exception as e:
            logger.warning(f"Failed to save optimizer configuration: {e}")
        
    # Run learning rate finder if requested
    if args.find_lr and not args.resume_ckpt:  # Don't run LR finder when resuming
        find_learning_rate(model, trainer, args)
        return model, trainer
    
    # Start training (with checkpoint resuming if specified)
    if args.resume_ckpt:
        # PyTorch Lightning will automatically restore optimizer state, LR scheduler state, 
        # epoch counter, step counter, and random states when resuming from checkpoint
        trainer.fit(model, ckpt_path=args.resume_ckpt)
        logger.info("Training resumed from checkpoint with full state restoration")
    else:
        trainer.fit(model)
        logger.info("Training started from scratch")

    return model, trainer


def load_model_for_testing(args, checkpoint_path, test_dir=None):
    """Load model from checkpoint for testing."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Test checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint for testing: {checkpoint_path}")
    
    # Get the appropriate model class
    ModelClass = get_model_class(args.approach)
    
    # Load model from checkpoint - let Lightning restore all saved hyperparameters
    # Only pass test-specific parameters
    model = ModelClass.load_from_checkpoint(
        checkpoint_path, 
        test_dir=test_dir or (args.test_dirs[0] if args.test_dirs else None),
        test_rotation_range=args.test_rotation_range,
        test_random_seed=args.test_random_seed
    )
    logger.info("Model loaded successfully for testing")
    return model


def run_test_evaluation(model, args, trainer=None):
    """Run test evaluation using PyTorch Lightning trainer."""
    logger.info("Starting test evaluation...")
    
    # Reuse existing trainer if provided, otherwise create a minimal one for testing
    if trainer is None:
        trainer_kwargs = {
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed" if args.mixed_precision else "32",
            "enable_progress_bar": True,
            "enable_model_summary": False,
            "logger": False  # Disable logging for test-only runs
        }
        trainer = pl.Trainer(**trainer_kwargs)
        logger.info("Created new trainer for testing")
    else:
        logger.info("Reusing existing trainer for testing")
    
    all_test_results = {}
    
    # Run test evaluation on each test directory
    for test_dir in args.test_dirs:
        if not os.path.exists(test_dir):
            logger.warning(f"Skipping non-existent test directory: {test_dir}")
            continue
            
        # Extract directory name for metric prefixing
        dir_name = os.path.basename(test_dir)
        logger.info(f"Testing on {dir_name} dataset: {test_dir}")
        
        # Update model's test_dir for this specific test
        model.test_dir = test_dir
        
        # Run test evaluation - use "best" checkpoint if available, otherwise use current model state
        # In test-only mode, model is already loaded from specified checkpoint, so use None
        if hasattr(args, 'test_only') and args.test_only:
            # Test-only mode: use the already loaded model state
            test_results = trainer.test(model, ckpt_path=None, verbose=True)
        else:
            # Training+test mode: use best validation checkpoint
            test_results = trainer.test(model, ckpt_path="best", verbose=True)
        
        if test_results:
            # Extract and prefix test metrics with directory name
            test_metrics = test_results[0]  # First (and only) test result
            
            # Prefix all metrics with directory name
            prefixed_metrics = {}
            for key, value in test_metrics.items():
                # Remove 'test_' prefix if it exists and add directory prefix
                clean_key = key.replace('test_', '', 1) if key.startswith('test_') else key
                prefixed_key = f"{dir_name}_{clean_key}"
                prefixed_metrics[prefixed_key] = value
            
            all_test_results.update(prefixed_metrics)
            
            logger.info(f"Test evaluation on {dir_name} completed!")
        else:
            logger.warning(f"No test results returned for {dir_name}")
    
    if all_test_results:
        logger.info("All test evaluations completed!")
        logger.info("Combined Test Results Summary:")
        
        # Print structured results for compare_approaches.py to parse
        print("=== TEST_RESULTS_START ===")
        for key, value in all_test_results.items():
            if isinstance(value, (int, float)):
                print(f"{key}={value:.3f}")
            else:
                print(f"{key}={value}")
        print("=== TEST_RESULTS_END ===")
        
        # Show comprehensive evaluation if requested
        if hasattr(args, 'comprehensive_eval') and args.comprehensive_eval:
            show_comprehensive_test_analysis(all_test_results)
    else:
        logger.warning("No test results returned from any dataset")


def show_comprehensive_test_analysis(test_metrics: dict):
    """Show comprehensive test analysis with statistical insights"""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST EVALUATION ANALYSIS")
    print("="*80)
    
    # Extract prediction data if available (this would need model-specific implementation)
    # For now, show available metrics in a structured way
    
    print("\n1. ACCURACY METRICS")
    print("-" * 25)
    
    accuracy_metrics = {
        'test_mae_deg': ('Mean Absolute Error', '°', 2),
        'test_median_deg': ('Median Error', '°', 2),
        'test_rmse_deg': ('Root Mean Square Error', '°', 2),
    }
    
    for key, (name, unit, decimals) in accuracy_metrics.items():
        if key in test_metrics:
            value = test_metrics[key]
            print(f"{name:.<30} {value:.{decimals}f}{unit}")
    
    print("\n2. DISTRIBUTION METRICS")
    print("-" * 30)
    
    distribution_metrics = {
        'test_p90_deg': ('90th Percentile Error', '°', 1),
        'test_p95_deg': ('95th Percentile Error', '°', 1),
    }
    
    for key, (name, unit, decimals) in distribution_metrics.items():
        if key in test_metrics:
            value = test_metrics[key]
            print(f"{name:.<30} {value:.{decimals}f}{unit}")
    
    print("\n3. PRACTICAL PERFORMANCE")
    print("-" * 35)
    
    practical_metrics = {
        'test_acc_2deg': ('Accuracy within 2°', '%', 1),
        'test_acc_5deg': ('Accuracy within 5°', '%', 1), 
        'test_acc_10deg': ('Accuracy within 10°', '%', 1),
        'test_auc_2deg': ('AUC @ 2° threshold', '', 3),
        'test_auc_5deg': ('AUC @ 5° threshold', '', 3),
        'test_auc_10deg': ('AUC @ 10° threshold', '', 3),
    }
    
    for key, (name, unit, decimals) in practical_metrics.items():
        if key in test_metrics:
            value = test_metrics[key]
            if unit == '%':
                value *= 100  # Convert to percentage
            print(f"{name:.<30} {value:.{decimals}f}{unit}")
    
    print("\n4. PERFORMANCE INTERPRETATION")
    print("-" * 40)
    
    mae = test_metrics.get('test_mae_deg', None)
    if mae is not None:
        if mae < 2.0:
            quality = "EXCELLENT"
        elif mae < 5.0:
            quality = "GOOD"
        elif mae < 10.0:
            quality = "ACCEPTABLE"
        else:
            quality = "NEEDS IMPROVEMENT"
        
        print(f"Overall Performance: {quality} (MAE: {mae:.2f}°)")
    
    auc_5 = test_metrics.get('test_auc_5deg', None)
    if auc_5 is not None:
        if auc_5 > 0.9:
            usability = "HIGHLY PRACTICAL"
        elif auc_5 > 0.8:
            usability = "PRACTICAL"
        elif auc_5 > 0.6:
            usability = "MODERATELY USEFUL"
        else:
            usability = "LIMITED UTILITY"
        
        print(f"Practical Usability: {usability} (AUC@5°: {auc_5:.3f})")
    
    p95 = test_metrics.get('test_p95_deg', None)
    if p95 is not None:
        if p95 < 10.0:
            reliability = "HIGH RELIABILITY"
        elif p95 < 20.0:
            reliability = "MODERATE RELIABILITY"
        else:
            reliability = "VARIABLE PERFORMANCE"
        
        print(f"Worst-case Reliability: {reliability} (P95: {p95:.1f}°)")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Training for image angle estimation with different approaches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model approach selection
    parser.add_argument("--approach", type=str, default="unit_vector",
                        choices=["unit_vector", "direct_angle", "classification", "cgd", "psc", "multibin"],
                        help="Model approach: unit_vector (2 outputs), direct_angle (1 output), classification (N classes), cgd (probability distribution), psc (phase shift coder), multibin (multiple heads)")

    # Model architecture
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_224",
                        help="Model architecture to use for feature extraction")
    
    # Training data
    parser.add_argument("--train-dir", type=str, default="data/datasets/train_drcd",
                        help="Path to directory containing correctly oriented training images")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="Fraction of data to use for validation (0.0-1.0)")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed for reproducible train/validation splits")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, help="Learning rate (overrides optimal rate for the model architecture)")
    parser.add_argument("--max-epochs", type=int, default=100000,
                        help="Maximum number of epochs")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override image input size (default: use TIMM model default)")
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='validate and save every n epoch')

    # Loss function options
    parser.add_argument("--loss-type", type=str, help="Loss function type")
       
    # Model saving
    parser.add_argument("--save-dir", type=str, default="data/saved_models",
                        help="Directory to save model weights")

    # Training options
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Enable mixed precision training (16-bit)")
    parser.add_argument("--accumulate-grad-batches", type=int, default=1,
                        help="Number of batches to accumulate gradients over")
    parser.add_argument("--early-stopping-patience", type=int, default=15,
                        help="Number of epochs with no improvement before stopping")
    parser.add_argument("--disable-early-stopping", action="store_true",
                        help="Disable early stopping (train for full max-epochs)")

    # Learning rate finder options
    parser.add_argument("--find-lr", action="store_true",
                        help="Run learning rate finder before training to find optimal LR")
    parser.add_argument("--lr-min", type=float, default=1e-8,
                        help="Minimum learning rate for LR finder")
    parser.add_argument("--lr-max", type=float, default=1.0,
                        help="Maximum learning rate for LR finder")
    parser.add_argument("--lr-plot", action="store_true",
                        help="Save learning rate finder plot to file")

    # Toroidal approach options
    parser.add_argument("--spring-weight", type=float, default=0.1,
                        help="Weight for circular spring loss in toroidal approach")

    # Development/testing options
    parser.add_argument("--dev-mode", action="store_true",
                        help="Enable development mode (overfit_batches=1, max_epochs=10)")
    parser.add_argument("--fast-dev-run", action="store_true",
                        help="Run single batch through train/val for quick testing")
    parser.add_argument("--overfit-batches", type=int,
                        help="Number of batches to overfit on (useful for testing)")

    # Checkpoint resuming and pretraining
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint file to resume training from (full state)")
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                        help="Path to checkpoint file to load weights from (fresh optimizer/scheduler)")

    # Test evaluation options
    parser.add_argument("--test-dirs", type=str, nargs='+', default=["data/datasets/test_drdc"],
                        help="Path(s) to test dataset directory(ies)")
    parser.add_argument("--run-test", action="store_true",
                        help="Run test evaluation after training. Uses the BEST validation checkpoint")
    parser.add_argument("--test-only", action="store_true", 
                        help="Only run test evaluation (requires --resume-ckpt)")
    parser.add_argument("--test-ckpt", type=str, default=None,
                        help="Specific checkpoint to use for test evaluation")
    parser.add_argument("--test-rotation-range", type=float, default=360.0,
                        help="Max rotation range for test evaluation (degrees). 360=full range, 45=±45°")
    parser.add_argument("--test-random-seed", type=int, default=27,
                        help="Random seed for repeatable test rotations")
    parser.add_argument("--comprehensive-eval", action="store_true",
                        help="Show comprehensive metrics with confidence intervals after test evaluation")

    args = parser.parse_args()

    # Validate arguments
    if not (0.0 < args.validation_split < 1.0):
        raise ValueError("validation-split must be between 0.0 and 1.0")
    
    # Validate checkpoint files
    if args.resume_ckpt and args.pretrained_ckpt:
        raise ValueError("Cannot use both --resume-ckpt and --pretrained-ckpt. Choose one.")
    
    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint file not found: {args.resume_ckpt}")
        if not args.resume_ckpt.endswith('.ckpt'):
            logger.warning(f"Checkpoint file should have .ckpt extension: {args.resume_ckpt}")
        logger.info(f"Resume checkpoint found: {args.resume_ckpt}")
    
    if args.pretrained_ckpt:
        if not os.path.exists(args.pretrained_ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint file not found: {args.pretrained_ckpt}")
        if not args.pretrained_ckpt.endswith('.ckpt'):
            logger.warning(f"Checkpoint file should have .ckpt extension: {args.pretrained_ckpt}")
        logger.info(f"Pretrained checkpoint found: {args.pretrained_ckpt}")
    
    # Validate test arguments
    if args.test_only and not args.resume_ckpt and not args.test_ckpt:
        raise ValueError("--test-only requires either --resume-ckpt or --test-ckpt")
    
    # Validate test directories
    for test_dir in args.test_dirs:
        if not os.path.exists(test_dir):
            logger.warning(f"Test directory not found: {test_dir}")
    
    # Determine checkpoint for test evaluation
    test_checkpoint_path = args.test_ckpt or args.resume_ckpt

    # Create weights directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Approach: {args.approach}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Train directory: {args.train_dir}")
    logger.info(f"  Validation split: {args.validation_split}")
    logger.info(f"  Random seed: {args.random_seed} (train/val split)")
    logger.info(f"  Test random seed: {args.test_random_seed} (test rotations)")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Loss function: {args.loss_type}")
    logger.info(f"  Max epochs: {args.max_epochs}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
        
    if args.resume_ckpt:
        logger.info(f"  Resume from: {args.resume_ckpt}")
    elif args.pretrained_ckpt:
        logger.info(f"  Pretrained weights: {args.pretrained_ckpt}")
        logger.info(f"  Starting: Fresh training with pretrained weights")
    else:
        logger.info(f"  Starting: Fresh training")

    # Handle different modes
    if args.test_only:
        # Test-only mode: load model and run test evaluation
        logger.info("Test-only mode: Loading model and running test evaluation")
        model = load_model_for_testing(args, test_checkpoint_path)
        run_test_evaluation(model, args)
    else:
        # Normal training mode
        model, trainer = train_model(args)
        
        # Run test evaluation after training if requested
        if args.run_test and args.test_dirs and any(os.path.exists(test_dir) for test_dir in args.test_dirs):
            logger.info("Training completed, starting test evaluation...")
            run_test_evaluation(model, args, trainer)
        
        logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
