#!/usr/bin/env python3
"""
Test Comparison Script for DIOAD Repository
==============================================

This script evaluates the DIOAD model using the same test protocol 
as the DIAD repository for fair comparison.

Usage:
    python test_comparison.py --test-dir path/to/test_drdc --model-path model-vit-ang-loss.h5

Requirements:
    - Place this file in the repository directory
    - Activate the 'dioad' conda environment
    - Have the model file (model-vit-ang-loss.h5) in the repository
"""

import os
import glob
import argparse
import json
import numpy as np
from PIL import Image
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Import DIOAD modules
from models import load_vit_model
from infer import Inference

# ==============================================================================
# METRICS CALCULATION (copied from DIAD repository)
# ==============================================================================

class CircularMetrics:
    """Statistical metrics for circular/angular data"""
    
    @staticmethod
    def angular_distance(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate angular distance between true and predicted angles (in degrees)"""
        diff = np.abs(y_true - y_pred)
        return np.minimum(diff, 360 - diff)
    
    @staticmethod
    def circular_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error for circular data"""
        return np.mean(CircularMetrics.angular_distance(y_true, y_pred))
    
    @staticmethod
    def circular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Square Error for circular data"""
        angular_diffs = CircularMetrics.angular_distance(y_true, y_pred)
        return np.sqrt(np.mean(angular_diffs ** 2))
    
    @staticmethod
    def circular_median(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median error for circular data"""
        return np.median(CircularMetrics.angular_distance(y_true, y_pred))
    
    @staticmethod
    def circular_percentile(y_true: np.ndarray, y_pred: np.ndarray, percentile: float) -> float:
        """Percentile error for circular data"""
        angular_diffs = CircularMetrics.angular_distance(y_true, y_pred)
        return np.percentile(angular_diffs, percentile)
    
    @staticmethod
    def accuracy_within_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold_deg: float) -> float:
        """Fraction of predictions within threshold degrees"""
        angular_diffs = CircularMetrics.angular_distance(y_true, y_pred)
        return np.mean(angular_diffs <= threshold_deg)
    
    @staticmethod
    def cumulative_accuracy_curve(y_true: np.ndarray, y_pred: np.ndarray, max_error: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate cumulative accuracy curve"""
        angular_diffs = CircularMetrics.angular_distance(y_true, y_pred)
        thresholds = np.linspace(0, max_error, 1000)
        accuracies = np.array([np.mean(angular_diffs <= t) for t in thresholds])
        return thresholds, accuracies
    
    @staticmethod
    def auc_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, max_threshold: float) -> float:
        """Area Under Curve up to max_threshold degrees"""
        thresholds, accuracies = CircularMetrics.cumulative_accuracy_curve(y_true, y_pred, max_threshold)
        # Normalize AUC by max_threshold to get value between 0 and 1
        return np.trapz(accuracies, thresholds) / max_threshold


def compute_test_metrics(y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str = "test") -> Dict[str, float]:
    """
    Compute comprehensive test metrics for angle prediction
    
    Args:
        y_true: True angles in degrees [0, 360)
        y_pred: Predicted angles in degrees [0, 360)  
        dataset_name: Name of the dataset for metric prefixing
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Basic error metrics
    metrics[f'{dataset_name}_mae_deg'] = CircularMetrics.circular_mae(y_true, y_pred)
    metrics[f'{dataset_name}_rmse_deg'] = CircularMetrics.circular_rmse(y_true, y_pred)
    metrics[f'{dataset_name}_median_deg'] = CircularMetrics.circular_median(y_true, y_pred)
    
    # Percentile metrics
    metrics[f'{dataset_name}_p90_deg'] = CircularMetrics.circular_percentile(y_true, y_pred, 90)
    metrics[f'{dataset_name}_p95_deg'] = CircularMetrics.circular_percentile(y_true, y_pred, 95)
    
    # Accuracy within thresholds
    for threshold in [2, 5, 10]:
        acc = CircularMetrics.accuracy_within_threshold(y_true, y_pred, threshold)
        metrics[f'{dataset_name}_acc_{threshold}deg'] = acc
    
    # Area Under Curve metrics
    for threshold in [2, 5, 10]:
        auc = CircularMetrics.auc_at_threshold(y_true, y_pred, threshold)
        metrics[f'{dataset_name}_auc_{threshold}deg'] = auc
    
    return metrics


# ==============================================================================
# DIOAD MODEL INTERFACE
# ==============================================================================

class DeepOADPredictor:
    """Wrapper for DIOAD model prediction"""
    
    def __init__(self, model_path: str = None):
        """Initialize the DIOAD model"""
        print("Loading DIOAD model...")
        self.inference = Inference(load_model_name=model_path)
        print("Model loaded successfully!")
    
    def predict_angle(self, image_path: str) -> float:
        """Predict orientation angle for a single image"""
        try:
            # Use the DIOAD inference method
            # Their model predicts the CORRECTION angle (counter-rotation), not the applied rotation
            raw_predicted_angle = self.inference.predict("vit", image_path, postprocess_and_save=False)
            
            # Convert correction angle to applied rotation angle
            # If image was rotated by +45°, their model predicts -45° (correction needed)
            # So we need to negate their prediction to get the applied rotation
            applied_rotation_angle = -raw_predicted_angle
            
            # Convert to [0, 360) range for consistency
            normalized_angle = applied_rotation_angle % 360
            
            return float(normalized_angle)
        
        except Exception as e:
            print(f"Error predicting angle for {image_path}: {e}")
            return 0.0  # Return 0 as fallback
    
    def predict_batch(self, image_paths: List[str], rotation_angles: np.ndarray = None) -> List[float]:
        """
        Predict orientation angles for a batch of images
        
        Args:
            image_paths: List of image file paths
            rotation_angles: Array of angles to rotate each image by (for testing)
        
        Returns:
            List of predicted angles
        """
        predictions = []
        total = len(image_paths)
        temp_files = []  # Keep track of temporary files for cleanup
        
        try:
            for i, image_path in enumerate(image_paths):
                if i % 10 == 0:
                    print(f"Processing image {i+1}/{total}")
                
                # If rotation angles are provided, rotate the image first
                if rotation_angles is not None:
                    rotation_angle = rotation_angles[i]
                    # Rotate the image by the specified angle
                    rotated_image_path = rotate_image(image_path, rotation_angle)
                    temp_files.append(rotated_image_path)
                    # Predict on the rotated image
                    predicted_angle = self.predict_angle(rotated_image_path)
                else:
                    # Predict on the original image
                    predicted_angle = self.predict_angle(image_path)
                
                predictions.append(predicted_angle)
            
        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    os.remove(temp_file)
                except:
                    pass  # Ignore cleanup errors
        
        return predictions


# ==============================================================================
# TEST DATASET LOADING
# ==============================================================================

def load_test_dataset(test_dir: str, rotation_range: float = 360.0) -> Tuple[List[str], np.ndarray]:
    """
    Load test dataset with ground truth angles
    
    Args:
        test_dir: Directory containing test images
        rotation_range: Maximum rotation range in degrees
        
    Returns:
        Tuple of (image_paths, true_angles)
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in image_extensions:
        pattern = os.path.join(test_dir, '**', ext)
        image_paths.extend(glob.glob(pattern, recursive=True))
    
    if not image_paths:
        raise ValueError(f"No images found in {test_dir}")
    
    print(f"Found {len(image_paths)} test images")
    
    # For fair testing, generate random ground truth angles
    # These will be the angles we rotate the images by
    np.random.seed(27)  # Match DIAD repository's test_random_seed
    n_images = len(image_paths)
    
    if rotation_range == 360.0:
        # Full rotation range [0, 360)
        true_angles = np.random.uniform(0, 360, n_images)
    else:
        # Limited rotation range [-rotation_range/2, +rotation_range/2]
        half_range = rotation_range / 2
        true_angles = np.random.uniform(-half_range, half_range, n_images)
        # Convert to [0, 360) range
        true_angles = (true_angles + 360) % 360
    
    return sorted(image_paths), true_angles


def rotate_image(image_path: str, angle: float, output_path: str = None) -> str:
    """
    Rotate an image by a given angle and save it
    
    Args:
        image_path: Path to the input image
        angle: Rotation angle in degrees (clockwise)
        output_path: Path to save rotated image (if None, use temporary file)
    
    Returns:
        Path to the rotated image file
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Rotate image (PIL uses counter-clockwise rotation, so negate angle)
    rotated_image = image.rotate(-angle, expand=True, fillcolor=(255, 255, 255))
    
    # Generate output path if not provided
    if output_path is None:
        import tempfile
        temp_dir = tempfile.mkdtemp()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(temp_dir, f"{base_name}_rotated_{angle:.1f}.jpg")
    
    # Save rotated image
    rotated_image.save(output_path, 'JPEG', quality=95)
    
    return output_path


# ==============================================================================
# MAIN EVALUATION FUNCTION
# ==============================================================================

def run_test_evaluation(test_dir: str, model_path: str = None, rotation_range: float = 360.0, output_dir: str = None, output_file: str = None):
    """
    Run comprehensive test evaluation using DIOAD model
    
    Args:
        test_dir: Directory containing test images
        model_path: Path to the DIOAD model file
        rotation_range: Maximum rotation range for testing
        output_dir: Directory to save output files (optional)
        output_file: Specific JSON filename override (optional)
    """
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine output file paths
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    base_dir = output_dir if output_dir else "."
    json_filename = output_file if output_file else f"{timestamp}_maji_dioad_results.json"
    overview_filename = f"{timestamp}_maji_dioad_overview.txt"
    
    json_path = os.path.join(base_dir, json_filename)
    overview_path = os.path.join(base_dir, overview_filename)
    
    # Collect all output for saving to file
    output_lines = []
    
    def print_and_save(*args, **kwargs):
        """Print to console and save to output list"""
        line = ' '.join(str(arg) for arg in args)
        print(line, **kwargs)
        output_lines.append(line)
    
    print_and_save("=" * 80)
    print_and_save("DIOAD MODEL EVALUATION")
    print_and_save("=" * 80)
    print_and_save(f"Test Directory: {test_dir}")
    print_and_save(f"Model Path: {model_path}")
    print_and_save(f"Rotation Range: {rotation_range}°")
    print_and_save()
    
    # Load test dataset
    print_and_save("Loading test dataset...")
    image_paths, true_angles = load_test_dataset(test_dir, rotation_range)
    print_and_save(f"Loaded {len(image_paths)} test images")
    print_and_save()
    
    # Initialize predictor
    predictor = DeepOADPredictor(model_path)
    
    # Run predictions
    print("Running predictions...")
    print("Note: Images will be rotated by random angles, then fed to the model for prediction")
    print()
    
    print("Running full evaluation...")
    start_time = time.time()
    predicted_angles = predictor.predict_batch(image_paths, true_angles)
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed in {prediction_time:.2f}s")
    print(f"Average time per image: {prediction_time/len(image_paths):.3f}s")
    print()
    
    # Convert to numpy arrays
    y_true = np.array(true_angles)
    y_pred = np.array(predicted_angles)
    
    print(f"\nOverall statistics:")
    print(f"Processed {len(y_true)} test images")
    print(f"True angles range: {y_true.min():.1f}° to {y_true.max():.1f}°")
    print(f"Predicted angles range: {y_pred.min():.1f}° to {y_pred.max():.1f}°")
    
    # Basic error analysis
    angular_errors = CircularMetrics.angular_distance(y_true, y_pred)
    large_error_mask = angular_errors > 30  # Errors larger than 30 degrees
    large_errors_count = np.sum(large_error_mask)
    
    print(f"\nError analysis:")
    print(f"Errors > 30°: {large_errors_count}/{len(y_true)} ({100*large_errors_count/len(y_true):.1f}%)")
    print()
    
    # Compute metrics with proper dataset naming (match DIAD format)
    dataset_name = os.path.basename(test_dir)  # e.g., "test_drdc"
    metrics = compute_test_metrics(y_true, y_pred, dataset_name)
    
    # Create JSON result structure (matching compare.py format)
    json_result = {
        'approach': 'maji_dioad',
        'model_name': 'vit-ang-loss',
        'model_short': 'vit-ang-loss',
        'loss_type': 'angular_loss',
        'learning_rate': 0,  # Not applicable for pre-trained model
        'epochs': 0,  # Not applicable for pre-trained model
        'training_time': 0,  # Not applicable for pre-trained model
        'test_mae': metrics.get(f'{dataset_name}_mae_deg', None),
        'success': True,
        **metrics  # Include all computed metrics
    }
    
    # Print structured results (matching DIAD format)
    print_and_save("=" * 80)
    print_and_save("TEST RESULTS")
    print_and_save("=" * 80)
    
    print_and_save("=== TEST_RESULTS_START ===")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print_and_save(f"{key}={value:.3f}")
        else:
            print_and_save(f"{key}={value}")
    print_and_save("=== TEST_RESULTS_END ===")
    
    # Print human-readable summary
    print_and_save("\n" + "=" * 80)
    print_and_save("PERFORMANCE SUMMARY")
    print_and_save("=" * 80)
    
    mae = metrics.get(f'{dataset_name}_mae_deg', 0)
    rmse = metrics.get(f'{dataset_name}_rmse_deg', 0)
    median = metrics.get(f'{dataset_name}_median_deg', 0)
    p95 = metrics.get(f'{dataset_name}_p95_deg', 0)
    
    acc_2 = metrics.get(f'{dataset_name}_acc_2deg', 0) * 100
    acc_5 = metrics.get(f'{dataset_name}_acc_5deg', 0) * 100
    acc_10 = metrics.get(f'{dataset_name}_acc_10deg', 0) * 100
    
    print_and_save(f"Mean Absolute Error: {mae:.2f}°")
    print_and_save(f"Root Mean Square Error: {rmse:.2f}°")
    print_and_save(f"Median Error: {median:.2f}°")
    print_and_save(f"95th Percentile Error: {p95:.2f}°")
    print_and_save()
    print_and_save(f"Accuracy within 2°: {acc_2:.1f}%")
    print_and_save(f"Accuracy within 5°: {acc_5:.1f}%")
    print_and_save(f"Accuracy within 10°: {acc_10:.1f}%")
    
    # Performance interpretation
    print_and_save(f"\nPerformance Assessment:")
    if mae < 2.0:
        quality = "EXCELLENT"
    elif mae < 5.0:
        quality = "GOOD"
    elif mae < 10.0:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print_and_save(f"Overall Performance: {quality} (MAE: {mae:.2f}°)")
    
    print_and_save("=" * 80)
    
    # Save results to files if output directory or file specified
    if output_dir or output_file:
        try:
            # Save JSON results
            with open(json_path, 'w') as f:
                json.dump(json_result, f, indent=2)
            print(f"\nJSON results saved to: {json_path}")
            
            # Save overview text file
            with open(overview_path, 'w') as f:
                f.write('\n'.join(output_lines))
            print(f"Overview saved to: {overview_path}")
            
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test DIOAD model with DIAD evaluation protocol",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--test-dir", type=str, required=True,
                       help="Directory containing test images")
    parser.add_argument("--model-path", type=str, default="model-vit-ang-loss.h5",
                       help="Path to the DIOAD model file")
    parser.add_argument("--rotation-range", type=float, default=360.0,
                       help="Maximum rotation range for testing (degrees)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save output files (creates JSON and overview files)")
    parser.add_argument("--output", type=str, default=None,
                       help="Specific JSON filename override (optional)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        return
    
    if args.model_path and not os.path.exists(args.model_path):
        print(f"Warning: Model file not found: {args.model_path}")
        print("Will attempt to load default model...")
    
    # Run evaluation
    try:
        run_test_evaluation(args.test_dir, args.model_path, args.rotation_range, args.output_dir, args.output)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()