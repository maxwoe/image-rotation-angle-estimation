#!/usr/bin/env python3
"""
Comprehensive Circular Metrics for Orientation Estimation
========================================================

This module provides industry-standard metrics for evaluating orientation estimation models,
including circular-aware statistics and bootstrap confidence intervals.

Metrics included:
- MAE (Mean Absolute Error)
- Median Error  
- RMSE (Root Mean Square Error)
- P90/P95 (90th/95th percentiles)
- AUC@{2°, 5°, 10°} (Area Under Curve at different thresholds)
- Bootstrap 95% confidence intervals
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for a metric value with confidence interval"""
    value: float
    ci_lower: float
    ci_upper: float
    
    def __str__(self):
        margin = (self.ci_upper - self.ci_lower) / 2
        return f"{self.value:.2f}±{margin:.2f} [{self.ci_lower:.2f}, {self.ci_upper:.2f}]"
    
    def __repr__(self):
        margin = (self.ci_upper - self.ci_lower) / 2
        return f"MetricResult({self.value:.2f}±{margin:.2f})"


class CircularMetrics:
    """
    Comprehensive circular metrics for orientation estimation evaluation.
    
    Computes various metrics from predicted and true angles, handling the circular
    nature of angular data (0° = 360°) and providing bootstrap confidence intervals.
    """
    
    def __init__(self, pred_angles: torch.Tensor, true_angles: torch.Tensor):
        """
        Initialize with predicted and true angles.
        
        Args:
            pred_angles: Predicted angles in degrees [B]
            true_angles: True angles in degrees [B]
        """
        self.pred_angles = pred_angles.detach().cpu()
        self.true_angles = true_angles.detach().cpu()
        self.errors = self._circular_distance(pred_angles, true_angles).detach().cpu()
        
    def _circular_distance(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Calculate circular distance between angles (shortest path around circle)"""
        diff = torch.abs(pred - true)
        return torch.minimum(diff, 360 - diff)
    
    def mae(self) -> float:
        """Mean Absolute Error in degrees"""
        return float(torch.mean(self.errors))
    
    def median(self) -> float:
        """Median error in degrees (more robust to outliers than MAE)"""
        return float(torch.median(self.errors))
    
    def rmse(self) -> float:
        """Root Mean Square Error in degrees (penalizes large errors more)"""
        return float(torch.sqrt(torch.mean(self.errors ** 2)))
    
    def percentile(self, p: float) -> float:
        """Calculate p-th percentile of errors"""
        return float(torch.quantile(self.errors, p / 100.0))
    
    def p90(self) -> float:
        """90th percentile error (shows tail behavior)"""
        return self.percentile(90)
    
    def p95(self) -> float:
        """95th percentile error (worst-case performance indicator)"""
        return self.percentile(95)
    
    def accuracy_within_threshold(self, threshold: float) -> float:
        """Fraction of predictions within threshold degrees"""
        return float(torch.mean((self.errors <= threshold).float()))
    
    def auc_at_threshold(self, threshold: float, n_points: int = 100) -> float:
        """
        Area Under Curve for cumulative accuracy up to threshold.
        
        Computes the area under the cumulative accuracy curve from 0° to threshold°.
        Higher values indicate better performance across all error levels.
        
        Args:
            threshold: Maximum error threshold in degrees
            n_points: Number of points to sample for AUC calculation
            
        Returns:
            AUC value between 0 and 1
        """
        thresholds = torch.linspace(0, threshold, n_points)
        accuracies = []
        
        for t in thresholds:
            acc = self.accuracy_within_threshold(float(t))
            accuracies.append(acc)
        
        # Compute AUC using trapezoidal rule, normalized by threshold
        accuracies = torch.tensor(accuracies)
        auc = torch.trapz(accuracies, thresholds) / threshold
        return float(auc)
    
    def bootstrap_metric(self, metric_fn, n_bootstrap: int = 1000, confidence: float = 0.95) -> MetricResult:
        """
        Compute bootstrap confidence interval for any metric function.
        
        Args:
            metric_fn: Function that takes CircularMetrics object and returns float
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            MetricResult with value and confidence interval
        """
        n_samples = len(self.errors)
        bootstrap_values = []
        
        # Generate bootstrap samples
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = torch.randint(0, n_samples, (n_samples,))
            boot_pred = self.pred_angles[indices]
            boot_true = self.true_angles[indices]
            
            # Create bootstrap metrics object
            boot_metrics = CircularMetrics(boot_pred, boot_true)
            
            # Compute metric on bootstrap sample
            boot_value = metric_fn(boot_metrics)
            bootstrap_values.append(boot_value)
        
        # Calculate confidence interval
        bootstrap_values = torch.tensor(bootstrap_values)
        alpha = 1 - confidence
        ci_lower = torch.quantile(bootstrap_values, alpha / 2)
        ci_upper = torch.quantile(bootstrap_values, 1 - alpha / 2)
        
        # Original metric value
        original_value = metric_fn(self)
        
        return MetricResult(
            value=original_value,
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper)
        )
    
    def compute_all_metrics(self, n_bootstrap: int = 1000) -> Dict[str, MetricResult]:
        """
        Compute all standard metrics with bootstrap confidence intervals.
        
        Args:
            n_bootstrap: Number of bootstrap samples for CI computation
            
        Returns:
            Dictionary of metric names to MetricResult objects
        """
        metrics = {}
        
        # Basic metrics
        metrics['mae_deg'] = self.bootstrap_metric(lambda m: m.mae(), n_bootstrap)
        metrics['median_deg'] = self.bootstrap_metric(lambda m: m.median(), n_bootstrap)
        metrics['rmse_deg'] = self.bootstrap_metric(lambda m: m.rmse(), n_bootstrap)
        metrics['p90_deg'] = self.bootstrap_metric(lambda m: m.p90(), n_bootstrap)
        metrics['p95_deg'] = self.bootstrap_metric(lambda m: m.p95(), n_bootstrap)
        
        # AUC metrics at different thresholds
        metrics['auc_2deg'] = self.bootstrap_metric(lambda m: m.auc_at_threshold(2.0), n_bootstrap)
        metrics['auc_5deg'] = self.bootstrap_metric(lambda m: m.auc_at_threshold(5.0), n_bootstrap)
        metrics['auc_10deg'] = self.bootstrap_metric(lambda m: m.auc_at_threshold(10.0), n_bootstrap)
        
        # Accuracy metrics (for quick reference)
        metrics['acc_2deg'] = self.bootstrap_metric(lambda m: m.accuracy_within_threshold(2.0), n_bootstrap)
        metrics['acc_5deg'] = self.bootstrap_metric(lambda m: m.accuracy_within_threshold(5.0), n_bootstrap)
        metrics['acc_10deg'] = self.bootstrap_metric(lambda m: m.accuracy_within_threshold(10.0), n_bootstrap)
        
        return metrics
    
    def quick_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics without confidence intervals (faster for logging).
        
        Returns:
            Dictionary of metric names to values
        """
        return {
            'mae_deg': self.mae(),
            'median_deg': self.median(),
            'rmse_deg': self.rmse(),
            'p90_deg': self.p90(),
            'p95_deg': self.p95(),
            'auc_2deg': self.auc_at_threshold(2.0),
            'auc_5deg': self.auc_at_threshold(5.0),
            'auc_10deg': self.auc_at_threshold(10.0),
            'acc_2deg': self.accuracy_within_threshold(2.0),
            'acc_5deg': self.accuracy_within_threshold(5.0),
            'acc_10deg': self.accuracy_within_threshold(10.0),
        }


def format_metrics_table(results: Dict[str, Dict[str, MetricResult]], 
                        approaches: List[str],
                        title: str = "Model Comparison Results") -> str:
    """
    Format comprehensive metrics results into a publication-ready table.
    
    Args:
        results: Dictionary mapping approach_name -> metrics_dict
        approaches: List of approach names in desired order
        title: Table title
        
    Returns:
        Formatted string table
    """
    # Metric display configuration
    metric_config = [
        ('mae_deg', 'MAE (°)', 1),
        ('median_deg', 'Median (°)', 1), 
        ('rmse_deg', 'RMSE (°)', 1),
        ('p90_deg', 'P90 (°)', 1),
        ('p95_deg', 'P95 (°)', 1),
        ('auc_2deg', 'AUC@2°', 3),
        ('auc_5deg', 'AUC@5°', 3),
        ('auc_10deg', 'AUC@10°', 3),
    ]
    
    # Build table
    lines = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    # Header
    header = f"{'Metric':<12}"
    for approach in approaches:
        header += f"{approach:>18}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Metrics rows
    for metric_key, metric_name, decimals in metric_config:
        row = f"{metric_name:<12}"
        for approach in approaches:
            if approach in results and metric_key in results[approach]:
                result = results[approach][metric_key]
                ci_width = (result.ci_upper - result.ci_lower) / 2
                formatted = f"{result.value:.{decimals}f}±{ci_width:.{decimals}f}"
                row += f"{formatted:>18}"
            else:
                row += f"{'N/A':>18}"
        lines.append(row)
    
    return "\n".join(lines)


# Convenience functions for model integration
def compute_validation_metrics(pred_angles: torch.Tensor, 
                             true_angles: torch.Tensor,
                             prefix: str = "val") -> Dict[str, float]:
    """
    Compute quick metrics for validation logging.
    
    Args:
        pred_angles: Predicted angles in degrees
        true_angles: True angles in degrees  
        prefix: Metric name prefix (e.g., "val", "test")
        
    Returns:
        Dictionary with prefixed metric names for logging
    """
    metrics = CircularMetrics(pred_angles, true_angles)
    quick_metrics = metrics.quick_metrics()
    
    # Add prefix to metric names
    return {f"{prefix}_{k}": v for k, v in quick_metrics.items()}


def compute_test_metrics(pred_angles: torch.Tensor,
                        true_angles: torch.Tensor,
                        n_bootstrap: int = 1000) -> Dict[str, MetricResult]:
    """
    Compute comprehensive test metrics with confidence intervals.
    
    Args:
        pred_angles: Predicted angles in degrees
        true_angles: True angles in degrees
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary of metrics with confidence intervals
    """
    metrics = CircularMetrics(pred_angles, true_angles)
    return metrics.compute_all_metrics(n_bootstrap)