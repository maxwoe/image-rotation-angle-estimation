#!/usr/bin/env python3
"""
Circular Gaussian Distribution (CGD) Approach for Orientation Estimation
======================================================================

This approach represents angles as probability distributions over the angular space.
Based on the paper: "A Circular Gaussian Distribution Method for Object Orientation" 
(Electronics 2023, 12, 3265)

Instead of predicting angles directly or as unit vectors, the model outputs a probability
distribution over discretized angle bins. The ground truth is represented as a Gaussian
distribution centered at the true angle.

Model output: Probability distribution over angle bins (e.g., 360 bins for 1° resolution)
Advantages: Natural periodicity, uncertainty estimation, smoother learning, no boundary issues
Disadvantages: Higher computational cost, more complex output processing
"""

import math
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import timm
import timm.data
from PIL import Image
import os
import glob
from loguru import logger
# Removed config.py imports - parameters now explicitly required
from data_loader import RotationDataset
from metrics import compute_validation_metrics, compute_test_metrics


class CircularGaussianDistribution(nn.Module):
    """
    Circular Gaussian Distribution module for 360° image orientation
    """
    
    def __init__(self, num_bins: int = 360, sigma: float = 6.0):
        """
        Args:
            num_bins: Number of angle bins (360 for 1° resolution)
            sigma: Standard deviation of Gaussian distribution in degrees
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma
        self.bin_size = 360.0 / num_bins  # degrees per bin
        
        # Pre-compute bin centers for full circle: [0°, 1°, 2°, ..., 359°]
        bin_centers = torch.arange(0, 360, self.bin_size)
        self.register_buffer('bin_centers', bin_centers)
        
        logger.info(f"CGD: {num_bins} bins, range [0°, 360°), σ={sigma}°")
    
    def angle_to_distribution(self, angles: torch.Tensor) -> torch.Tensor:
        """
        Convert angles to Gaussian probability distributions following paper Eq. 3
        
        Paper formula: d_l = exp(-(l - θ_t)²/(2σ²))
        Adapted for circular distances to handle angle wrapping correctly.
        
        Args:
            angles: Tensor of angles in degrees [B]
            
        Returns:
            distributions: Tensor of probability distributions [B, num_bins]
        """       
        # Normalize angles to [0°, 360°) range
        angles_norm = angles % 360.0
        
        # Expand for vectorized computation
        angles_expanded = angles_norm.unsqueeze(1)  # [B, 1]
        bin_centers_expanded = self.bin_centers.unsqueeze(0)  # [1, num_bins]
        
        # Calculate circular distances (handle angle wrapping)
        diff = torch.abs(angles_expanded - bin_centers_expanded)  # [B, num_bins]
        # Handle wraparound: distance between 359° and 1° should be 2°, not 358°
        diff = torch.minimum(diff, 360.0 - diff)
        
        # Paper formula: d_l = exp(-(l - θ_t)²/(2σ²))
        distributions = torch.exp(-(diff ** 2) / (2 * self.sigma ** 2))
        
        # Normalize to probability distribution
        distributions = distributions / (distributions.sum(dim=1, keepdim=True) + 1e-8)
        
        return distributions
    
    def distribution_to_angle(self, distributions: torch.Tensor, method: str = 'argmax') -> torch.Tensor:
        """
        Extract angles from probability distributions
        
        Args:
            distributions: Probability distributions [B, num_bins]
            method: Extraction method ('argmax', 'weighted_average', 'peak_fitting')
            
        Returns:
            angles: Extracted angles in degrees [B] in [0°, 360°) range
        """
        if method == 'argmax':
            # Simple peak detection
            peak_indices = torch.argmax(distributions, dim=1)
            angles = self.bin_centers[peak_indices]
            
        elif method == 'weighted_average':
            # Weighted average for sub-bin precision with circular wrapping
            weights = distributions / (distributions.sum(dim=1, keepdim=True) + 1e-8)
            
            # Convert to unit vectors for circular averaging
            bin_angles_rad = self.bin_centers * torch.pi / 180.0
            cos_components = torch.cos(bin_angles_rad)
            sin_components = torch.sin(bin_angles_rad)
            
            # Calculate weighted averages of cos and sin components
            avg_cos = torch.sum(weights * cos_components.unsqueeze(0), dim=1)
            avg_sin = torch.sum(weights * sin_components.unsqueeze(0), dim=1)
            
            # Convert back to angles
            angles = torch.atan2(avg_sin, avg_cos) * 180.0 / torch.pi
            angles = angles % 360.0  # Normalize to [0°, 360°)
            
        elif method == 'peak_fitting':
            # Quadratic peak fitting for higher precision
            peak_indices = torch.argmax(distributions, dim=1)
            angles = torch.zeros_like(peak_indices, dtype=torch.float)
            
            for i in range(distributions.shape[0]):
                peak_idx = peak_indices[i].item()
                if 0 < peak_idx < self.num_bins - 1:
                    # Use quadratic interpolation
                    y1 = distributions[i, peak_idx - 1]
                    y2 = distributions[i, peak_idx]
                    y3 = distributions[i, peak_idx + 1]
                    
                    # Quadratic fit to find sub-bin peak
                    a = 0.5 * (y1 - 2*y2 + y3)
                    b = 0.5 * (y3 - y1)
                    
                    if abs(a) > 1e-8:
                        offset = -b / (2 * a)
                        offset = torch.clamp(offset, -0.5, 0.5)
                    else:
                        offset = 0
                    
                    angles[i] = self.bin_centers[peak_idx] + offset * self.bin_size
                else:
                    angles[i] = self.bin_centers[peak_idx]
        else:
            raise ValueError(f"Unknown extraction method: {method}")
        
        # Ensure angles are in [0°, 360°) range
        angles = angles % 360.0
        return angles
    
    def get_distribution_uncertainty(self, distributions: torch.Tensor) -> torch.Tensor:
        """
        Calculate uncertainty/confidence from distribution entropy
        
        Args:
            distributions: Probability distributions [B, num_bins]
            
        Returns:
            uncertainty: Entropy-based uncertainty measure [B]
        """
        # Calculate entropy
        log_probs = torch.log(distributions + 1e-8)
        entropy = -torch.sum(distributions * log_probs, dim=1)
        
        # Normalize by max possible entropy
        max_entropy = math.log(self.num_bins)
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy


class CGDAngleEstimation(pl.LightningModule):
    """
    Circular Gaussian Distribution (CGD) model for 360° image orientation estimation
    
    Uses probability distributions over the full 360° angle space.
    Based on the paper: Electronics 2023, 12, 3265 (adapted for full circle)
    """
    
    def __init__(
        self,
        batch_size: int,
        train_dir: str,
        model_name: str = "vit_tiny_patch16_224",
        learning_rate: float = 0.001,
        validation_split: float = 0.1,
        random_seed: int = 42,
        image_size: int = 224,
        num_bins: int = 360,  # Full circle with 1° resolution
        sigma: float = 6.0,   # Gaussian standard deviation in degrees (Xu et al. 2023 recommendation)
        inference_method: str = 'argmax',
        loss_type: str = 'kl_divergence',
        test_dir=None,
        test_rotation_range=360.0,
        test_random_seed=42
    ) -> None:
        """
        Args:
            model_name: Backbone model name
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            train_dir: Training data directory
            validation_split: Validation split ratio
            random_seed: Random seed
            image_size: Input image size
            num_bins: Number of angle bins (360 for 1° resolution)
            sigma: Gaussian standard deviation in degrees
            inference_method: Method for extracting angles ('argmax', 'weighted_average', 'peak_fitting')
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        # Store hyperparameters
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.image_size = image_size
        self.num_bins = num_bins
        self.sigma = sigma
        self.inference_method = inference_method
        self.loss_type = loss_type
        
        # Set up loss function (KL divergence as per paper)
        if loss_type == "kl_divergence" or loss_type == "kl":
            # Use PyTorch's optimized KL divergence (expects log-probabilities)
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. CGD only supports 'kl_divergence' or 'kl'")
        
        # Create model with classification head for probability distribution
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_bins)
        
        # Initialize CGD module
        self.cgd = CircularGaussianDistribution(num_bins=num_bins, sigma=sigma)
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @classmethod
    def try_load(cls, checkpoint_path=None, **kwargs):
        """Try to load model from checkpoint, fallback to new model if loading fails"""
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading model from checkpoint: {checkpoint_path}")
                model = cls.load_from_checkpoint(checkpoint_path, **kwargs)
                logger.info("Model loaded successfully from checkpoint")
                return model
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}")
                raise FileNotFoundError("Checkpoint file not found")

        except (RuntimeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Creating new model with pretrained weights")
            model = cls(**kwargs)
            logger.info("New model created successfully")
            return model

    def load_pretrained_weights(self, checkpoint_path):
        raise NotImplementedError("Pretrained weights loading not implemented for classification model")


    def calculate_angular_mae_from_distribution(self, y_true_angles: torch.Tensor, pred_distributions: torch.Tensor) -> torch.Tensor:
        """Calculate angular MAE from predicted distributions and true angles"""
        # Extract angles from predicted distributions
        pred_angles = self.cgd.distribution_to_angle(pred_distributions, method=self.inference_method)
        
        # Calculate angular distance (shorter path around circle)
        angular_errors = torch.abs(pred_angles - y_true_angles)
        angular_errors = torch.minimum(angular_errors, 360 - angular_errors)
        return torch.mean(angular_errors)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass returning probability distribution over angles
        
        Args:
            x: Input tensor [B, C, H, W]
            return_logits: If True, return raw logits; if False, return probabilities
            
        Returns:
            distributions: Probability distributions over angles [B, num_bins] or raw logits
        """
        # Forward pass through backbone with built-in classification head
        logits = self.model(x)  # Shape: [batch_size, num_bins]
        
        if return_logits:
            return logits
        else:
            # Apply softmax to get probability distribution (for inference)
            distributions = F.softmax(logits, dim=1)  # Shape: [batch_size, num_bins]
            return distributions

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        x, y = batch
        
        # Convert angle targets to probability distributions
        target_distributions = self.cgd.angle_to_distribution(y)
        
        # Forward pass - get raw logits for loss computation
        logits = self(x, return_logits=True)
        
        # Compute KL divergence loss using log_softmax for numerical stability
        log_pred = F.log_softmax(logits, dim=1)
        loss = self.loss_fn(log_pred, target_distributions)
        
        # Get probability distributions for metrics (softmax of logits)
        pred_distributions = F.softmax(logits, dim=1)
        
        # Calculate angular MAE for monitoring
        angular_mae = self.calculate_angular_mae_from_distribution(y, pred_distributions)
        
        # Calculate distribution uncertainty for monitoring
        uncertainty = self.cgd.get_distribution_uncertainty(pred_distributions)
        mean_uncertainty = torch.mean(uncertainty)
        
        # Extract predicted angles for logging
        pred_angles = self.cgd.distribution_to_angle(pred_distributions, method=self.inference_method)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_uncertainty', mean_uncertainty, on_step=True, on_epoch=True)
        self.log('train_pred_angle_mean', torch.mean(pred_angles), on_step=True, on_epoch=True)
        self.log('train_target_angle_mean', torch.mean(y), on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch
        
        # Convert angle targets to probability distributions
        target_distributions = self.cgd.angle_to_distribution(y)
        
        # Forward pass - get raw logits for loss computation
        logits = self(x, return_logits=True)
        
        # Compute KL divergence loss using log_softmax for numerical stability
        log_pred = F.log_softmax(logits, dim=1)
        loss = self.loss_fn(log_pred, target_distributions)
        
        # Get probability distributions for metrics (softmax of logits)
        pred_distributions = F.softmax(logits, dim=1)
        
        # Extract predicted angles from distributions
        pred_angles = self.cgd.distribution_to_angle(pred_distributions, method=self.inference_method)
        
        # Calculate comprehensive metrics
        val_metrics = compute_validation_metrics(pred_angles, y)
        
        # Calculate distribution uncertainty
        uncertainty = self.cgd.get_distribution_uncertainty(pred_distributions)
        mean_uncertainty = torch.mean(uncertainty)
        
        # Log all metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_uncertainty', mean_uncertainty, on_step=False, on_epoch=True)
        for metric_name, metric_value in val_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, 
                    prog_bar=(metric_name == 'val_mae_deg'))  # Only show MAE in progress bar


    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x, y = batch
        
        # Convert angle targets to probability distributions
        target_distributions = self.cgd.angle_to_distribution(y)
        
        # Forward pass - get raw logits for loss computation
        logits = self(x, return_logits=True)
        
        # Compute KL divergence loss using log_softmax for numerical stability
        log_pred = F.log_softmax(logits, dim=1)
        loss = self.loss_fn(log_pred, target_distributions)
        
        # Get probability distributions for metrics (softmax of logits)
        pred_distributions = F.softmax(logits, dim=1)
        
        # Extract predicted angles from distributions
        pred_angles = self.cgd.distribution_to_angle(pred_distributions, method=self.inference_method)
        
        # Calculate comprehensive test metrics
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in test_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers"""
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # For overfitting mode, disable scheduler
        is_overfitting = (hasattr(self.trainer, 'overfit_batches') and self.trainer.overfit_batches > 0)

        if is_overfitting:
            return {"optimizer": optimizer}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_mae_deg", "frequency": 1}
            }

    def setup(self, stage=None):
        """Setup datasets"""
        if stage == "fit" or stage is None:
            enable_overfitting = (hasattr(self.trainer, 'overfit_batches') and
                                  self.trainer.overfit_batches > 0)

            self.train_dataset, self.val_dataset = RotationDataset.create_datasets(
                image_dir=self.train_dir,
                validation_split=self.validation_split,
                random_seed=self.random_seed,
                image_size=self.image_size,
                enable_overfitting=enable_overfitting,
                model_name=self.hparams.model_name
            )
        elif stage == "test":
            # Setup test dataset
            if self.test_dir and os.path.exists(self.test_dir):
                # Get all image files in test directory
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                test_image_paths = []
                for ext in image_extensions:
                    test_image_paths.extend(glob.glob(os.path.join(self.test_dir, f"*{ext}")))
                    test_image_paths.extend(glob.glob(os.path.join(self.test_dir, f"*{ext.upper()}")))
                
                if test_image_paths:
                    self.test_dataset = RotationDataset(
                        image_paths=test_image_paths,
                        image_size=self.image_size,
                        mode="test",
                        model_name=self.hparams.model_name,
                        test_rotation_range=self.test_rotation_range,
                        test_random_seed=self.test_random_seed
                    )
                    logger.info(f"Created test dataset with {len(test_image_paths)} images")
                    logger.info(f"Test rotation range: ±{self.test_rotation_range/2:.1f}° (seed={self.test_random_seed})")
                else:
                    logger.warning(f"No test images found in {self.test_dir}")
                    self.test_dataset = None
            else:
                logger.warning(f"Test directory not found: {self.test_dir}")
                self.test_dataset = None

    def train_dataloader(self):        
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=12, persistent_workers=True, pin_memory=True,
            prefetch_factor=4, drop_last=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True,
            prefetch_factor=2, drop_last=False
        )
    
    def test_dataloader(self):
        if hasattr(self, 'test_dataset') and self.test_dataset is not None:
            return DataLoader(
                self.test_dataset, batch_size=self.batch_size,
                shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True,
                prefetch_factor=2, drop_last=False
            )
        else:
            logger.warning("Test dataset not available")
            return None

    def predict_angle(self, image_path: str) -> float:
        """Detect the current orientation angle of an image"""
        self.eval()
        image = Image.open(image_path).convert('RGB')

        # Use TIMM transforms if possible
        try:
            # Get model-specific data configuration
            data_config = timm.data.resolve_model_data_config(self.hparams.model_name)
            transform = timm.data.create_transform(**data_config, is_training=False)
        except:
            # Fallback to standard transforms
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            pred_distributions = self(image_tensor)
            angle = self.cgd.distribution_to_angle(pred_distributions, method=self.inference_method).item()

        return angle

