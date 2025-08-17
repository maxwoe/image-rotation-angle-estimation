#!/usr/bin/env python3
"""
Classification-Based Approach for Orientation Detection
======================================================

This approach discretizes angles into bins and treats orientation detection as a 
classification problem. The model outputs class probabilities for each angle bin.

Includes state-of-the-art implementations:
- Circular Smooth Label (CSL) from Yang et al. ECCV 2020
- Dense Coded Labels (DCL) from Yang et al. CVPR 2021

Model output: Probability distribution over angle classes (or bit codes for DCL)
Advantages: Can capture multi-modal distributions, robust training dynamics, handles boundaries
Disadvantages: Discretization artifacts, requires more parameters
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import timm
from PIL import Image
import os
import glob
from loguru import logger
# Removed config.py imports - parameters now explicitly required
from data_loader import RotationDataset
from metrics import compute_validation_metrics, compute_test_metrics


class ClassificationAngleDetection(pl.LightningModule):
    """
    Classification-based angle detection model.
    
    Discretizes the 360° angle space into bins and treats orientation detection
    as a multi-class classification problem. Each bin represents a range of angles.
    """

    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 num_classes=360, loss_type="cross_entropy", label_smoothing=0.0, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        # Classification setup
        self.num_classes = num_classes  # Number of angle bins
        self.bin_size = 360.0 / num_classes  # Degrees per bin
        self.label_smoothing = label_smoothing
        
        # Model with classification head
        self.model = timm.create_model(model_name, pretrained=True, num_classes=self.num_classes)
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.image_size = image_size
        self.loss_type = loss_type

        # Choose loss function
        if loss_type == "cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        elif loss_type == "csl":
            # CSL with best performing configuration (Gaussian window, radius=6, sigma=2.0)
            self.loss_fn = lambda y_pred, y_true: self.csl_loss(y_true, y_pred, 'gaussian', 6, 2.0)
        elif loss_type == "dcl":
            # DCL with best performing configuration (Binary Coded Labels)
            self.loss_fn = lambda y_pred, y_true: self.dcl_loss(y_true, y_pred, 'bcl')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: 'cross_entropy', 'csl', 'dcl'")

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

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Creating new model with pretrained weights")
            model = cls(**kwargs)
            logger.info("New model created successfully")
            return model

    def load_pretrained_weights(self, checkpoint_path):
        raise NotImplementedError("Pretrained weights loading not implemented for classification model")

    def angle_to_class(self, angle):
        """Convert continuous angle to discrete class index"""
        # Normalize angle to [0, 360)
        angle = angle % 360
        # Convert to class index
        class_idx = (angle / self.bin_size).long()
        # Handle edge case where angle = 360 (should map to class 0)
        class_idx = class_idx % self.num_classes
        return class_idx

    def class_to_angle(self, class_idx):
        """Convert class index to angle (center of bin)"""
        return (class_idx.float() + 0.5) * self.bin_size


    def csl_loss(self, y_true_class, y_pred_logits, window_type='gaussian', radius=6, sigma=2.0):
        """
        Circular Smooth Label (CSL) loss from Yang et al. ECCV 2020
        
        Implements sophisticated circular label smoothing with various window functions
        to address boundary discontinuity in angle classification.
        
        Args:
            y_true_class: Ground truth class indices [B]
            y_pred_logits: Predicted logits [B, num_classes]
            window_type: Type of smoothing window ('impulse', 'rectangular', 'triangular', 'gaussian')
            radius: Smoothing window radius in degrees (default: 6, best from paper)
            sigma: Standard deviation for Gaussian window (default: 2.0)
        """
        batch_size = y_true_class.size(0)
        device = y_true_class.device
        
        # Convert radius from degrees to class bins
        radius_bins = int(radius * self.num_classes / 360.0)
        radius_bins = max(1, radius_bins)  # At least 1 bin
        
        # Create soft labels with CSL smoothing
        soft_labels = torch.zeros(batch_size, self.num_classes, device=device)
        
        # Create class indices for vectorized computation
        class_indices = torch.arange(self.num_classes, device=device).float()
        
        for i in range(batch_size):
            # Handle case where y_true_class might have wrong shape
            if y_true_class.dim() == 1:
                true_class = y_true_class[i].item()
            else:
                # If tensor is not 1D, try to extract the scalar value
                true_class = y_true_class[i].flatten()[0].item()
            
            # Calculate circular distances to all classes
            dist_forward = (class_indices - true_class) % self.num_classes
            dist_backward = (true_class - class_indices) % self.num_classes
            circular_dist = torch.minimum(dist_forward, dist_backward)
            
            # Apply window function
            if window_type == 'impulse':
                # Delta function - only true class gets weight
                weights = (circular_dist == 0).float()
                
            elif window_type == 'rectangular':
                # Uniform within radius
                weights = (circular_dist <= radius_bins).float()
                
            elif window_type == 'triangular':
                # Linear decay from center
                weights = torch.clamp(1.0 - circular_dist / radius_bins, min=0.0)
                
            elif window_type == 'gaussian':
                # Gaussian decay (best performing according to paper)
                weights = torch.exp(-(circular_dist ** 2) / (2 * sigma ** 2))
                weights = weights * (circular_dist <= radius_bins).float()
                
            else:
                raise ValueError(f"Unknown window_type: {window_type}")
            
            # Normalize to create probability distribution
            weights = weights / (torch.sum(weights) + 1e-8)
            soft_labels[i] = weights
        
        # Compute loss with soft labels
        log_probs = F.log_softmax(y_pred_logits, dim=1)
        loss = -torch.sum(soft_labels * log_probs, dim=1)
        return torch.mean(loss)
    
    def dcl_loss(self, y_true_class, y_pred_logits, coding_type='bcl'):
        """
        Dense Coded Labels (DCL) inspired loss from Yang et al. CVPR 2021
        
        Implements angle-aware loss weighting based on Gray coding principles to achieve
        better handling of adjacent angle relationships and faster training.
        
        Note: This is an adapted version that works with standard classification heads
        rather than requiring separate bit-wise outputs.
        
        Args:
            y_true_class: Ground truth class indices [B]
            y_pred_logits: Predicted logits [B, num_classes]
            coding_type: Type of dense coding ('bcl' for Binary, 'gcl' for Gray)
        """
        batch_size = y_true_class.size(0)
        device = y_true_class.device
        
        # For compatibility with existing architecture, we implement DCL principles
        # by creating soft labels based on Gray code distances
        
        if coding_type == 'gcl':
            # Gray Coded Labels: Create soft labels based on Hamming distances in Gray code space
            soft_labels = self._create_gray_soft_labels(y_true_class, device)
        else:
            # Binary Coded Labels: Create soft labels based on binary Hamming distances  
            soft_labels = self._create_binary_soft_labels(y_true_class, device)
        
        # Compute cross-entropy with soft labels
        log_probs = F.log_softmax(y_pred_logits, dim=1)
        loss = -torch.sum(soft_labels * log_probs, dim=1)
        return torch.mean(loss)
    
    def _create_gray_soft_labels(self, y_true_class, device):
        """Create soft labels based on Gray code Hamming distances"""
        batch_size = y_true_class.size(0)
        soft_labels = torch.zeros(batch_size, self.num_classes, device=device)
        
        # Calculate Gray code for each class
        def binary_to_gray(n):
            return n ^ (n >> 1)
        
        def hamming_distance(a, b, num_bits=8):
            """Calculate Hamming distance between two Gray codes"""
            return bin(a ^ b).count('1')
        
        for i in range(batch_size):
            # Handle case where y_true_class might have wrong shape
            if y_true_class.dim() == 1:
                true_class = y_true_class[i].item()
            else:
                # If tensor is not 1D, try to extract the scalar value
                true_class = y_true_class[i].flatten()[0].item()
            true_gray = binary_to_gray(true_class)
            
            # Calculate weights based on Gray code Hamming distances
            for c in range(self.num_classes):
                class_gray = binary_to_gray(c)
                hamming_dist = hamming_distance(true_gray, class_gray)
                
                # Closer Gray codes get higher weights (exponential decay)
                weight = torch.exp(torch.tensor(-hamming_dist * 0.5, device=device))  # Tunable decay factor
                soft_labels[i, c] = weight
            
            # Normalize to probability distribution
            soft_labels[i] = soft_labels[i] / (torch.sum(soft_labels[i]) + 1e-8)
        
        return soft_labels
    
    def _create_binary_soft_labels(self, y_true_class, device):
        """Create soft labels based on binary Hamming distances"""
        batch_size = y_true_class.size(0)
        soft_labels = torch.zeros(batch_size, self.num_classes, device=device)
        
        def hamming_distance(a, b):
            """Calculate Hamming distance between two binary numbers"""
            return bin(a ^ b).count('1')
        
        for i in range(batch_size):
            # Handle case where y_true_class might have wrong shape
            if y_true_class.dim() == 1:
                true_class = y_true_class[i].item()
            else:
                # If tensor is not 1D, try to extract the scalar value
                true_class = y_true_class[i].flatten()[0].item()
            
            # Calculate weights based on binary Hamming distances
            for c in range(self.num_classes):
                hamming_dist = hamming_distance(true_class, c)
                
                # Closer binary representations get higher weights
                weight = torch.exp(torch.tensor(-hamming_dist * 0.3, device=device))  # Tunable decay factor
                soft_labels[i, c] = weight
            
            # Normalize to probability distribution
            soft_labels[i] = soft_labels[i] / (torch.sum(soft_labels[i]) + 1e-8)
        
        return soft_labels


    def calculate_angular_mae_from_classes(self, y_true_angles, y_pred_logits):
        """Calculate angular MAE from class predictions and true angles"""
        # Get predicted class probabilities
        y_pred_probs = F.softmax(y_pred_logits, dim=1)
        
        # Method 1: Use argmax (discrete prediction)
        y_pred_classes = torch.argmax(y_pred_probs, dim=1)
        pred_angles = self.class_to_angle(y_pred_classes)
        
        # Calculate angular distance (shorter path around circle)
        angular_errors = torch.abs(pred_angles - y_true_angles)
        angular_errors = torch.minimum(angular_errors, 360 - angular_errors)
        return torch.mean(angular_errors)


    def forward(self, x):
        # Classification output (logits over angle classes)
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Convert continuous angles to class labels
        y_class = self.angle_to_class(y)
        
        y_hat_logits = self(x)
        loss = self.loss_fn(y_hat_logits, y_class)

        # Calculate MAE in degrees for comparison
        angular_mae = self.calculate_angular_mae_from_classes(y, y_hat_logits)

        # Calculate accuracy
        y_pred_class = torch.argmax(y_hat_logits, dim=1)
        accuracy = torch.mean((y_pred_class == y_class).float())

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True)
        
        # Log predicted vs target for monitoring
        pred_angles = self.class_to_angle(y_pred_class)
        self.log('train_pred_angle_mean', torch.mean(pred_angles), on_step=True, on_epoch=True)
        self.log('train_target_angle_mean', torch.mean(y), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Convert continuous angles to class labels
        y_class = self.angle_to_class(y)
        
        y_hat_logits = self(x)
        loss = self.loss_fn(y_hat_logits, y_class)

        # Calculate predicted angles from class probabilities
        y_pred_probs = F.softmax(y_hat_logits, dim=1)
        y_pred_classes = torch.argmax(y_pred_probs, dim=1)
        pred_angles = self.class_to_angle(y_pred_classes)
        
        # Calculate comprehensive metrics
        val_metrics = compute_validation_metrics(pred_angles, y)
        
        # Calculate classification accuracy
        accuracy = torch.mean((y_pred_classes == y_class).float())
        
        # Log all metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in val_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, 
                    prog_bar=(metric_name == 'val_mae_deg'))  # Only show MAE in progress bar
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Convert continuous angles to class labels
        y_class = self.angle_to_class(y)
        
        y_hat_logits = self(x)
        loss = self.loss_fn(y_hat_logits, y_class)

        # Calculate predicted angles from class probabilities
        y_pred_probs = F.softmax(y_hat_logits, dim=1)
        y_pred_classes = torch.argmax(y_pred_probs, dim=1)
        pred_angles = self.class_to_angle(y_pred_classes)
        
        # Calculate comprehensive test metrics
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Calculate classification accuracy
        accuracy = torch.mean((y_pred_classes == y_class).float())
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_accuracy', accuracy, on_step=False, on_epoch=True)
        for metric_name, metric_value in test_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # For overfitting mode, disable scheduler
        is_overfitting = (hasattr(self.trainer, 'overfit_batches') and self.trainer.overfit_batches > 0)

        if is_overfitting:
            return {"optimizer": optimizer}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1}
            }

    def setup(self, stage=None):
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

    def predict_angle(self, image_path):
        """Detect the current orientation angle of an image"""
        self.eval()
        image = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self(image_tensor)
            
            # Use argmax for consistency with evaluation metrics
            predicted_class = torch.argmax(logits, dim=1)
            angle = self.class_to_angle(predicted_class).item()

        return angle

