#!/usr/bin/env python3
"""
Classification-Based Approach for Orientation Detection
======================================================

This approach discretizes angles into bins and treats orientation detection as a 
classification problem. The model outputs class probabilities for each angle bin.

Model output: Probability distribution over angle classes
Advantages: Can capture multi-modal distributions, robust training dynamics
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
        elif loss_type == "focal":
            self.loss_fn = self.focal_loss
        elif loss_type == "smooth_cross_entropy":
            self.loss_fn = self.smooth_cross_entropy_loss
        elif loss_type == "weighted_cross_entropy":
            self.loss_fn = self.weighted_cross_entropy_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

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

    def focal_loss(self, y_true_class, y_pred_logits, alpha=1.0, gamma=2.0):
        """
        Focal loss for handling class imbalance
        Focuses on hard examples by down-weighting easy examples
        """
        ce_loss = F.cross_entropy(y_pred_logits, y_true_class, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return torch.mean(focal_loss)

    def smooth_cross_entropy_loss(self, y_true_class, y_pred_logits):
        """
        Cross-entropy with circular label smoothing
        Smooths labels with neighboring angle classes
        """
        batch_size = y_true_class.size(0)
        
        # Create soft labels with circular smoothing
        soft_labels = torch.zeros(batch_size, self.num_classes, device=y_true_class.device)
        
        for i in range(batch_size):
            true_class = y_true_class[i].item()
            
            # Main class gets most probability
            soft_labels[i, true_class] = 0.7
            
            # Neighboring classes get some probability (circular)
            prev_class = (true_class - 1) % self.num_classes
            next_class = (true_class + 1) % self.num_classes
            soft_labels[i, prev_class] += 0.15
            soft_labels[i, next_class] += 0.15
        
        # Compute loss with soft labels
        log_probs = F.log_softmax(y_pred_logits, dim=1)
        loss = -torch.sum(soft_labels * log_probs, dim=1)
        return torch.mean(loss)

    def weighted_cross_entropy_loss(self, y_true_class, y_pred_logits):
        """
        Weighted cross-entropy with circular weights
        Penalizes errors more heavily when prediction is far from true class
        """
        batch_size = y_true_class.size(0)
        
        # Get predicted classes
        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        
        # Calculate circular distance weights
        weights = torch.ones_like(y_true_class, dtype=torch.float)
        
        for i in range(batch_size):
            true_class = y_true_class[i].item()
            pred_class = y_pred_class[i].item()
            
            # Calculate circular distance
            dist1 = abs(pred_class - true_class)
            dist2 = self.num_classes - dist1
            circular_dist = min(dist1, dist2)
            
            # Weight increases with distance
            weights[i] = 1.0 + (circular_dist / self.num_classes) * 2.0
        
        # Apply weighted cross-entropy
        ce_loss = F.cross_entropy(y_pred_logits, y_true_class, reduction='none')
        weighted_loss = weights * ce_loss
        return torch.mean(weighted_loss)

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

    def calculate_expected_angle(self, y_pred_logits):
        """
        Calculate expected angle using circular statistics
        More sophisticated than argmax - uses full probability distribution
        """
        probs = F.softmax(y_pred_logits, dim=1)
        
        # Convert classes to angles (radians)
        class_indices = torch.arange(self.num_classes, device=probs.device)
        angles_rad = (class_indices.float() + 0.5) * self.bin_size * torch.pi / 180.0
        
        # Calculate expected cos and sin
        cos_angles = torch.cos(angles_rad)
        sin_angles = torch.sin(angles_rad)
        
        expected_cos = torch.sum(probs * cos_angles, dim=1)
        expected_sin = torch.sum(probs * sin_angles, dim=1)
        
        # Convert back to degrees
        expected_angles = torch.atan2(expected_sin, expected_cos) * 180.0 / torch.pi
        expected_angles = expected_angles % 360
        
        return expected_angles

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
                optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
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
            
            # Method 1: Argmax prediction
            predicted_class = torch.argmax(logits, dim=1)
            angle_argmax = self.class_to_angle(predicted_class).item()
            
            # Method 2: Expected angle (circular statistics)
            angle_expected = self.calculate_expected_angle(logits).item()
            
            # Use expected angle for better precision
            angle = angle_expected

        return angle

