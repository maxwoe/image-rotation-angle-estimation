#!/usr/bin/env python3
"""
Direct Angle Prediction Approach
================================

This approach uses a single output neuron to predict angles directly.
The model outputs a single value representing the orientation angle in degrees.
Uses traditional angular loss functions that handle the circular nature of angles.

Model output: Single value (angle in degrees)
Advantages: Simple output, easy to interpret
Disadvantages: Must handle angle wrapping in loss functions
"""

from regression_heads import ConfigurableRegressionHead
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import timm
from PIL import Image
import numpy as np
import os
import glob
from loguru import logger
# Removed config.py imports - parameters now explicitly required
from data_loader import RotationDataset
from metrics import compute_validation_metrics, compute_test_metrics


class DirectAngleEstimation(pl.LightningModule):
    """
    Direct angle prediction model with single output neuron.

    This model directly predicts the orientation angle using a single output.
    Requires careful handling of circular angle properties in loss functions.
    """

    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 loss_type="mae", use_custom_head: bool = False, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        if not use_custom_head:
            # Use timm's full model directly
            self.model = timm.create_model(model_name, pretrained=True, num_classes=1)
            self.head = None  # Not needed, backbone handles regression
        else:
            # Feature extractor + custom head
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg') # global_pool='avg'
            num_features = self.backbone.num_features
    
            # Dynamically determine actual feature dimensions
            # Some models (like EfficientViT) have incorrect num_features in timm
            with torch.no_grad():
                # Use image_size if specified, otherwise default to 224
                input_size = image_size if image_size else 224
                dummy_input = torch.randn(1, 3, input_size, input_size)
                dummy_features = self.backbone(dummy_input)
                num_features = dummy_features.shape[-1]
            
            self.head = ConfigurableRegressionHead(
                in_features=num_features,
                out_features=1,
                mlp_dims=[num_features//2, num_features//4],
                dropout_rate=0.1,  # Lower dropout for unit vector stability
                normalization="layer"  # "batch" Batch norm often works well for unit vectors, # Layer norm often better for precision
            )
            logger.info(f"Head: {self.head}")
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.image_size = image_size
        self.loss_type = loss_type

        # Choose loss function
        if loss_type == "mae":
            self.loss_fn = self.angular_mae_loss
        elif loss_type == "mse":
            self.loss_fn = self.angular_mse_loss
        elif loss_type == "smooth_l1":
            self.loss_fn = self.angular_smooth_l1_loss
        elif loss_type == "smooth_l1_sin":
            self.loss_fn = self.angular_smooth_l1_sin_loss
        elif loss_type == "smooth_l1_cos":
            self.loss_fn = self.angular_smooth_l1_cos_loss
        elif loss_type == "cosine":
            self.loss_fn = self.cosine_loss
        elif loss_type == "chord":
            self.loss_fn = self.chord_loss
        elif loss_type == "von_mises":
            self.loss_fn = self.von_mises_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}.")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _normalize_angles(self, angles):
        """Normalize angles to [0, 360) range"""
        return torch.remainder(angles, 360.0)

    def _degrees_to_radians(self, angles):
        """Convert angles from degrees to radians"""
        return angles * torch.pi / 180.0

    def _angular_distance(self, pred_angles, true_angles):
        """Calculate angular distance (shorter path around circle)"""
        pred_norm = self._normalize_angles(pred_angles)
        true_norm = self._normalize_angles(true_angles)
        
        error1 = torch.abs(pred_norm - true_norm)
        error2 = 360 - error1
        return torch.min(error1, error2)

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

    def angular_mae_loss(self, y_pred, y_true):
        """
        Angular MAE loss for direct angle prediction
        Handles circular nature of angles (0° = 360°)
        """
        error = self._angular_distance(y_pred, y_true)
        return torch.mean(error)

    def angular_mse_loss(self, y_pred, y_true):
        """
        Angular MSE loss for direct angle prediction
        Handles circular nature of angles properly
        """
        error = self._angular_distance(y_pred, y_true)
        return torch.mean(error ** 2)

    def angular_smooth_l1_loss(self, y_pred, y_true, beta=1.0):
        """
        Angular Smooth L1 loss (Huber loss) for direct angle prediction
        Combines benefits of L1 (robust to outliers) and L2 (smooth gradients)
        
        Args:
            beta: Threshold for switching between L1 and L2 behavior
                  Lower values = more L1-like, Higher values = more L2-like
        """
        angular_error = self._angular_distance(y_pred, y_true)
        
        # Apply smooth L1 loss
        # For |error| < beta: 0.5 * error^2 / beta
        # For |error| >= beta: |error| - 0.5 * beta
        smooth_l1 = torch.where(
            angular_error < beta,
            0.5 * angular_error ** 2 / beta,
            angular_error - 0.5 * beta
        )
        
        return torch.mean(smooth_l1)

    def angular_smooth_l1_sin_loss(self, y_pred, y_true, beta=1.0):
        """
        Smooth L1 loss applied to sin((θ_p - θ_t) / 2)
        L_θ = SmoothL1(sin((θ_p - θ_t) / 2))

        Uses half-angle sine: sin(Δ/2) is monotonically increasing over [0°, 180°],
        equalling 0 only at 0° error and 1 at 180° error. This avoids the zero-gradient
        trap of sin(Δ), which is also zero at 180° error (the worst case).
        """
        # Convert to radians
        y_true_rad = self._degrees_to_radians(y_true)
        y_pred_rad = self._degrees_to_radians(y_pred)

        # Half-angle sine: monotonically increasing in [0°, 180°]
        angle_diff = y_pred_rad - y_true_rad
        sin_diff = torch.sin(angle_diff / 2)
        
        # Apply smooth L1 to sin difference
        smooth_l1 = torch.where(
            torch.abs(sin_diff) < beta,
            0.5 * sin_diff ** 2 / beta,
            torch.abs(sin_diff) - 0.5 * beta
        )
        
        return torch.mean(smooth_l1)

    def angular_smooth_l1_cos_loss(self, y_pred, y_true, beta=1.0):
        """
        Smooth L1 loss applied to (1 - cos(θ_p - θ_t))
        Better for image orientation: L_θ = SmoothL1(1 - cos(θ_p - θ_t))
        
        Properties:
        - 0° error: cos(0) = 1 → 1-cos = 0 (perfect)
        - 90° error: cos(90°) = 0 → 1-cos = 1 (medium)  
        - 180° error: cos(180°) = -1 → 1-cos = 2 (maximum)
        """
        # Convert to radians
        y_true_rad = self._degrees_to_radians(y_true)
        y_pred_rad = self._degrees_to_radians(y_pred)
        
        # Calculate 1 - cos of angle difference
        angle_diff = y_pred_rad - y_true_rad
        cos_diff = torch.cos(angle_diff)
        error = 1 - cos_diff  # Range [0, 2]
        
        # Apply smooth L1 to the error
        smooth_l1 = torch.where(
            error < beta,
            0.5 * error ** 2 / beta,
            error - 0.5 * beta
        )
        
        return torch.mean(smooth_l1)

    def cosine_loss(self, y_pred, y_true):
        """
        Cosine loss: 1 - cos(Δ)
        Simple circular loss based on cosine of angle difference
        """
        # Convert angles to radians
        y_true_rad = self._degrees_to_radians(y_true)
        y_pred_rad = self._degrees_to_radians(y_pred)

        # Direct cosine of angle difference
        angle_diff = y_true_rad - y_pred_rad

        return torch.mean(1 - torch.cos(angle_diff))

    def chord_loss(self, y_pred, y_true):
        """
        Chord loss: sqrt(2*(1 - cos Δ))
        Circular analog of absolute difference
        """
        y_true_rad = self._degrees_to_radians(y_true)
        y_pred_rad = self._degrees_to_radians(y_pred)
        angle_diff = y_true_rad - y_pred_rad
        return torch.mean(torch.sqrt(2*(1 - torch.cos(angle_diff))))

    def von_mises_loss(self, y_pred, y_true, kappa=1.0):
        """
        Von Mises loss for direct angle prediction
        Based on von Mises distribution (circular equivalent of Gaussian)
        """
        # Convert to radians
        y_true_rad = self._degrees_to_radians(y_true)
        y_pred_rad = self._degrees_to_radians(y_pred)

        # Von Mises negative log-likelihood
        angle_diff = y_true_rad - y_pred_rad
        nll = -kappa * torch.cos(angle_diff)

        return torch.mean(nll)

    def forward(self, x):
        if self.head is not None:
            # Custom head path
            feats = self.backbone(x)
            logits = self.head(feats)
        else:
            # Direct model path
            logits = self.model(x)
        
        return logits.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Calculate MAE in degrees for comparison
        angular_mae = self.angular_mae_loss(y_hat, y)

        # Log loss and human-interpretable metrics (match unit vector naming)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Normalize predicted angles
        pred_angles = self._normalize_angles(y_hat)
        
        # Calculate comprehensive metrics
        val_metrics = compute_validation_metrics(pred_angles, y)
        
        # Log all metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in val_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, 
                    prog_bar=(metric_name == 'val_mae_deg'))  # Only show MAE in progress bar
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)

        # Normalize predicted angles
        pred_angles = self._normalize_angles(y_hat)
        
        # Calculate comprehensive test metrics
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in test_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate) # is used by Maji and Bose
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # For overfitting mode, disable scheduler to maintain aggressive learning rate
        # For normal training, use scheduler for learning rate decay
        is_overfitting = (hasattr(self.trainer, 'overfit_batches') and self.trainer.overfit_batches > 0)

        if is_overfitting:
            # No scheduler for overfitting - maintain constant LR for aggressive optimization
            return {"optimizer": optimizer}
        else:
            # Use scheduler for normal training
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-5)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1}
            }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            enable_overfitting = (hasattr(self.trainer, 'overfit_batches') and self.trainer.overfit_batches > 0)

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

    def predict_angle(self, image) -> float:
        """Detect the current orientation angle of an image.

        Args:
            image: PIL Image, numpy array, or file path string.
                   Note: JPEG file paths will go through lossy decompression which
                   can degrade accuracy. For best results, pass PIL/numpy directly.
        """
        self.eval()

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, numpy array, or file path, got {type(image)}")
        else:
            image = image.convert('RGB')

        try:
            import timm.data
            data_config = timm.data.resolve_model_data_config(self.hparams.model_name)
            # Match DataLoader settings exactly
            data_config['crop_pct'] = 1.0
            data_config['input_size'] = (3, self.image_size, self.image_size)
            transform = timm.data.create_transform(**data_config, is_training=False)
        except Exception:
            transform = transforms.Compose([
                transforms.Resize((self.image_size or 224, self.image_size or 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            angle = self(image_tensor).item()
            angle = self._normalize_angles(torch.tensor(angle)).item()

        return angle

