#!/usr/bin/env python3
"""
Phase Shift Coder (PSC) Approach for Orientation Estimation
==========================================================

This approach implements Phase Shift Coder from the paper "On Boundary Discontinuity 
in Angle Regression Based Arbitrary Oriented Object Estimation".

PSC addresses boundary discontinuity in angle regression using continuous phase-shifting codes.
Mathematical formulation: M = {mn = cos(ωθ + 2nπ/Nstep)} for encoding
Decoding: θ = -(1/ω)arctan(Σsin/Σcos)

Model output: Multiple phase-shifted cosine values
Advantages: Boundary discontinuity-free, continuous representation, better gradient flow
Disadvantages: More complex decoding, requires careful frequency selection
"""

from regression_heads import ConfigurableRegressionHead
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import timm
import timm.data
from PIL import Image
import numpy as np
import os
import glob
from loguru import logger
# Removed config.py imports - parameters now explicitly required
from data_loader import RotationDataset
from metrics import compute_validation_metrics, compute_test_metrics


class PSCAngleEstimation(pl.LightningModule):
    """
    Phase Shift Coder (PSC) angle estimation model.
    
    Uses phase-shifting codes to encode angles without boundary discontinuity:
    M = {mn = cos(ωθ + 2nπ/Nstep)} for n = 0, 1, ..., Nstep-1
    
    This provides continuous representation that eliminates the boundary discontinuity
    problem in angle regression.
    """
    
    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 loss_type="mae", num_phases=3, omega=1.0, use_custom_head=True, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        self.num_phases = num_phases  # Nstep in paper
        self.omega = omega  # Frequency parameter ω
        
        logger.info(f"PSC: {num_phases} phases, ω={omega}")

        if not use_custom_head:
            # Use timm's full model directly
            self.model = timm.create_model(model_name, pretrained=True, num_classes=num_phases)
            self.head = None
        else:
            # Feature extractor + custom head
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
            
            num_features = self.backbone.num_features
            # Dynamically determine actual feature dimensions
            with torch.no_grad():
                input_size = image_size if image_size else 224
                dummy_input = torch.randn(1, 3, input_size, input_size)
                dummy_features = self.backbone(dummy_input)
                num_features = dummy_features.shape[-1]
            
            self.head = ConfigurableRegressionHead(
                in_features=num_features,
                out_features=num_phases,
                mlp_dims=[num_features // 2, num_features // 4, num_features // 8],  # hidden layers
                act_layer=nn.ReLU, # nn.ReLU nn.GELU nn.SiLU
                dropout_rate=0.2,
                normalization="layer",
                # final_activation=nn.Tanh  # Final activation to ensure outputs are in [-1, 1]
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
            self.loss_fn = nn.L1Loss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()
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

        except (RuntimeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Creating new model with pretrained weights")
            model = cls(**kwargs)
            logger.info("New model created successfully")
            return model

    def load_pretrained_weights(self, checkpoint_path):
        """Load model weights from a checkpoint for fine-tuning (fresh optimizer/scheduler)."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys when loading pretrained weights: {unexpected}")
        logger.info(f"Loaded pretrained weights from {checkpoint_path}")

    def angle_to_psc(self, angle):
        """
        Convert angle (degrees) to Phase Shift Coder representation
        
        PSC encoding: M = {mn = cos(ωθ + 2nπ/Nstep)} for n = 0, 1, ..., Nstep-1
        
        Args:
            angle: Angles in degrees, shape [B]
            
        Returns:
            psc_codes: PSC encoding, shape [B, num_phases]
        """
        angle_rad = angle * torch.pi / 180.0
        
        # Single frequency PSC
        codes = []
        for n in range(self.num_phases):
            phase_shift = 2 * torch.pi * n / self.num_phases
            codes.append(torch.cos(self.omega * angle_rad + phase_shift))
        
        return torch.stack(codes, dim=1)

    def psc_to_angle(self, psc_codes):
        """
        Convert PSC representation back to angle
        
        Args:
            psc_codes: PSC codes, shape [B, num_phases]
            
        Returns:
            angles: Decoded angles in degrees, shape [B]
        """
        return self._decode_single_frequency(psc_codes, self.omega)
    
    def _decode_single_frequency(self, codes, omega):
        """
        Decode PSC codes for single frequency
        
        Args:
            codes: PSC codes for single frequency, shape [B, num_phases]
            omega: Frequency parameter
            
        Returns:
            angle: Decoded angles in degrees, shape [B]
        """
        batch_size = codes.shape[0]
        
        # Calculate Σsin and Σcos
        sum_sin = torch.zeros(batch_size, device=codes.device)
        sum_cos = torch.zeros(batch_size, device=codes.device)
        
        for n in range(self.num_phases):
            phase_shift = 2 * n * torch.pi / self.num_phases
            phase_shift_tensor = torch.tensor(phase_shift, device=codes.device)
            sum_sin += codes[:, n] * torch.sin(phase_shift_tensor)
            sum_cos += codes[:, n] * torch.cos(phase_shift_tensor)
        
        # PSC decoding: θ = -(1/ω)arctan(Σsin/Σcos)
        angle_rad = -torch.atan2(sum_sin, sum_cos) / omega
        angle_deg = angle_rad * 180.0 / torch.pi
        
        # Normalize to [0, 360)
        return angle_deg % 360


    def calculate_angular_mae_from_psc(self, y_true_angles, y_pred_psc):
        """Calculate angular MAE from PSC predictions and true angles"""
        # Convert PSC predictions to angles
        pred_angles = self.psc_to_angle(y_pred_psc)
        
        # Calculate angular distance (shorter path around circle)
        angular_errors = torch.abs(pred_angles - y_true_angles)
        angular_errors = torch.minimum(angular_errors, 360 - angular_errors)
        return torch.mean(angular_errors)

    def forward(self, x):
        if self.head is not None:
            # Custom head path
            feats = self.backbone(x)
            logits = self.head(feats)
        else:
            # Direct model path
            logits = self.model(x)
        
        # PSC codes don't need normalization like unit vectors
        # They naturally have bounded range due to cosine function
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Convert angle targets to PSC representation
        y_psc = self.angle_to_psc(y)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y_psc)

        # Calculate angular MAE for monitoring
        angular_mae = self.calculate_angular_mae_from_psc(y, y_hat)
        
        # Convert predictions back to angles for logging
        pred_angles = self.psc_to_angle(y_hat)

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pred_angle_mean', torch.mean(pred_angles), on_step=True, on_epoch=True)
        self.log('train_target_angle_mean', torch.mean(y), on_step=True, on_epoch=True)

        # Log PSC code statistics for monitoring
        self.log('train_psc_mean', torch.mean(y_hat), on_step=False, on_epoch=True)
        self.log('train_psc_std', torch.std(y_hat), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Convert angle targets to PSC representation
        y_psc = self.angle_to_psc(y)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y_psc)

        # Convert PSC predictions to angles
        pred_angles = self.psc_to_angle(y_hat)
        
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
        # Convert angle targets to PSC representation
        y_psc = self.angle_to_psc(y)

        y_hat = self(x)
        loss = self.loss_fn(y_hat, y_psc)

        # Convert PSC predictions to angles
        pred_angles = self.psc_to_angle(y_hat)
        
        # Calculate comprehensive test metrics
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
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

        # Use TIMM transforms if possible
        try:
            # Get model-specific data configuration
            data_config = timm.data.resolve_model_data_config(self.hparams.model_name)
            # Match DataLoader settings exactly
            data_config['crop_pct'] = 1.0
            data_config['input_size'] = (3, self.image_size, self.image_size)
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
            psc_codes = self(image_tensor)  # PSC representation
            angle = self.psc_to_angle(psc_codes).item()

        return angle

