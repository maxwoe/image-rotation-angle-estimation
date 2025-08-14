#!/usr/bin/env python3
"""
Unit Vector Approach for Orientation Detection
==============================================

This approach uses two output neurons to predict unit vectors [cos(θ), sin(θ)].
The model outputs are normalized to unit length, naturally handling circular angles.
Based on section 3.2 "Unit Vector Coding" from electronics-13-04402.pdf.

Model output: Two values [cos(θ), sin(θ)]
Advantages: No angle wrapping issues, mathematically stable, better convergence
Disadvantages: Slightly more complex output interpretation
"""

from regression_heads import ConfigurableRegressionHead, TensorFlowStyleUnitVectorHead
import torch
import torch.nn as nn
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

      
class UnitVectorAngleDetection(pl.LightningModule):
    """
    Unit vector angle detection model.
    
    Uses two output neurons to predict [cos(θ), sin(θ)] unit vectors.
    This approach naturally handles the circular nature of angles without
    wrapping issues, leading to more stable training.
    """
    
    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 loss_type="mse", use_custom_head: bool = True, use_unit_regularization: bool = True,
                 reg_weight: float = 0.01, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        if not use_custom_head:
            # Use timm's full model directly
            self.model = timm.create_model(model_name, pretrained=True, num_classes=2)
            self.head = None  # Not needed, backbone handles regression
        else:
            # Feature extractor + custom head
            self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')

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
                out_features=2,
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
        self.use_unit_regularization = use_unit_regularization
        self.reg_weight = reg_weight

        # Choose loss function
        if loss_type == "mae":
            self.loss_fn = nn.L1Loss()  # Direct Euclidean distance between unit vectors
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()  # Squared Euclidean distance between unit vectors
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()  # Robust loss, less sensitive to outliers
        elif loss_type == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss()  # Robust loss, less sensitive to outliers
        elif loss_type == "smooth_l1_cos":
            self.loss_fn = self.smooth_l1_cos_loss
        elif loss_type == "cdl":
            self.loss_fn = self.cosine_distance_loss
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

    def unit_vector_regularization_loss(self, predictions, targets):
        """
        Combine main loss with unit vector regularization instead of explicit normalization.
        
        This approach encourages unit vectors through optimization rather than forcing them,
        which preserves gradient flow and allows the model to learn naturally.
        
        Args:
            predictions: Raw model outputs [B, 2]
            targets: Target unit vectors [B, 2]
            
        Returns:
            total_loss: Main loss + regularization term
        """
        main_loss = self.loss_fn(predictions, targets)
        
        if self.use_unit_regularization:
            # Regularization: penalize deviation from unit magnitude
            # Sec. 3.2. https://arxiv.org/abs/1805.06485
            # Penalty term in the loss function acts as a regularizer and leads to better training stability. 
            # The choice of λ is not crucial; we found that any value between 0.1 and 0.001 serves the purpose (we use λ = 0.01)
            pred_magnitudes = torch.norm(predictions, p=2, dim=1)
            unit_reg = torch.mean((pred_magnitudes - 1.0) ** 2)
            
            # Combine losses
            total_loss = main_loss + self.reg_weight * unit_reg
            
            # Store components for logging
            self.last_main_loss = main_loss.detach()
            self.last_unit_reg = unit_reg.detach()
            
            return total_loss
        else:
            # No regularization - just main loss
            return main_loss

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
    
    def smooth_l1_cos_loss(self, y_pred_unit, y_true_unit, beta=1.0):
        """
        Smooth L1 loss applied to (1 - cos(θ_p - θ_t)) for unit vectors
        Uses trigonometric identity: cos(θ_p - θ_t) = cos(θ_p)cos(θ_t) + sin(θ_p)sin(θ_t)
        
        For unit vectors: [cos(θ), sin(θ)]
        cos(θ_p - θ_t) = y_pred[0]*y_true[0] + y_pred[1]*y_true[1] (dot product)
        """
        # Calculate cos(θ_p - θ_t) using dot product
        cos_diff = torch.sum(y_pred_unit * y_true_unit, dim=1)
        
        # Calculate 1 - cos(θ_p - θ_t)
        error = 1 - cos_diff  # Range [0, 2]
        
        # Apply smooth L1 to the error
        smooth_l1 = torch.where(
            error < beta,
            0.5 * error ** 2 / beta,
            error - 0.5 * beta
        )
        
        return torch.mean(smooth_l1)

    def cosine_distance_loss(self, y_pred_unit, y_true_unit):
        """
        Cosine Distance Loss (CDL) from electronics-13-04402.pdf Equation (6)
        
        CDL = Σ[1-CS(pred,gt)]² × [|t_sin - sin(θ_gt)| + |t_cos - cos(θ_gt)|] / N_b
        
        Args:
            y_true_unit: Ground truth unit vectors [cos(θ), sin(θ)] of shape [B, 2]
            y_pred_unit: Predicted unit vectors [cos(θ), sin(θ)] of shape [B, 2]
            
        Returns:
            CDL loss value
        """
        # Calculate cosine similarity: CS = dot product for unit vectors
        cosine_similarity = torch.sum(y_pred_unit * y_true_unit, dim=1)
        
        # First term: [1 - CS]² (cosine distance squared)
        cosine_distance_squared = (1 - cosine_similarity) ** 2
        
        # Second term: L1 component error
        l1_component_error = torch.sum(torch.abs(y_pred_unit - y_true_unit), dim=1)
        
        # Combine terms: CDL = [1-CS]² × [L1_component_error]
        cdl_loss = cosine_distance_squared * l1_component_error
        
        return torch.mean(cdl_loss)

    def cosine_loss(self, y_pred_unit, y_true_unit):
        """
        Cosine loss: 1 - cos(Δ) for unit vectors
        Simple circular loss based on cosine similarity
        """
        # For unit vectors, cos(angle_diff) = dot_product
        cosine_sim = torch.sum(y_true_unit * y_pred_unit, dim=1)
        cosine_distance = 1 - cosine_sim
        
        return torch.mean(cosine_distance)
        
    def chord_loss(self, y_pred_unit, y_true_unit):
        """
        Chord loss: sqrt(2*(1 - cos Δ)) for unit vectors
        Circular analog of absolute difference
        """
        # For unit vectors, cos(angle_diff) = dot_product
        cosine_sim = torch.sum(y_true_unit * y_pred_unit, dim=1)
        chord_distance = torch.sqrt(2 * (1 - cosine_sim))
        
        return torch.mean(chord_distance)
    
    def von_mises_loss(self, y_pred_unit, y_true_unit, kappa=1.0):
        """
        Von Mises loss for unit vectors
        Based on von Mises distribution (circular equivalent of Gaussian)
        """
        # For unit vectors, cos(angle_diff) = dot_product
        # Von Mises negative log-likelihood (ignoring constants)
        cosine_sim = torch.sum(y_true_unit * y_pred_unit, dim=1)
        nll = -kappa * cosine_sim

        return torch.mean(nll)
    
    def calculate_angular_mae_from_unit_vectors(self, y_true_angles, y_pred_unit_vectors):
        """Calculate angular MAE from unit vectors and true angles"""
        # Convert predictions back to angles
        pred_angles = torch.atan2(y_pred_unit_vectors[:, 1], y_pred_unit_vectors[:, 0]) * 180.0 / torch.pi
        pred_angles = torch.remainder(pred_angles, 360.0)  # Normalize to [0, 360)
        
        # Calculate angular distance (shorter path around circle)
        angular_errors = torch.abs(pred_angles - y_true_angles)
        angular_errors = torch.minimum(angular_errors, 360 - angular_errors)
        return torch.mean(angular_errors)

    def forward(self, x):
        if self.head is not None:
            # Custom head path
            feats = self.backbone(x)
            out = self.head(feats)
        else:
            # Direct model path
            out = self.model(x)
        
        # out = torch.tanh(out)
        # out = nn.functional.normalize(out, p=2, dim=1, eps=1e-8) # better to avoid explicit normalization
        # Return raw outputs - let regularization encourage unit vectors naturally
        return out  # [B, 2] raw outputs that will be encouraged to be unit vectors
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Convert angle targets to unit vectors [cos(θ), sin(θ)]
        y_rad = y * torch.pi / 180.0
        y_unit = torch.stack([torch.cos(y_rad), torch.sin(y_rad)], dim=1)

        y_hat = self(x)
        
        # Use regularization loss instead of direct loss
        loss = self.unit_vector_regularization_loss(y_hat, y_unit)

        # Calculate angular MAE in degrees (human interpretable)
        angular_mae = self.calculate_angular_mae_from_unit_vectors(y, y_hat)
        
        # Convert predictions back to angles for logging
        pred_angles = torch.atan2(y_hat[:, 1], y_hat[:, 0]) * 180.0 / torch.pi
        pred_angles = torch.remainder(pred_angles, 360.0)  # Normalize to [0, 360)

        # Log total loss and human-interpretable metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pred_angle_mean', torch.mean(pred_angles), on_step=True, on_epoch=True)
        self.log('train_target_angle_mean', torch.mean(y), on_step=True, on_epoch=True)
        
        # Log regularization components if enabled
        if self.use_unit_regularization:
            self.log('train_main_loss', self.last_main_loss, on_step=True, on_epoch=True)
            self.log('train_unit_reg', self.last_unit_reg, on_step=True, on_epoch=True)
            
            # Log average prediction magnitude for monitoring
            pred_magnitudes = torch.norm(y_hat, p=2, dim=1)
            self.log('train_pred_magnitude_mean', torch.mean(pred_magnitudes), on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Convert angle targets to unit vectors [cos(θ), sin(θ)]
        y_rad = y * torch.pi / 180.0
        y_unit = torch.stack([torch.cos(y_rad), torch.sin(y_rad)], dim=1)

        y_hat = self(x)
        
        # Use regularization loss for validation too
        loss = self.unit_vector_regularization_loss(y_hat, y_unit)

        # Calculate comprehensive metrics
        pred_angles = torch.atan2(y_hat[:, 1], y_hat[:, 0]) * 180.0 / torch.pi
        pred_angles = torch.remainder(pred_angles, 360.0)
        
        # Quick metrics for logging
        val_metrics = compute_validation_metrics(pred_angles, y)
        
        # Log all metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        for metric_name, metric_value in val_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, 
                    prog_bar=(metric_name == 'val_mae_deg'))  # Only show MAE in progress bar
        
        # Log regularization components if enabled
        if self.use_unit_regularization:
            self.log('val_main_loss', self.last_main_loss, on_step=False)
            self.log('val_unit_reg', self.last_unit_reg, on_step=False)
            
            # Log average prediction magnitude for monitoring
            pred_magnitudes = torch.norm(y_hat, p=2, dim=1)
            self.log('val_pred_magnitude_mean', torch.mean(pred_magnitudes), on_step=False)
            
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # Convert angle targets to unit vectors
        y_rad = y * torch.pi / 180.0
        y_unit = torch.stack([torch.cos(y_rad), torch.sin(y_rad)], dim=1)

        y_hat = self(x)
        
        # Use regularization loss for test too
        loss = self.unit_vector_regularization_loss(y_hat, y_unit)

        # Calculate comprehensive test metrics
        pred_angles = torch.atan2(y_hat[:, 1], y_hat[:, 0]) * 180.0 / torch.pi
        pred_angles = torch.remainder(pred_angles, 360.0)
        
        # Quick metrics for logging
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Log all metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in test_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        
        # Log regularization components if enabled
        if self.use_unit_regularization:
            self.log('test_main_loss', self.last_main_loss, on_step=False, on_epoch=True)
            self.log('test_unit_reg', self.last_unit_reg, on_step=False, on_epoch=True)
            
        return loss


    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adadelta(self.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate) # amsgrad=True
        # optimizer = torch.optim.Adamax(self.parameters(), lr=self.learning_rate)

        is_overfitting = (hasattr(self.trainer, 'overfit_batches') and self.trainer.overfit_batches > 0)

        if is_overfitting:
            return {"optimizer": optimizer}
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss", "frequency": 1}

            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7)
            # lr_schedule = 'lr_scheduler': {'scheduler': scheduler,'interval': 'epoch'}

            # Learning rate scheduler; adjusts the learning rate during training
            # total_steps = self.trainer.max_epochs * len(self.train_dataloader()) # n_epochs * steps_per_epoch
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, total_steps=total_steps)
            # lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # Check if we're in overfitting mode (for consistent angle generation)
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
            # Setup test dataset (assuming images are correctly oriented, i.e., 0 degrees)
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
            raw_output = self(image_tensor)  # Raw outputs (not necessarily unit vectors)
            
            # Normalize for angle calculation if not already unit vectors
            if self.use_unit_regularization:
                # With regularization, outputs should be close to unit vectors but may not be exact
                normalized_output = nn.functional.normalize(raw_output, p=2, dim=1, eps=1e-8)
            else:
                # Without regularization, assume outputs are already reasonable
                normalized_output = raw_output
            
            cos_val, sin_val = normalized_output[0, 0].item(), normalized_output[0, 1].item()
            # Convert to angle using atan2
            angle = torch.atan2(torch.tensor(sin_val), torch.tensor(cos_val)).item() * 180.0 / torch.pi
            angle = torch.remainder(angle, 360.0) % 360  # Normalize to [0, 360)

        return angle

