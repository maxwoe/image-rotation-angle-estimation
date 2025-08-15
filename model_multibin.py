#!/usr/bin/env python3
"""
MultiBin Approach for Orientation Detection
===========================================

This approach uses multiple classification heads with different bin resolutions
and confidence estimation. Based on Mousavian et al. CVPR 2017 and Lee et al. 2022.

The model outputs predictions from multiple heads (coarse to fine) and combines
them using confidence weighting for robust angle estimation.

Model output: Multiple classification heads + confidence estimates
Advantages: Hierarchical prediction, confidence estimation, robust to ambiguity
Disadvantages: More complex architecture, higher computational cost
"""

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


class MultiBinAngleDetection(pl.LightningModule):
    """
    MultiBin angle detection model with multiple classification heads
    and confidence estimation.
    
    Uses hierarchical bin structure:
    - Head 1: 4 bins (90° each) - coarse orientation
    - Head 2: 8 bins (45° each) - medium resolution  
    - Head 3: 24 bins (15° each) - fine resolution
    - Head 4: 72 bins (5° each) - very fine resolution
    
    Each head also outputs a confidence score for prediction combination.
    """

    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 bin_counts=[4, 8, 24, 72], confidence_weight=0.1, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        # MultiBin setup
        self.bin_counts = bin_counts  # Number of bins for each head
        self.num_heads = len(bin_counts)
        self.confidence_weight = confidence_weight  # Weight for confidence loss
        
        # Calculate bin sizes for each head
        self.bin_sizes = [360.0 / count for count in bin_counts]
        
        logger.info(f"MultiBin: {self.num_heads} heads with bins {bin_counts}")
        logger.info(f"Bin sizes: {[f'{size:.1f}°' for size in self.bin_sizes]}")

        # Feature extractor (backbone without classifier)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='avg')
        
        # Get feature dimension
        with torch.no_grad():
            input_size = image_size if image_size else 224
            dummy_input = torch.randn(1, 3, input_size, input_size)
            dummy_features = self.backbone(dummy_input)
            self.feature_dim = dummy_features.shape[-1]
        
        # Create multiple classification heads
        self.heads = nn.ModuleList()
        self.confidence_heads = nn.ModuleList()
        
        for i, num_bins in enumerate(bin_counts):
            # Classification head
            head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim // 2, num_bins)
            )
            self.heads.append(head)
            
            # Confidence head (outputs single confidence score)
            conf_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.feature_dim // 4, 1),
                nn.Sigmoid()  # Confidence in [0, 1]
            )
            self.confidence_heads.append(conf_head)
        
        # Store hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.image_size = image_size
        self.loss_type = "multibin"  # For compatibility with training script

        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        self.confidence_loss = nn.MSELoss()

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
        raise NotImplementedError("Pretrained weights loading not implemented for MultiBin model")

    def angle_to_class_multiple(self, angle):
        """Convert continuous angle to class indices for all heads"""
        # Normalize angle to [0, 360)
        angle = angle % 360
        
        class_indices = []
        for i, (bin_count, bin_size) in enumerate(zip(self.bin_counts, self.bin_sizes)):
            # Convert to class index for this head
            class_idx = (angle / bin_size).long()
            # Handle edge case where angle = 360 (should map to class 0)
            class_idx = class_idx % bin_count
            class_indices.append(class_idx)
        
        return class_indices

    def class_to_angle_multiple(self, class_indices, confidences=None):
        """Convert class indices from multiple heads to angle using confidence weighting"""
        batch_size = class_indices[0].shape[0]
        angles = torch.zeros(batch_size, device=class_indices[0].device)
        
        for b in range(batch_size):
            # Get predictions from all heads for this sample
            head_angles = []
            head_confidences = []
            
            for i, (class_idx, bin_size) in enumerate(zip(class_indices, self.bin_sizes)):
                # Convert class to angle (center of bin)
                angle = (class_idx[b].float() + 0.5) * bin_size
                head_angles.append(angle)
                
                # Get confidence for this head
                if confidences is not None:
                    head_confidences.append(confidences[i][b])
                else:
                    head_confidences.append(1.0 / self.num_heads)  # Equal weighting
            
            # Convert to tensors
            head_angles = torch.stack(head_angles)
            head_confidences = torch.stack(head_confidences)
            
            # Normalize confidences
            head_confidences = head_confidences / (torch.sum(head_confidences) + 1e-8)
            
            # Weighted average using unit vectors (circular averaging)
            head_angles_rad = head_angles * torch.pi / 180.0
            cos_components = torch.cos(head_angles_rad)
            sin_components = torch.sin(head_angles_rad)
            
            # Weighted average of unit vectors
            avg_cos = torch.sum(head_confidences * cos_components)
            avg_sin = torch.sum(head_confidences * sin_components)
            
            # Convert back to angle
            final_angle = torch.atan2(avg_sin, avg_cos) * 180.0 / torch.pi
            angles[b] = final_angle % 360.0
        
        return angles

    def calculate_angular_mae_from_multibin(self, y_true_angles, head_logits, head_confidences):
        """Calculate angular MAE from MultiBin predictions and true angles"""
        # Get predicted classes from each head
        class_indices = [torch.argmax(logits, dim=1) for logits in head_logits]
        
        # Get confidence values
        confidences = [conf.squeeze(1) for conf in head_confidences]
        
        # Convert to angles using confidence weighting
        pred_angles = self.class_to_angle_multiple(class_indices, confidences)
        
        # Calculate angular distance (shorter path around circle)
        angular_errors = torch.abs(pred_angles - y_true_angles)
        angular_errors = torch.minimum(angular_errors, 360 - angular_errors)
        return torch.mean(angular_errors)

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get predictions from all heads
        head_logits = []
        head_confidences = []
        
        for i in range(self.num_heads):
            logits = self.heads[i](features)
            confidence = self.confidence_heads[i](features)
            
            head_logits.append(logits)
            head_confidences.append(confidence)
        
        return head_logits, head_confidences

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Convert angles to class labels for each head
        y_classes = self.angle_to_class_multiple(y)
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses for each head
        total_loss = 0
        total_classification_loss = 0
        total_confidence_loss = 0
        
        for i in range(self.num_heads):
            # Classification loss
            cls_loss = self.classification_loss(head_logits[i], y_classes[i])
            total_classification_loss += cls_loss
            
            # Confidence target: higher confidence for more accurate predictions
            with torch.no_grad():
                pred_classes = torch.argmax(head_logits[i], dim=1)
                # Simple confidence target: 1.0 if correct, lower if wrong
                confidence_targets = (pred_classes == y_classes[i]).float()
                # Add some noise to avoid overconfident predictions
                confidence_targets = 0.8 * confidence_targets + 0.1
            
            # Confidence loss
            conf_loss = self.confidence_loss(head_confidences[i].squeeze(1), confidence_targets)
            total_confidence_loss += conf_loss
        
        # Combine losses
        total_loss = total_classification_loss + self.confidence_weight * total_confidence_loss
        
        # Calculate angular MAE for monitoring
        angular_mae = self.calculate_angular_mae_from_multibin(y, head_logits, head_confidences)
        
        # Get final predicted angles for logging
        class_indices = [torch.argmax(logits, dim=1) for logits in head_logits]
        confidences = [conf.squeeze(1) for conf in head_confidences]
        pred_angles = self.class_to_angle_multiple(class_indices, confidences)
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_cls_loss', total_classification_loss, on_step=True, on_epoch=True)
        self.log('train_conf_loss', total_confidence_loss, on_step=True, on_epoch=True)
        self.log('train_mae_deg', angular_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_pred_angle_mean', torch.mean(pred_angles), on_step=True, on_epoch=True)
        self.log('train_target_angle_mean', torch.mean(y), on_step=True, on_epoch=True)
        
        # Log per-head confidence statistics
        for i in range(self.num_heads):
            self.log(f'train_conf_head_{i}', torch.mean(head_confidences[i]), on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Convert angles to class labels for each head
        y_classes = self.angle_to_class_multiple(y)
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses
        total_loss = 0
        total_classification_loss = 0
        total_confidence_loss = 0
        
        for i in range(self.num_heads):
            # Classification loss
            cls_loss = self.classification_loss(head_logits[i], y_classes[i])
            total_classification_loss += cls_loss
            
            # Confidence loss (same logic as training)
            with torch.no_grad():
                pred_classes = torch.argmax(head_logits[i], dim=1)
                confidence_targets = (pred_classes == y_classes[i]).float()
                confidence_targets = 0.8 * confidence_targets + 0.1
            
            conf_loss = self.confidence_loss(head_confidences[i].squeeze(1), confidence_targets)
            total_confidence_loss += conf_loss
        
        total_loss = total_classification_loss + self.confidence_weight * total_confidence_loss
        
        # Calculate final predicted angles
        class_indices = [torch.argmax(logits, dim=1) for logits in head_logits]
        confidences = [conf.squeeze(1) for conf in head_confidences]
        pred_angles = self.class_to_angle_multiple(class_indices, confidences)
        
        # Calculate comprehensive metrics
        val_metrics = compute_validation_metrics(pred_angles, y)
        
        # Log all metrics
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cls_loss', total_classification_loss, on_step=False, on_epoch=True)
        self.log('val_conf_loss', total_confidence_loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in val_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True, 
                    prog_bar=(metric_name == 'val_mae_deg'))  # Only show MAE in progress bar
        
        # Log per-head confidence statistics
        for i in range(self.num_heads):
            self.log(f'val_conf_head_{i}', torch.mean(head_confidences[i]), on_step=False, on_epoch=True)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        # Convert angles to class labels for each head
        y_classes = self.angle_to_class_multiple(y)
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses
        total_loss = 0
        for i in range(self.num_heads):
            cls_loss = self.classification_loss(head_logits[i], y_classes[i])
            total_loss += cls_loss
        
        # Calculate final predicted angles
        class_indices = [torch.argmax(logits, dim=1) for logits in head_logits]
        confidences = [conf.squeeze(1) for conf in head_confidences]
        pred_angles = self.class_to_angle_multiple(class_indices, confidences)
        
        # Calculate comprehensive test metrics
        test_metrics = compute_validation_metrics(pred_angles, y, prefix="test")
        
        # Log all metrics
        self.log('test_loss', total_loss, on_step=False, on_epoch=True)
        for metric_name, metric_value in test_metrics.items():
            self.log(metric_name, metric_value, on_step=False, on_epoch=True)
        
        return total_loss

    def configure_optimizers(self):
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
            head_logits, head_confidences = self(image_tensor)
            
            # Get predicted classes and confidences
            class_indices = [torch.argmax(logits, dim=1) for logits in head_logits]
            confidences = [conf.squeeze(1) for conf in head_confidences]
            
            # Convert to final angle using confidence weighting
            angle = self.class_to_angle_multiple(class_indices, confidences).item()

        return angle