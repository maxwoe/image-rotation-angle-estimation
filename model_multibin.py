#!/usr/bin/env python3
"""
MultiBin Approach for Orientation Estimation
===========================================

This approach uses multiple classification heads with different bin resolutions
Prototype approach loosely inspired by Mousavian et al. CVPR 2017 ("3D Bounding Box
Estimation Using Deep Learning and Geometry"). Not a faithful implementation —
uses multiple heads at different resolutions (coarse-to-fine) rather than the
original overlapping-bin + residual regression design.

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


class MultiBinAngleEstimation(pl.LightningModule):
    """
    Prototype hierarchical multi-resolution classification model for angle estimation.
    Loosely inspired by Mousavian et al. CVPR 2017. Not included in the paper comparison.
    
    Uses hierarchical overlapping bin structure (all covering full 360°):
    - Head 1: 36 bins (10° resolution) - coarse orientation
    - Head 2: 72 bins (5° resolution) - medium resolution  
    - Head 3: 144 bins (2.5° resolution) - fine resolution
    
    Each head outputs predictions for the full 360° range with different granularities
    and confidence scores for hierarchical prediction combination.
    """

    def __init__(self, batch_size, train_dir, model_name="vit_tiny_patch16_224", learning_rate=0.001,
                 validation_split=0.1, random_seed=42, image_size=224,
                 bin_counts=[36, 72, 144], confidence_weight=0.3, overlap_margin=0.5, test_dir=None, test_rotation_range=360.0, test_random_seed=42):
        super().__init__()
        self.save_hyperparameters()
        
        # Store test parameters
        self.test_dir = test_dir
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed

        # MultiBin setup - all heads cover full 360° range with different resolutions
        self.bin_counts = bin_counts  # Number of bins for each head
        self.num_heads = len(bin_counts)
        self.confidence_weight = confidence_weight  # Weight for confidence loss
        self.overlap_margin = overlap_margin  # Overlap margin for soft bin assignment
        
        # Calculate bin sizes for each head - all cover 360° with different resolutions
        self.bin_sizes = [360.0 / count for count in bin_counts]
        
        logger.info(f"MultiBin: {self.num_heads} heads with overlapping bins {bin_counts}")
        logger.info(f"Bin resolutions: {[f'{size:.1f}°' for size in self.bin_sizes]}")
        logger.info(f"Overlap margin: {overlap_margin}°")

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

        except (RuntimeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Creating new model with pretrained weights")
            model = cls(**kwargs)
            logger.info("New model created successfully")
            return model

    def load_pretrained_weights(self, checkpoint_path):
        raise NotImplementedError("Pretrained weights loading not implemented for MultiBin model")

    def angle_to_class_multiple(self, angle):
        """Convert continuous angle to class indices for all heads"""
        # Normalize angle to [0, 360) range with proper handling of edge cases
        angle = angle % 360.0
        # Ensure we handle the case where angle might be exactly 360 due to floating point precision
        angle = torch.where(angle >= 360.0, angle - 360.0, angle)
        
        class_indices = []
        for i, (bin_count, bin_size) in enumerate(zip(self.bin_counts, self.bin_sizes)):
            # Convert to class index for this head
            class_idx = (angle / bin_size).long()
            # Handle edge case where angle = 360 (should map to class 0)
            class_idx = class_idx % bin_count
            class_indices.append(class_idx)
        
        return class_indices
    
    def angle_to_soft_labels(self, angle, head_idx):
        """Convert angle to soft labels with overlapping bins for specified head.

        Vectorized: operates on the full [B] batch without Python-level loops.
        Creates soft assignment to the primary bin and its two neighbors within
        the overlap margin for smoother training.
        """
        batch_size = angle.shape[0]
        bin_count = self.bin_counts[head_idx]
        bin_size = self.bin_sizes[head_idx]
        device = angle.device

        angle = angle % 360.0

        # Primary bin index [B]
        primary_bin = (angle / bin_size).long() % bin_count  # [B]

        def circular_dist(a, b):
            d = torch.abs(a - b)
            return torch.minimum(d, 360.0 - d)

        soft_labels = torch.zeros(batch_size, bin_count, device=device)

        # Assign weights for primary bin and its two direct neighbors
        for offset in [0, -1, 1]:
            neighbor_bin = (primary_bin + offset) % bin_count  # [B]
            neighbor_center = (neighbor_bin.float() + 0.5) * bin_size  # [B]
            dist = circular_dist(angle, neighbor_center)  # [B]
            weight = (1.0 - dist / (bin_size * self.overlap_margin)).clamp(min=0.0)  # [B]
            soft_labels.scatter_(1, neighbor_bin.unsqueeze(1), weight.unsqueeze(1))

        # Normalize rows to sum to 1; fall back to hard assignment if all weights are zero
        row_sums = soft_labels.sum(dim=1, keepdim=True)
        zero_rows = (row_sums == 0).squeeze(1)
        soft_labels = soft_labels / row_sums.clamp(min=1e-8)
        if zero_rows.any():
            soft_labels[zero_rows] = 0.0
            soft_labels[zero_rows, primary_bin[zero_rows]] = 1.0

        return soft_labels

    def class_to_angle_multiple(self, class_indices, confidences=None):
        """Convert class indices from multiple heads to angle using hierarchical coarse-to-fine combination.

        Vectorized: operates on the full [B] batch without Python-level loops over samples.
        """
        device = class_indices[0].device

        # Start with coarsest head [B]
        current_angle = (class_indices[0].float() + 0.5) * self.bin_sizes[0]
        current_conf = (confidences[0] if confidences is not None
                        else torch.ones(current_angle.shape, device=device))

        coarse_bin_size = self.bin_sizes[0]
        for i in range(1, self.num_heads):
            fine_angle = (class_indices[i].float() + 0.5) * self.bin_sizes[i]
            fine_conf = (confidences[i] if confidences is not None
                         else torch.ones_like(fine_angle))

            diff = torch.abs(fine_angle - current_angle)
            diff = torch.minimum(diff, torch.tensor(360.0, device=device) - diff)

            window_size = 4 * coarse_bin_size
            within_window = diff <= window_size  # [B] bool mask

            total_conf = current_conf + fine_conf
            w_curr = current_conf / total_conf
            w_fine = fine_conf / total_conf

            curr_rad = current_angle * torch.pi / 180.0
            fine_rad = fine_angle * torch.pi / 180.0
            avg_cos = w_curr * torch.cos(curr_rad) + w_fine * torch.cos(fine_rad)
            avg_sin = w_curr * torch.sin(curr_rad) + w_fine * torch.sin(fine_rad)
            blended = torch.atan2(avg_sin, avg_cos) * 180.0 / torch.pi % 360.0

            current_angle = torch.where(within_window, blended, current_angle)
            current_conf = torch.where(within_window, total_conf / 2, current_conf)
            coarse_bin_size = self.bin_sizes[i]

        return current_angle

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
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses for each head
        total_loss = 0
        total_classification_loss = 0
        total_confidence_loss = 0
        
        for i in range(self.num_heads):
            # Use soft labels with overlapping bins for smoother training
            soft_labels = self.angle_to_soft_labels(y, i)
            
            # Convert logits to probabilities
            pred_probs = F.softmax(head_logits[i], dim=1)
            
            # Use KL divergence loss for soft targets (more appropriate than cross-entropy)
            # KL(soft_labels || pred_probs) = sum(soft_labels * log(soft_labels / pred_probs))
            # For numerical stability, use log_softmax
            log_pred_probs = F.log_softmax(head_logits[i], dim=1)
            cls_loss = F.kl_div(log_pred_probs, soft_labels, reduction='batchmean')
            total_classification_loss += cls_loss
            
            # Entropy-based confidence target: higher confidence for sharper predictions
            with torch.no_grad():
                # Calculate prediction entropy (uncertainty measure)
                pred_probs = F.softmax(head_logits[i], dim=1)
                entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=1)
                
                # Normalize entropy to [0, 1] range (0 = confident, 1 = uncertain)
                max_entropy = torch.log(torch.tensor(self.bin_counts[i], dtype=torch.float, device=entropy.device))
                normalized_entropy = entropy / max_entropy
                
                # Confidence target: 1 - normalized_entropy (higher for sharper predictions)
                confidence_targets = 1.0 - normalized_entropy
            
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
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses
        total_loss = 0
        total_classification_loss = 0
        total_confidence_loss = 0
        
        for i in range(self.num_heads):
            # Use soft labels with overlapping bins (same as training)
            soft_labels = self.angle_to_soft_labels(y, i)
            
            # Use KL divergence loss for soft targets
            log_pred_probs = F.log_softmax(head_logits[i], dim=1)
            cls_loss = F.kl_div(log_pred_probs, soft_labels, reduction='batchmean')
            total_classification_loss += cls_loss
            
            # Entropy-based confidence loss (same logic as training)
            with torch.no_grad():
                # Calculate prediction entropy (uncertainty measure)
                pred_probs = F.softmax(head_logits[i], dim=1)
                entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=1)
                
                # Normalize entropy to [0, 1] range
                max_entropy = torch.log(torch.tensor(self.bin_counts[i], dtype=torch.float, device=entropy.device))
                normalized_entropy = entropy / max_entropy
                
                # Confidence target: 1 - normalized_entropy
                confidence_targets = 1.0 - normalized_entropy
            
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
        
        # Forward pass
        head_logits, head_confidences = self(x)
        
        # Calculate losses using soft labels (consistent with training/validation)
        total_loss = 0
        for i in range(self.num_heads):
            soft_labels = self.angle_to_soft_labels(y, i)
            log_pred_probs = F.log_softmax(head_logits[i], dim=1)
            cls_loss = F.kl_div(log_pred_probs, soft_labels, reduction='batchmean')
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