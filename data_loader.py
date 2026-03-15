
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import timm.data
import random
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from rotation_utils import rotate_preserve_content
# Removed config.py import - using inline default

class RotationDataset(Dataset):
    def __init__(self, image_paths, image_size=None, mode="train", val_angles=None, train_angles=None, model_name=None, 
                 test_rotation_range=360.0, test_random_seed=42):
        """
        Dataset for image orientation estimation with automatic train/val splitting
        
        Args:
            image_paths: List of image file paths
            image_size: Size to resize images to
            mode: "train", "val", or "test"
            val_angles: For validation, list of predefined angles to test (one per image)
            train_angles: For training, list of predefined angles (enables consistent overfitting)
            model_name: TIMM model name for model-specific transforms
            test_rotation_range: Max rotation range for test mode (360=full, 45=±45°)
            test_random_seed: Random seed for repeatable test rotations
        """
        self.image_paths = image_paths
        self.image_size = image_size
        self.mode = mode
        self.val_angles = val_angles
        self.train_angles = train_angles
        self.test_rotation_range = test_rotation_range
        self.test_random_seed = test_random_seed
        
        # Generate repeatable test angles if in test mode
        if mode == "test":
            self.test_angles = self._generate_test_angles()
        
        self.transform = self._create_transforms(model_name, image_size)
    
    def _generate_test_angles(self):
        """Generate repeatable test angles for each image"""
        import numpy as np
        np.random.seed(self.test_random_seed)
        
        if self.test_rotation_range >= 360:
            # Full rotation range: 0-360°
            test_angles = np.random.uniform(0, 360, len(self.image_paths))
        else:
            # Limited range: ±rotation_range/2
            half_range = self.test_rotation_range / 2
            test_angles = np.random.uniform(-half_range, half_range, len(self.image_paths))
        
        return test_angles.tolist()
    
    def _create_transforms(self, model_name, image_size):
        """Create image transforms, preferring TIMM model-specific transforms"""
        def _create_standard_transforms(image_size=224):
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        if model_name:
            try:
                # Get model-specific data configuration
                data_config = timm.data.resolve_model_data_config(model_name)
                # Override crop settings for orientation estimation - we need full image content
                data_config['crop_pct'] = 1.0  # Use full image, no center cropping
                
                # Determine input size with priority: explicit > model default > fallback
                if image_size is not None:
                    # Explicit override from --image-size
                    final_size = image_size
                    size_source = "explicit"
                    data_config['input_size'] = (3, image_size, image_size)
                else:
                    # Use model's actual optimal size
                    model = timm.create_model(model_name, pretrained=False)
                    actual_input_size = model.default_cfg.get('input_size', (3, 224, 224))
                    final_size = actual_input_size[1]  # Height (assuming square)
                    size_source = f"TIMM default for {model_name}"
                    data_config['input_size'] = actual_input_size
                
                # Log the resolution being used
                logger.info(f"Input resolution: {final_size}x{final_size} ({size_source})")
                
                # Create model-specific transforms (inference transforms, no augmentation)
                transform = timm.data.create_transform(
                    **data_config, 
                    is_training=False  # No augmentation, we handle rotation ourselves
                )
                return transform
            except Exception as e:
                logger.warning(f"Failed to create TIMM transforms for {model_name}: {e}")
                # Fallback to standard transforms
                fallback_size = image_size if image_size is not None else 224
                logger.info(f"Input resolution: {fallback_size}x{fallback_size} (fallback - TIMM failed)")
                return _create_standard_transforms(fallback_size)
        else:
            # Standard ImageNet transforms
            fallback_size = image_size if image_size is not None else 224
            logger.info(f"Input resolution: {fallback_size}x{fallback_size} (standard transforms)")
            return _create_standard_transforms(fallback_size)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        if self.mode == "test":
            # Test mode: use predefined repeatable angles
            angle = self.test_angles[idx]
            
            # Apply rotation with the predefined angle
            rotated_image = rotate_preserve_content(image_path, angle)
            
            # Transform to tensor
            image_tensor = self.transform(rotated_image)
            
            # Label is the angle we applied (what the model should predict)
            return image_tensor, torch.tensor(angle, dtype=torch.float32)
        elif self.mode == "train":
            # Use predefined angles for consistent overfitting, or random for normal training
            if self.train_angles is not None:
                angle = self.train_angles[idx]
            else:
                # Random rotation for normal training
                angle = random.uniform(0, 360)
        else:
            # Validation mode: use predefined angles for consistent validation
            if self.val_angles is not None:
                angle = self.val_angles[idx]
            else:
                # Fallback to random angle if no predefined angles
                angle = random.uniform(0, 360)
        
        # Apply proper rotation that preserves content (no artificial borders)
        # Positive angle to create an image at that orientation
        rotated_image = rotate_preserve_content(image_path, angle)
        
        # Transform to tensor
        image_tensor = self.transform(rotated_image)
        
        # Label is the current orientation angle of the rotated image
        return image_tensor, torch.tensor(angle, dtype=torch.float32)
    
    @staticmethod
    def create_datasets(image_dir, validation_split=0.2, random_seed=42, image_size=None, enable_overfitting=False, model_name=None):
        """
        Create train and validation datasets with automatic splitting
        
        Args:
            image_dir: Directory containing correctly oriented images
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducible splits
            image_size: Size to resize images to
            enable_overfitting: If True, generate consistent training angles for overfitting
            model_name: TIMM model name for model-specific transforms
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Get all image paths
        image_dir = Path(image_dir)
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff",
                      "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(image_dir.glob(ext))

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {image_dir} (searched: jpg, jpeg, png, bmp, tiff)")
        
        # Split into train and validation sets
        train_paths, val_paths = train_test_split(
            image_paths, 
            test_size=validation_split, 
            random_state=random_seed
        )
        
        # Create predefined validation angles for consistent testing
        # Use a variety of angles including edge cases
        val_angles = []
        random.seed(random_seed)  # Ensure reproducible validation angles
        for i in range(len(val_paths)):
            angle = random.uniform(0, 360)  # Uniform random — no bias toward cardinal/diagonal
            val_angles.append(angle)
        
        # Create predefined training angles for consistent overfitting if enabled
        train_angles = None
        if enable_overfitting:
            train_angles = []
            random.seed(random_seed + 1000)  # Different seed for training angles
            for i in range(len(train_paths)):
                # Generate consistent training angles for overfitting
                angle = random.uniform(0, 359)
                train_angles.append(angle) 
        
        # Create datasets
        train_dataset = RotationDataset(
            image_paths=train_paths,
            image_size=image_size,
            mode="train",
            train_angles=train_angles,
            model_name=model_name
        )
        
        val_dataset = RotationDataset(
            image_paths=val_paths,
            image_size=image_size,
            mode="val",
            val_angles=val_angles,
            model_name=model_name
        )
        
        logger.info(f"Created datasets: {len(train_paths)} training, {len(val_paths)} validation images")
        
        return train_dataset, val_dataset