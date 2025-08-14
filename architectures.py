"""
Shared architectures reference with research-based optimal learning rates

Learning rates are based on original research papers and optimized for:
- AdamW optimizer (recommended for fine-tuning)
- Transfer learning / fine-tuning scenarios
- ImageNet pretrained models

RESEARCH CITATIONS & LEARNING RATE RATIONALE:
==============================================

EfficientNet V2 (5e-5): 
- Multiple sources confirm 5e-5 optimal for AdamW fine-tuning
- Original paper used RMSprop with 0.256 for training from scratch (batch size 4096)
- Fine-tuning consensus: 5e-5 to 1e-4 range widely recommended

Vision Transformers (1e-4):
- Original ViT paper used SGD with 0.06 for large-scale training
- For AdamW fine-tuning, 1e-4 prevents convergence issues seen with higher rates
- Reduced from 3e-4 based on fine-tuning best practices

ConvNeXt V2 (2e-4 to 6.25e-4):
- Original paper: "AdamW optimizer with learning rate sweep {1e-4, 2e-4, 3e-4}"
- Atto (3.7M params): 2e-4 optimal, Base (89M params): 6.25e-4 optimal
- Rates scale with model capacity as intended by authors

EdgeNeXt (1e-3):
- Original paper used 6e-3 for training from scratch
- 1e-3 recommended for fine-tuning (reduced from original 5e-5 which was too low)
- Significant increase needed for proper convergence

MambaOut (4e-3):
- Original paper: "4e-3 for ImageNet classification"
- Increased from 1e-3 to match paper specifications
- Higher rate needed due to Mamba architecture characteristics

VOLO (1e-3):
- Original paper: "1.6e-3 for VOLO-D1, 1.0e-3 for VOLO-D2"
- Significantly increased from 5e-5 which was too conservative
- VOLO requires higher learning rates for optimal performance

Swin Transformer (1e-3):
- Original paper: "1e-3 with batch size 1024"
- Standard rate for Swin architecture family
- Increased from 1e-4 to match paper specifications

ADAMW OPTIMIZER COMPATIBILITY:
=============================
All learning rates are optimized for AdamW optimizer, which is:
- Recommended by most recent papers for fine-tuning
- Better weight decay handling than standard Adam
- Standard choice for transfer learning scenarios

If using different optimizers (SGD, RMSprop, etc.), these rates may need adjustment.
Original papers often used different optimizers with different optimal rates.

Note: These rates are specifically tuned for AdamW optimizer.
If using different optimizers, rates may need adjustment.
"""

ARCHITECTURES = {
    # Vision Transformers (1e-4 for AdamW fine-tuning, original paper used SGD with 0.06)
    # "vit_tiny_patch16_224.augreg_in21k_ft_in1k": {
    #     "name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    #     "input_size": 224,
    #     "type": "vision_transformer",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 512
    # },
    "vit_small_patch16_224.augreg_in21k_ft_in1k": {
        "name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "input_size": 224,
        "type": "vision_transformer",
        "default_lr": 0.0001,
        "default_batch_size": 512
    },
    
    # EfficientViT (1e-4 similar to ViT but more efficient)
    "efficientvit_b0.r224_in1k": {
        "name": "efficientvit_b0.r224_in1k",
        "input_size": 224,
        "type": "efficientvit",
        "default_lr": 0.0001,
        "default_batch_size": 512
    },
    # "efficientvit_b1.r224_in1k": {
    #     "name": "efficientvit_b1.r224_in1k",
    #     "input_size": 224,
    #     "type": "efficientvit",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 512
    # },
    # "efficientvit_b2.r224_in1k": {
    #     "name": "efficientvit_b2.r224_in1k",
    #     "input_size": 224,
    #     "type": "efficientvit",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 512
    # },
    "efficientvit_b3.r224_in1k": {
        "name": "efficientvit_b3.r224_in1k",
        "input_size": 224,
        "type": "efficientvit",
        "default_lr": 0.0001,
        "default_batch_size": 512
    },
    
    # ConvNeXt V2 (research-based LRs from original paper: Atto=2e-4, Base=6.25e-4)
    "convnextv2_atto.fcmae_ft_in1k": {
        "name": "convnextv2_atto.fcmae_ft_in1k",
        "input_size": 224,
        "type": "convnext",
        "default_lr": 0.0002,
        "default_batch_size": 1024
    },
    # "convnextv2_femto.fcmae_ft_in1k": {
    #     "name": "convnextv2_femto.fcmae_ft_in1k",
    #     "input_size": 224,
    #     "type": "convnext",
    #     "default_lr": 0.0002,
    #     "default_batch_size": 1024
    # },
    # "convnextv2_pico.fcmae_ft_in1k": {
    #     "name": "convnextv2_pico.fcmae_ft_in1k",
    #     "input_size": 224,
    #     "type": "convnext",
    #     "default_lr": 0.0002,
    #     "default_batch_size": 1024
    # },
    "convnextv2_nano.fcmae_ft_in1k": {
        "name": "convnextv2_nano.fcmae_ft_in1k",
        "input_size": 224,
        "type": "convnext",
        "default_lr": 0.0002,
        "default_batch_size": 1024
    },
    # "convnextv2_base.fcmae_ft_in1k": {
    #     "name": "convnextv2_base.fcmae_ft_in1k",
    #     "input_size": 224,
    #     "type": "convnext",
    #     "default_lr": 0.000625,
    #     "default_batch_size": 1024
    # },
    "convnextv2_base.fcmae_ft_in22k_in1k": {
        "name": "convnextv2_base.fcmae_ft_in22k_in1k",
        "input_size": 224,
        "type": "convnext",
        "default_lr": 0.000625,
        "default_batch_size": 1024
    },
    # "convnextv2_large.fcmae_ft_in1k": {
    #     "name": "convnextv2_large.fcmae_ft_in1k",
    #     "input_size": 224,
    #     "type": "convnext",
    #     "default_lr": 0.000625,
    #     "default_batch_size": 1024
    # },
    
    # ResNet (1e-4 for fine-tuning, classic CNN architecture)
    # "resnet18.a1_in1k": {
    #     "name": "resnet18.a1_in1k",
    #     "input_size": 224,
    #     "type": "resnet",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 256
    # },
    # "resnet34.a1_in1k": {
    #     "name": "resnet34.a1_in1k",
    #     "input_size": 224,
    #     "type": "resnet",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 256
    # },
    "resnet50.a1_in1k": {
        "name": "resnet50.a1_in1k",
        "input_size": 224,
        "type": "resnet",
        "default_lr": 0.0001,
        "default_batch_size": 256
    },
    # "resnet101.a1_in1k": {
    #     "name": "resnet101.a1_in1k",
    #     "input_size": 224,
    #     "type": "resnet",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 256
    # },
    
    # RegNet (5e-5 for fine-tuning) - COMMENTED OUT: Poor CGD performance (6.4°, 4.6°)
    # "regnetx_002.pycls_in1k": {
    #     "name": "regnetx_002.pycls_in1k",
    #     "input_size": 224,
    #     "type": "regnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "regnetx_004.pycls_in1k": {
    #     "name": "regnetx_004.pycls_in1k",
    #     "input_size": 224,
    #     "type": "regnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    
    # EfficientNet V2 (5e-5 optimal for AdamW fine-tuning) - COMMENTED OUT: Poor CGD performance (4.1-4.7°)
    # "tf_efficientnetv2_b0.in1k": {
    #     "name": "tf_efficientnetv2_b0.in1k",
    #     "input_size": 192,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "tf_efficientnetv2_b1.in1k": {
    #     "name": "tf_efficientnetv2_b1.in1k",
    #     "input_size": 192,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "tf_efficientnetv2_b2.in1k": {
    #     "name": "tf_efficientnetv2_b2.in1k",
    #     "input_size": 208,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "tf_efficientnetv2_b3.in1k": {
    #     "name": "tf_efficientnetv2_b3.in1k",
    #     "input_size": 240,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "tf_efficientnetv2_s.in21k": {
    #     "name": "tf_efficientnetv2_s.in21k",
    #     "input_size": 300,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    # "efficientnetv2_rw_t.ra2_in1k": {
    #     "name": "efficientnetv2_rw_t.ra2_in1k",
    #     "input_size": 224,
    #     "type": "efficientnet",
    #     "default_lr": 0.00005,
    #     "default_batch_size": 512
    # },
    "efficientnetv2_rw_s.ra2_in1k": {
        "name": "efficientnetv2_rw_s.ra2_in1k",
        "input_size": 224,
        "type": "efficientnet",
        "default_lr": 0.00005,
        "default_batch_size": 512
    },
    
    # MambaOut (4e-3 from original paper for ImageNet classification)
    "mambaout_tiny.in1k": {
        "name": "mambaout_tiny.in1k",
        "input_size": 224,
        "type": "mamba",
        "default_lr": 0.004,
        "default_batch_size": 4096
    },
    "mambaout_small.in1k": {
        "name": "mambaout_small.in1k",
        "input_size": 224,
        "type": "mamba",
        "default_lr": 0.004,
        "default_batch_size": 4096
    },
    
    # FocalNet (1e-4)
    "focalnet_tiny_lrf.ms_in1k": {
        "name": "focalnet_tiny_lrf.ms_in1k",
        "input_size": 224,
        "type": "focalnet",
        "default_lr": 0.0001,
        "default_batch_size": 512
    },
    # "focalnet_small_lrf.ms_in1k": {
    #     "name": "focalnet_small_lrf.ms_in1k",
    #     "input_size": 224,
    #     "type": "focalnet",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 512
    # },
    
    # EdgeNeXt (1e-3 for fine-tuning, original paper used 6e-3 for training from scratch)
    "edgenext_xx_small.in1k": {
        "name": "edgenext_xx_small.in1k",
        "input_size": 256,
        "type": "edgenext",
        "default_lr": 0.001,
        "default_batch_size": 512
    },
    "edgenext_x_small.in1k": {
        "name": "edgenext_x_small.in1k",
        "input_size": 256,
        "type": "edgenext",
        "default_lr": 0.001,
        "default_batch_size": 512
    },
    # "edgenext_small.usi_in1k": {  # Keeping x_small and base for size coverage
    #     "name": "edgenext_small.usi_in1k",
    #     "input_size": 256,
    #     "type": "edgenext",
    #     "default_lr": 0.001,
    #     "default_batch_size": 512
    # },
    "edgenext_base.in21k_ft_in1k": {
        "name": "edgenext_base.in21k_ft_in1k",
        "input_size": 256,
        "type": "edgenext",
        "default_lr": 0.001,
        "default_batch_size": 512
    },
    # "edgenext_base.usi_in1k": {  # Keeping edgenext_base.in21k_ft_in1k instead
    #     "name": "edgenext_base.usi_in1k",
    #     "input_size": 256,
    #     "type": "edgenext",
    #     "default_lr": 0.001,
    #     "default_batch_size": 512
    # },
    
    # TinyViT (1e-4)
    # "tiny_vit_5m_224.in1k": {  # Already have ViT coverage with vit_small
    #     "name": "tiny_vit_5m_224.in1k",
    #     "input_size": 224,
    #     "type": "tinyvit",
    #     "default_lr": 0.0001,
    #     "default_batch_size": 512
    # },
    
    # Volo (1e-3 for fine-tuning, original paper used 1.6e-3)
    # "volo_d1_224.sail_in1k": {  # Focus on core high-performers
    #     "name": "volo_d1_224.sail_in1k",
    #     "input_size": 224,
    #     "type": "volo",
    #     "default_lr": 0.001,
    #     "default_batch_size": 1024
    # },
    
    # Swin Transformer (1e-3 from original paper)
    "swin_tiny_patch4_window7_224": {
        "name": "swin_tiny_patch4_window7_224",
        "input_size": 224,
        "type": "swin",
        "default_lr": 0.001,
        "default_batch_size": 1024
    },
    "swin_base_patch4_window7_224": {
        "name": "swin_base_patch4_window7_224",
        "input_size": 224,
        "type": "swin",
        "default_lr": 0.001,
        "default_batch_size": 1024
    }
}

def get_architecture_names():
    """Get list of architecture names"""
    return list(ARCHITECTURES.keys())

def get_default_input_size(architecture_name):
    """Get default input size for an architecture"""
    return ARCHITECTURES.get(architecture_name, {}).get("input_size", 224)

def get_architecture_info(architecture_name):
    """Get full architecture information"""
    return ARCHITECTURES.get(architecture_name, {})

def get_default_learning_rate(model_name):
    """Get the default learning rate for a model"""
    # Try exact match first
    architecture_info = ARCHITECTURES.get(model_name, {})
    if architecture_info:
        return architecture_info.get("default_lr", 0.001)
    
    # If not found, try base name (split by first dot)
    base_name = model_name.split('.')[0]
    for arch_name in ARCHITECTURES.keys():
        if arch_name.split('.')[0] == base_name:
            return ARCHITECTURES[arch_name]["default_lr"]
    
    # Fallback to default
    return 0.001

def get_scaled_learning_rate(model_name, actual_batch_size, scaling_rule="sqrt"):
    """
    Scale learning rate based on batch size using linear or square root scaling
    
    Args:
        model_name: Model architecture name
        actual_batch_size: Batch size being used for training
        scaling_rule: Either 'linear' or 'sqrt' scaling rule
        
    Returns:
        Scaled learning rate appropriate for the given batch size
    """
    # Get architecture info
    architecture_info = ARCHITECTURES.get(model_name, {})
    if not architecture_info:
        # Try base name fallback
        base_name = model_name.split('.')[0]
        for arch_name in ARCHITECTURES.keys():
            if arch_name.split('.')[0] == base_name:
                architecture_info = ARCHITECTURES[arch_name]
                break
    
    # Get base LR and default batch size
    base_lr = architecture_info.get("default_lr", 0.001)
    default_batch_size = architecture_info.get("default_batch_size", 16)
    
    # Calculate scaling ratio
    ratio = actual_batch_size / default_batch_size
    
    # Apply scaling rule
    if scaling_rule == "linear":
        scaled_lr = base_lr * ratio
    elif scaling_rule == "sqrt":
        scaled_lr = base_lr * (ratio ** 0.5)
    else:
        raise ValueError("scaling_rule must be 'linear' or 'sqrt'")
    
    return scaled_lr

def get_enabled_architectures():
    """Get list of enabled (uncommented) architectures from ARCHITECTURES dict"""
    return list(ARCHITECTURES.keys())