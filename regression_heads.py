from typing import Type
import torch
import torch.nn as nn
from timm.models.layers import Mlp


class SimpleRegressionHead(nn.Module):
    """
    Simple 2-layer MLP: C → C/2 → out_features
    Uses timm's Mlp with basic ReLU activation and dropout.
    """
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.net = Mlp(
            in_features=in_features,
            hidden_features=in_features//2,
            out_features=out_features,
            act_layer=torch.nn.ReLU,  # torch.nn.GELU
            drop=0.2
        )

    def forward(self, x):
        return self.net(x)


class ThreeLayerRegressionHead(nn.Module):
    """
    3-layer MLP: C → C/2 → C/4 → out_features
    Combines timm's Mlp with an additional linear layer.
    """
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.mlp = Mlp(
            in_features=in_features,
            hidden_features=in_features//2,
            out_features=in_features//4,
            act_layer=torch.nn.ReLU,  # torch.nn.GELU
            drop=0.2)
        self.fc = nn.Linear(in_features//4, out_features)

    def forward(self, x):
        x = self.mlp(x)
        x = self.fc(x)
        return x


class ProgressiveRegressionHead(nn.Module):
    """
    Progressive dimensionality reduction: C → C/2 → C/4 → out_features
    Each layer halves the feature dimension with dropout regularization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_layer: Type[nn.Module] = torch.nn.ReLU,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features//2),
            act_layer(),
            nn.Dropout(p=0.3),

            nn.Linear(in_features=in_features//2, out_features=in_features//4),
            act_layer(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=in_features//4, out_features=out_features)
        )

    def forward(self, x):
        return self.net(x)


class NormalizedRegressionHead(nn.Module):
    """
    2-layer MLP with extensive LayerNorm: C → 512 → 512 → out_features
    Features heavy normalization and fixed hidden dimension of 512.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 512,
        act_layer: Type[nn.Module] = torch.nn.ReLU,  # torch.nn.GELU
        drop: float = 0.2
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features, bias=False),
            act_layer(),
            nn.Dropout(drop),

            nn.LayerNorm(hidden_features),
            # bias=True by default
            nn.Linear(hidden_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),

            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        return self.net(x)


class ConfigurableRegressionHead(nn.Module):
    """
    Highly configurable regression head with optional MLP layers.
    
    Supports:
    - Configurable MLP dimensions or simple linear projection
    - Multiple normalization options (batch, layer, none)
    - Customizable activation and dropout
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mlp_dims: list[int] = None,
        act_layer: Type[nn.Module] = torch.nn.ReLU,
        dropout_rate: float = 0.2,
        normalization: str = "batch",  # "batch", "layer", or "none"
        final_activation: Type[nn.Module] = None,  # Optional final activation
    ):
        super().__init__()
        mlp_dims = mlp_dims or []

        # create normalization layer
        if normalization == "batch":
            norm_layer = nn.BatchNorm1d(in_features)
        elif normalization == "layer":
            norm_layer = nn.LayerNorm(in_features)
        else:
            norm_layer = nn.Identity()

        layers: list[nn.Module] = [norm_layer]

        if mlp_dims:
            dims = [in_features] + mlp_dims + [out_features]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(act_layer())
                    layers.append(nn.Dropout(dropout_rate * (0.5 ** i)))  # progressively reduce dropout
        else:
            # simple linear head
            layers.append(nn.Linear(in_features, out_features))

        # Add final activation if specified
        if final_activation is not None:
            layers.append(final_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TensorFlowStyleUnitVectorHead(nn.Module):
    """
    TensorFlow-style unit vector head that matches the original implementation.
    
    Architecture:
    - Dropout
    - Optional middle layer with ReLU
    - Final layer with tanh activation
    - No explicit L2 normalization (relies on MSE loss to learn unit vectors)
    """
    
    def __init__(
        self,
        in_features: int,
        n_neurons_middle_layer: int = 100,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        
        layers = []
        
        # Initial dropout
        # layers.append(nn.Dropout(dropout_rate))
        
        # Optional middle layer
        if n_neurons_middle_layer:
            layers.extend([
                nn.Linear(in_features, n_neurons_middle_layer),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_features = n_neurons_middle_layer
        
        # Final layer with tanh activation
        layers.extend([
            nn.Linear(in_features, 2),
            nn.Tanh()
        ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ConvolutionalRegressionHead(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_features, num_features//4, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(num_features//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features//4, num_features//16, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(num_features//16),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features//16, num_features//32, kernel_size=3, padding=3//2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(num_features//32, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.fc(x)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)
        return x


if __name__ == "__main__":
    in_features = 1024  # Example input feature size
    out_features = 2  # Example output size

    model = nn.Linear(in_features, 2)
    print(model)

    model = ConfigurableRegressionHead(
        in_features=in_features,
        out_features=out_features,
        #mlp_dims=[512, 256],       # two hidden layers: 512 → 256
        dropout_rate=0.1,          # small dropout for robustness
        normalization="layer"      # BatchNorm1d — plays nicely with CNN features, LayerNorm if using a ViT
    )
    print(model)

    model = NormalizedRegressionHead(in_features, out_features)
    print(model)

    #  Simple 2-layer head MLP head with dropout
    model = Mlp(
        in_features=in_features,
        hidden_features=512,
        out_features=2,
        act_layer=nn.GELU,
        drop=0.1
    )
    print(model)

    # For ConvolutionalRegressionHead, preserve spatial dimensions (no global pooling) with global_pool=''
    model = ConvolutionalRegressionHead(in_features, 2)
    print(model)
