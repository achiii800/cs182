"""
Model architectures for experiments.

Provides:
    - SmallConvNet: Tiny CNN for quick iteration
    - ResNet18CIFAR: ResNet-18 adapted for CIFAR-10 (32x32 images)
    - TinyViT: Small Vision Transformer for CIFAR
    - MLPMixer: MLP-Mixer architecture
    - WidthScalableMLP: MLP with configurable width for muP experiments
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


# =============================================================================
# SmallConvNet (Original baseline)
# =============================================================================

class SmallConvNet(nn.Module):
    """
    Tiny CNN for CIFAR-10. Fast for inner-solver debugging.

    Architecture:
        Conv(3->32) -> ReLU -> Conv(32->64) -> ReLU -> Pool
        -> Pool -> FC(64*8*8->256) -> ReLU -> FC(256->10)
    """

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        c1 = int(32 * width_mult)
        c2 = int(64 * width_mult)
        fc1_dim = int(256 * width_mult)

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(c2 * 8 * 8, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))           # [B, c1, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # [B, c2, 16, 16]
        x = self.pool(x)                    # [B, c2, 8, 8]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =============================================================================
# ResNet-18 for CIFAR (adapted: smaller stem, no early pooling)
# =============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).

    Modifications from ImageNet ResNet:
        - First conv: 3x3 kernel, stride 1, no max pooling
        - Smaller feature maps throughout
        - Width multiplier for muP experiments
    """

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()

        self.in_planes = int(64 * width_mult)
        base_width = int(64 * width_mult)

        # Stem: single 3x3 conv (no 7x7, no pool)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        # Residual layers
        self.layer1 = self._make_layer(base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * BasicBlock.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =============================================================================
# Tiny Vision Transformer (ViT) for CIFAR
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Standard transformer block: Attention + FFN with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """
    Small Vision Transformer for CIFAR-10.

    Architecture:
        - Patch embedding (patch_size x patch_size patches)
        - Learnable position embeddings
        - N transformer blocks
        - Global average pooling + linear head

    Default: 4x4 patches -> 64 patches, dim=128, 4 blocks, 4 heads
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        width_mult: float = 1.0,
    ):
        super().__init__()

        dim = int(dim * width_mult)
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        # Patch embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size

        # Patchify: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C * P * P)

        # Patch embedding
        x = self.patch_embed(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Classification head (use CLS token)
        cls_out = x[:, 0]
        return self.head(cls_out)


# =============================================================================
# MLP-Mixer for CIFAR
# =============================================================================

class MixerBlock(nn.Module):
    """MLP-Mixer block: token mixing + channel mixing."""

    def __init__(self, num_patches: int, dim: int, token_mlp_dim: int, channel_mlp_dim: int):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = nn.Sequential(
            nn.Linear(num_patches, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_patches),
        )

        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = nn.Sequential(
            nn.Linear(dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing (across patches)
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, dim, num_patches)
        y = self.token_mix(y)
        y = y.transpose(1, 2)  # (B, num_patches, dim)
        x = x + y

        # Channel mixing
        y = self.norm2(x)
        y = self.channel_mix(y)
        x = x + y

        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer for CIFAR-10.

    Architecture:
        - Patch embedding
        - N Mixer blocks (token mixing + channel mixing)
        - Global average pooling + linear head
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 128,
        depth: int = 4,
        token_mlp_dim: int = 64,
        channel_mlp_dim: int = 256,
        width_mult: float = 1.0,
    ):
        super().__init__()

        dim = int(dim * width_mult)
        token_mlp_dim = int(token_mlp_dim * width_mult)
        channel_mlp_dim = int(channel_mlp_dim * width_mult)

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, dim)

        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, dim, token_mlp_dim, channel_mlp_dim)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size

        # Patchify
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C * P * P)

        # Patch embedding
        x = self.patch_embed(x)

        # Mixer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)
        return self.head(x)


# =============================================================================
# Width-Scalable MLP for muP experiments
# =============================================================================

class WidthScalableMLP(nn.Module):
    """
    Simple MLP with configurable width for muP transfer experiments.

    Tests whether hyperparameters (LR, spectral budget) transfer across widths
    when using spectral-norm constrained optimizers.
    """

    def __init__(
        self,
        input_dim: int = 3072,  # 32x32x3 flattened
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 10,
        width_mult: float = 1.0,
        activation: str = "relu",
    ):
        super().__init__()

        hidden_dim = int(hidden_dim * width_mult)

        layers = []
        in_dim = input_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)


# =============================================================================
# Model registry
# =============================================================================

MODEL_REGISTRY = {
    "small_cnn": SmallConvNet,
    "resnet18": ResNet18CIFAR,
    "tiny_vit": TinyViT,
    "mlp_mixer": MLPMixer,
    "mlp": WidthScalableMLP,
}


def get_model(
    name: str,
    num_classes: int = 10,
    width_mult: float = 1.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        name: One of 'small_cnn', 'resnet18', 'tiny_vit', 'mlp_mixer', 'mlp'.
        num_classes: Number of output classes.
        width_mult: Width multiplier for muP experiments.
        **kwargs: Additional model-specific arguments.

    Returns:
        Model instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[name](num_classes=num_classes, width_mult=width_mult, **kwargs)


__all__ = [
    "SmallConvNet",
    "ResNet18CIFAR",
    "TinyViT",
    "MLPMixer",
    "WidthScalableMLP",
    "MODEL_REGISTRY",
    "get_model",
]
