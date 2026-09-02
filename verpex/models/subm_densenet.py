"""Sparse-convolution DenseNet backbones, for mostly-empty input volumes.

Adapted from MONAI's DenseNet3D. Relative to the dense version: convolutions
become submanifold convolutions, batch norms and pooling become their sparse
equivalents, the final flattening and output layers are removed, and a final
convolution produces the landmark heatmaps and feature maps together.

See https://arxiv.org/pdf/1608.06993.pdf and
https://docs.monai.io/en/stable/_modules/monai/networks/nets/densenet.html
"""

from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from monai.data.meta_tensor import MetaTensor
from spconv.pytorch.modules import SparseModule
from torch import nn


class SubmDenseLayer(spconv.SparseModule):
    """One sparse dense-block layer: bottleneck 1x1x1 then a 3x3x3 submanifold conv.

    Args:
        in_channels: Channels entering the layer.
        growth_rate: Channels this layer contributes to the block output.
        bn_size: Bottleneck width multiplier for the 1x1x1 convolution.
        dropout: Dropout probability applied to the layer output.
    """

    def __init__(self, in_channels, growth_rate, bn_size, dropout):
        super().__init__()
        out_channels = bn_size * growth_rate
        self.layers = spconv.SparseSequential(
            spconv.SparseBatchNorm(num_features=in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, 1, padding=1, bias=False),
            spconv.SparseBatchNorm(num_features=out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, growth_rate, 3, padding=1, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x) -> torch.Tensor:
        """Run the module on a sparse tensor and return its output."""
        new_features = self.layers(x)
        return x.replace_feature(torch.cat([x.features, new_features.features], 1))


class SubmDenseBlock(spconv.SparseSequential):
    """A stack of :class:`SubmDenseLayer`, each seeing all preceding outputs.

    Args:
        layers: Number of layers in the block.
        in_channels: Channels entering the block.
        bn_size: Bottleneck width multiplier.
        growth_rate: Channels each layer contributes.
        dropout: Dropout probability.
    """

    def __init__(self, layers, in_channels, bn_size, growth_rate, dropout):
        super().__init__()

        for i in range(layers):
            layer = SubmDenseLayer(in_channels, growth_rate, bn_size, dropout)
            in_channels += growth_rate
            self.add_module(f"SubmDenseLayer{i + 1}", layer)


class SubmTransition(SparseModule):
    """Transition between dense blocks: 1x1x1 conv then a stride-2 pool.

    Args:
        in_channels: Channels entering the transition.
        out_channels: Channels leaving it.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = spconv.SparseSequential(
            spconv.SparseBatchNorm(num_features=in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, 1, bias=False),
            spconv.SparseAvgPool3d(2, 2),
        )

    def forward(self, x) -> torch.Tensor:
        """Run the module on a sparse tensor and return its output."""
        return self.layers(x)


class HeatmapSubmDenseNet(SparseModule):
    """Sparse-convolution DenseNet producing one heatmap per landmark.

    The sparse counterpart of :class:`~verpex.models.densenet.HeatmapDenseNet`,
    for inputs where most voxels are background.
    """

    def __init__(
        self,
        in_channels: int,
        n_landmarks: int,
        init_features: int = 64,
        feature_l: int = 256,
        growth_rate: int = 32,
        block_config=(6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        self.feature_l = feature_l
        self.features = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, init_features, 7, padding=3, bias=False),
            spconv.SparseBatchNorm(64),
            nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = SubmDenseBlock(num_layers, in_channels, bn_size, growth_rate, dropout_prob)
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(f"norm{i + 1}", spconv.SparseBatchNorm(in_channels))
            else:
                _out_channels = in_channels // 2
                transition = SubmTransition(in_channels, _out_channels)
                self.features.add_module(f"transition{i + 1}", transition)
                in_channels = _out_channels

        self.features.add_module("todense", spconv.ToDense())

        # Add 3 normal convolutions (kernel size 3) to create a dense feature map of depth n_landmarks + feature_l
        self.features.add_module(
            "final_convs",
            torch.nn.Sequential(
                torch.nn.Conv3d(in_channels, in_channels, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels, in_channels, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(in_channels, n_landmarks + feature_l, 3, padding=1),
            ),
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the module on a sparse tensor and return its output."""
        x = x.float()
        # The sparse library does not like monai Metatensors, so we convert to torch tensor
        if isinstance(x, MetaTensor):
            x = x.as_tensor()
        # Convert x to spconv.SparseConvTensor
        x = x.squeeze(1)
        batch_size, *spatial_shape = x.shape
        x = x.to_sparse()
        idcs = x.indices().t().int()
        feats = x.values().unsqueeze(1)
        x = spconv.SparseConvTensor(feats, idcs, spatial_shape=spatial_shape, batch_size=batch_size)
        x = self.features(x)
        # Split the output into landmark heatmaps and feature maps
        heatmaps, feature_map = x.split([self.n_landmarks, self.feature_l], dim=1)

        # Normalize the heatmaps so that they sum to 1 by applying spatial softmax
        B, N, *spatial_shape = heatmaps.shape
        normalized_heatmaps = heatmaps.view(B, N, -1)
        normalized_heatmaps = torch.softmax(normalized_heatmaps, dim=-1)
        normalized_heatmaps = normalized_heatmaps.view(B, N, *spatial_shape)

        # Apply the heatmaps to the feature maps to get the feature encoding for each landmark
        # Expand feature map dimensions from (B, F, H, W, D) to (B, 1, F, H, W, D) for broadcasting
        feature_map_expanded = feature_map.unsqueeze(1)

        # Perform weighted sum
        # heatmaps_normalized is (B, N, H, W, D)
        # feature_map_expanded is (B, 1, F, H, W, D)
        # We want the output to be (B, N, F), summing over the spatial dimensions (H, W, D)
        weighted_features = (normalized_heatmaps.unsqueeze(2) * feature_map_expanded).sum(dim=(3, 4, 5))

        return normalized_heatmaps, weighted_features


class SubmDenseNet(SparseModule):
    """Adapted from MonAI's implementation of DenseNet3D: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Does not predict heatmaps, but direct coordinate estimates and landmark features.
    """

    def __init__(
        self,
        in_channels: int,
        n_landmarks: int,
        init_features: int = 64,
        feature_l: int = 256,
        growth_rate: int = 32,
        block_config=(6, 12, 24, 16),
        bn_size: int = 4,
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_landmarks = n_landmarks
        self.feature_l = feature_l
        self.features = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, init_features, 7, padding=3, bias=False),
            spconv.SparseBatchNorm(64),
            nn.ReLU(),
            spconv.SparseMaxPool3d(2, 2),
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = SubmDenseBlock(num_layers, in_channels, bn_size, growth_rate, dropout_prob)
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(f"norm{i + 1}", spconv.SparseBatchNorm(in_channels))
            else:
                _out_channels = in_channels // 2
                transition = SubmTransition(in_channels, _out_channels)
                self.features.add_module(f"transition{i + 1}", transition)
                in_channels = _out_channels

        # pooling and Linear layer
        self.features.add_module("todense", spconv.ToDense())
        self.feature_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", nn.ReLU()),
                    ("avg_pool", nn.AdaptiveAvgPool3d(1)),
                    ("flatten", nn.Flatten(1)),
                    ("fc", nn.Linear(in_channels, n_landmarks * (3 + feature_l))),
                ]
            )
        )

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the module on a sparse tensor and return its output."""
        x = x.float()
        # The sparse library does not like monai Metatensors, so we convert to torch tensor
        if isinstance(x, MetaTensor):
            x = x.as_tensor()
        # Convert x to spconv.SparseConvTensor
        x = x.squeeze(1)
        batch_size, *spatial_shape = x.shape
        x = x.to_sparse()
        idcs = x.indices().t().int()
        feats = x.values().unsqueeze(1)
        x = spconv.SparseConvTensor(feats, idcs, spatial_shape=spatial_shape, batch_size=batch_size)
        x = self.features(x)
        x = self.feature_layers(x)  # (B, N_landmarks * (3 + feature_l))

        x = x.reshape(x.shape[0], self.n_landmarks, -1)
        # Split the output into landmark coordinates and feature maps
        coord_estimates, landmark_features = x.split([3, self.feature_l], dim=-1)  # (B, N_landmarks, 3), (B, N_landmarks, feature_l)

        return coord_estimates, landmark_features
