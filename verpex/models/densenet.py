"""3D DenseNet backbones producing per-landmark heatmaps.

Adapted from MONAI's DenseNet implementation. The final flattening and output
layers are removed, because this model produces a downsized feature map rather
than a classification.

See https://docs.monai.io/en/stable/_modules/monai/networks/nets/densenet.html
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence

import torch
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from torch import nn


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """Initialise the module.

        Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        growth_rate: how many filters to add each layer (k in paper).
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        dropout_prob: dropout rate after each dense layer.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module(
            "norm1",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        )
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module(
            "norm2",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels),
        )
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module(
            "conv2",
            conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """Initialise the module.

        Args:
        spatial_dims: number of spatial dimensions of the input image.
        layers: number of layers in the block.
        in_channels: number of the input channel.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        growth_rate: how many filters to add each layer (k in paper).
        dropout_prob: dropout rate after each dense layer.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(
                spatial_dims,
                in_channels,
                growth_rate,
                bn_size,
                dropout_prob,
                act=act,
                norm=norm,
            )
            in_channels += growth_rate
            self.add_module(f"denselayer{i + 1}", layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """Initialise the module.

        Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module(
            "norm",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        )
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class _TransitionNoPool(nn.Sequential):
    """A `_Transition` without the 2x spatial pool.

    Halves the channel count between dense blocks without downsampling further, so
    the final heatmap comes out at higher resolution.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]

        self.add_module(
            "norm",
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
        )
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))


class HeatmapDenseNet(nn.Module):
    """Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_landmarks: int,
        init_features: int = 64,
        feature_l: int = 256,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
        weight_features: bool = True,
        skip_last_transition_pool: bool = False,
    ) -> None:
        super().__init__()

        self.n_landmarks = n_landmarks
        self.feature_l = feature_l
        self.skip_last_transition_pool = skip_last_transition_pool

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        conv_type(
                            in_channels,
                            init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    (
                        "norm0",
                        get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features),
                    ),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5",
                    get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels),
                )
            else:
                _out_channels = in_channels // 2
                # For the transition right before the final dense block, optionally
                # skip the 2x spatial pool so the heatmap ends up at 2x resolution.
                is_last_transition = i == len(block_config) - 2
                trans_cls = _TransitionNoPool if (is_last_transition and self.skip_last_transition_pool) else _Transition
                trans = trans_cls(
                    spatial_dims,
                    in_channels=in_channels,
                    out_channels=_out_channels,
                    act=act,
                    norm=norm,
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # Final convolution to produce n_landmarks heatmaps + feature_l feature maps simultaneously
        self.features.add_module(
            "conv_final",
            conv_type(in_channels, n_landmarks + feature_l, kernel_size=1, bias=False),
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

        self.weight_features = weight_features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-landmark heatmaps and the shared feature map.

        Args:
            x: Input volume, ``(batch, channels, depth, height, width)``.

        Returns:
            A tuple of the landmark heatmaps and the feature map they were split from.
        """
        x = x.float()  # Ensure input is float
        x = self.features(x)  # (batch_size, n_landmarks + feature_l, *spatial_shape // (len(block_config)**2))
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
        if self.weight_features:
            # weight the feature maps with the heatmaps
            landmark_features = (normalized_heatmaps.unsqueeze(2).detach() * feature_map_expanded).sum(dim=(3, 4, 5))

        else:
            # global average pooling of feature maps (no heatmap weighting)
            global_features = feature_map.mean(dim=(2, 3, 4))  # (B, feature_l)
            landmark_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, N, feature_l)

        return normalized_heatmaps, landmark_features, feature_map


class UNetHeatmapDenseNet(nn.Module):
    """DenseNet encoder with a one-stage U-Net decoder for higher-resolution heatmaps.

    Lifts the heatmap to twice the encoder's output resolution - 32x32x36 with the
    default 128x128x144 input and ``block_config=[6, 12, 12]``. Skips the encoder's
    last transition pool (as ``skip_last_transition_pool=True`` does on
    :class:`HeatmapDenseNet`), then adds one transposed-convolution upsample fused
    with the skip connection from the first dense block.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        n_landmarks: int,
        init_features: int = 64,
        feature_l: int = 256,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 12),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
        weight_features: bool = True,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()

        assert spatial_dims == 3, "UNetHeatmapDenseNet is 3D-only"
        assert len(block_config) == 3, "expected 3 dense blocks"

        self.n_landmarks = n_landmarks
        self.feature_l = feature_l
        self.weight_features = weight_features

        conv_type = Conv[Conv.CONV, spatial_dims]
        pool_type = Pool[Pool.MAX, spatial_dims]
        tconv_type = nn.ConvTranspose3d

        # ----- Stem -----
        self.stem = nn.Sequential(
            conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features),
            get_act_layer(name=act),
            pool_type(kernel_size=3, stride=2, padding=1),
        )
        # After stem: spatial 128 → 32, channels = init_features (64)

        # ----- Encoder blocks -----
        c = init_features
        self.denseblock1 = _DenseBlock(spatial_dims, block_config[0], c, bn_size, growth_rate, dropout_prob, act, norm)
        c1 = c + block_config[0] * growth_rate  # skip1 channels (32x32x36)
        self.transition1 = _Transition(spatial_dims, c1, c1 // 2, act, norm)  # downsample to 16x16x18
        c = c1 // 2

        self.denseblock2 = _DenseBlock(spatial_dims, block_config[1], c, bn_size, growth_rate, dropout_prob, act, norm)
        c2 = c + block_config[1] * growth_rate
        # No pool on this transition — keep 16x16x18
        self.transition2 = _TransitionNoPool(spatial_dims, c2, c2 // 2, act, norm)
        c = c2 // 2

        self.denseblock3 = _DenseBlock(spatial_dims, block_config[2], c, bn_size, growth_rate, dropout_prob, act, norm)
        c3 = c + block_config[2] * growth_rate  # deep channels (16x16x18)

        # ----- Decoder: one 2x upsample fused with skip1 → 32x32x36 -----
        self.decoder_up = tconv_type(c3, decoder_channels, kernel_size=2, stride=2)
        # After upsample: (B, decoder_channels, 32, 32, 36)
        # Concat with skip1 (c1 channels) → (decoder_channels + c1)
        self.decoder_refine = nn.Sequential(
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=decoder_channels + c1),
            get_act_layer(name=act),
            conv_type(decoder_channels + c1, decoder_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=decoder_channels),
            get_act_layer(name=act),
        )
        # ----- Head -----
        self.conv_final = conv_type(decoder_channels, n_landmarks + feature_l, kernel_size=1, bias=False)

        # Kaiming init for convs
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-landmark heatmaps and features via a U-Net decoder.

        Args:
            x: Input volume, ``(batch, channels, depth, height, width)``.

        Returns:
            A tuple of the landmark heatmaps and the feature map they were split from.
        """
        x = x.float()
        s0 = self.stem(x)  # (B, 64, 32, 32, 36)
        b1 = self.denseblock1(s0)  # (B, c1, 32, 32, 36)   ← skip
        t1 = self.transition1(b1)  # (B, c1/2, 16, 16, 18)
        b2 = self.denseblock2(t1)  # (B, c2, 16, 16, 18)
        t2 = self.transition2(b2)  # (B, c2/2, 16, 16, 18) — no pool
        b3 = self.denseblock3(t2)  # (B, c3, 16, 16, 18)

        up = self.decoder_up(b3)  # (B, decoder_channels, 32, 32, 36)
        fused = torch.cat([up, b1], dim=1)  # skip concat
        dec = self.decoder_refine(fused)  # (B, decoder_channels, 32, 32, 36)

        out = self.conv_final(dec)  # (B, N+F, 32, 32, 36)
        heatmaps, feature_map = out.split([self.n_landmarks, self.feature_l], dim=1)

        B, N, *spatial = heatmaps.shape
        normalized_heatmaps = torch.softmax(heatmaps.view(B, N, -1), dim=-1).view(B, N, *spatial)

        feature_map_expanded = feature_map.unsqueeze(1)
        if self.weight_features:
            landmark_features = (normalized_heatmaps.unsqueeze(2).detach() * feature_map_expanded).sum(dim=(3, 4, 5))
        else:
            global_features = feature_map.mean(dim=(2, 3, 4))
            landmark_features = global_features.unsqueeze(1).expand(-1, N, -1)

        return normalized_heatmaps, landmark_features, feature_map
