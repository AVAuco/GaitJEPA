# (c) MJMJ/2026
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: Union[int, Sequence[int]] = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: Union[int, Sequence[int]] = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SetBlockWrapper(nn.Module):
    """Applies a 2D module across frames of a 5D video tensor [B, C, T, H, W]."""

    def __init__(self, forward_block: nn.Module):
        super().__init__()
        self.forward_block = forward_block

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        n, c, s, h, w = x.size()
        out = self.forward_block(x.transpose(1, 2).reshape(-1, c, h, w), *args, **kwargs)
        _, c_out, h_out, w_out = out.size()
        return out.reshape(n, s, c_out, h_out, w_out).transpose(1, 2).contiguous()


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Sequence[int] = (1, 1),
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, tuple(stride))
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class BasicBlockP3D(nn.Module):
    """Pseudo-3D basic block combining 2D spatial convolutions with temporal 3D shortcut."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Sequence[int] = (1, 1),
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(inplanes, planes, tuple(stride)),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            )
        )
        self.conv2 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(planes, planes),
                nn.BatchNorm2d(planes),
            )
        )
        self.shortcut3d = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=False)
        self.sbn = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out + self.sbn(self.shortcut3d(out)))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: Sequence[int] = (1, 1, 1),
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        temporal_padding = 1 if int(stride[0]) <= 2 else 0
        self.conv1 = nn.Conv3d(
            inplanes,
            planes,
            kernel_size=(3, 3, 3),
            stride=tuple(stride),
            padding=(temporal_padding, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class HorizontalPoolingPyramid:
    """Horizontal pooling pyramid (HPP) dividing spatial feature maps into horizontal strips."""

    def __init__(self, bin_num: Optional[Sequence[int]] = None):
        self.bin_num = list(bin_num or [16])

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n, c = x.size()[:2]
        features = []
        for bins in self.bin_num:
            if (x.shape[2] * x.shape[3]) % bins != 0:
                raise ValueError(
                    f"HPP requires (H*W) divisible by bins. Got H={x.shape[2]}, W={x.shape[3]}, bins={bins}."
                )
            z = x.view(n, c, bins, -1)
            z = z.mean(dim=-1) + z.max(dim=-1)[0]
            features.append(z)
        return torch.cat(features, dim=-1)


class SeparateFCs(nn.Module):
    """Separate linear mappings per horizontal part bin."""

    def __init__(self, parts_num: int, in_channels: int, out_channels: int, norm: bool = False):
        super().__init__()
        self.fc_bin = nn.Parameter(torch.zeros(parts_num, in_channels, out_channels))
        nn.init.xavier_uniform_(self.fc_bin)
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1).contiguous()
        weights = F.normalize(self.fc_bin, dim=1) if self.norm else self.fc_bin
        out = x.matmul(weights)
        return out.permute(1, 2, 0).contiguous()


class DeepGaitV2JEPAEncoder(nn.Module):
    """
    Minimal standalone encoder for GaitJEPA (IJCB 2026).

    Input:
        silhouettes: Tensor of shape [B, T, 1, 64, 44] (or [B, 1, T, 64, 44]) with values in [0.0, 1.0].
    Output representations:
        - global_embedding: Normalized global embedding vector [B, 256].
        - pooled_part_embedding: Part embeddings [B, 256, 16] (dimension x parts).
        - sequence_part_tokens: Temporal part tokens [B, T, 16, 256] (batch x time x parts x dim).
    """

    def __init__(
        self,
        in_channels: int = 1,
        backbone_mode: str = "p3d",
        layers: Sequence[int] = (1, 1, 1, 1),
        channels: Sequence[int] = (64, 128, 256, 512),
        hpp_bins: Sequence[int] = (16,),
        embed_dim: int = 256,
        separate_fcs_out_dim: int = 256,
        temporal_pool: str = "max",
    ):
        super().__init__()
        self.backbone_mode = str(backbone_mode).lower()
        self.hpp_bins = tuple(int(v) for v in hpp_bins)
        self.num_parts = sum(self.hpp_bins)
        self.embed_dim = int(embed_dim)
        self.part_dim = int(separate_fcs_out_dim)
        self.temporal_pool = str(temporal_pool).lower()
        self.out_channels = int(channels[-1])

        if self.backbone_mode not in {"2d", "p3d", "3d"}:
            raise ValueError(f"Unsupported backbone_mode: {backbone_mode}")

        block_map = {"2d": BasicBlock2D, "p3d": BasicBlockP3D, "3d": BasicBlock3D}
        block = block_map[self.backbone_mode]

        if self.backbone_mode == "3d":
            strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (1, 1, 1)]
        else:
            strides = [(1, 1), (2, 2), (2, 2), (1, 1)]

        self.inplanes = int(channels[0])
        self.layer0 = SetBlockWrapper(
            nn.Sequential(
                conv3x3(in_channels, self.inplanes),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
            )
        )
        self.layer1 = SetBlockWrapper(self._make_layer(BasicBlock2D, int(channels[0]), strides[0], int(layers[0]), mode="2d"))
        self.layer2 = self._make_layer(block, int(channels[1]), strides[1], int(layers[1]), mode=self.backbone_mode)
        self.layer3 = self._make_layer(block, int(channels[2]), strides[2], int(layers[2]), mode=self.backbone_mode)
        self.layer4 = self._make_layer(block, int(channels[3]), strides[3], int(layers[3]), mode=self.backbone_mode)

        if self.backbone_mode == "2d":
            self.layer2 = SetBlockWrapper(self.layer2)
            self.layer3 = SetBlockWrapper(self.layer3)
            self.layer4 = SetBlockWrapper(self.layer4)

        self.hpp = HorizontalPoolingPyramid(self.hpp_bins)
        self.part_proj = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.Linear(self.out_channels, self.embed_dim),
        )
        self.fcs = SeparateFCs(self.num_parts, self.out_channels, self.part_dim)

    def _make_layer(self, block, planes: int, stride: Sequence[int], blocks_num: int, mode: str) -> nn.Sequential:
        if mode == "3d":
            max_stride = max(stride)
            if max_stride > 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=tuple(stride), bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = None
        elif mode == "p3d":
            max_stride = max(stride)
            if max_stride > 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, int(stride[0]), int(stride[1])),
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = None
        else:
            max_stride = max(stride)
            if max_stride > 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, tuple(stride)),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                downsample = None

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        same_stride = (1, 1, 1) if mode == "3d" else (1, 1)
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=same_stride))
        return nn.Sequential(*layers)

    def _temporal_pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_pool == "mean":
            return x.mean(dim=2)
        return x.max(dim=2).values

    def _standardize_input(self, silhouettes: torch.Tensor) -> torch.Tensor:
        """Ensure input tensor is [B, C, T, H, W]."""
        if silhouettes.dim() == 4:
            # Assume [T, 1, H, W] -> unbatched
            silhouettes = silhouettes.unsqueeze(0)
        if silhouettes.dim() != 5:
            raise ValueError(f"Expected 4D or 5D silhouette tensor, got shape {tuple(silhouettes.shape)}")

        # If [B, T, C, H, W], transpose to [B, C, T, H, W]
        # In gait benchmarks, standard input is [B, T, 1, 64, 44]
        if silhouettes.shape[2] == 1 and silhouettes.shape[1] != 1:
            sils = silhouettes.transpose(1, 2).contiguous()
        elif silhouettes.shape[1] == 1:
            sils = silhouettes
        else:
            # Fallback: if channel is at dim 2
            sils = silhouettes.transpose(1, 2).contiguous()
        return sils

    def forward(
        self,
        silhouettes: torch.Tensor,
        visible_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract gait representations from silhouette sequences.

        Args:
            silhouettes: Tensor [B, T, 1, 64, 44] (or [B, 1, T, 64, 44]).
            visible_mask: Optional boolean or float mask [B, T] for masking frames.

        Returns:
            Dict containing:
                - "global_embedding": [B, 256] L2-normalized embedding.
                - "pooled_part_embedding": [B, 256, 16] part features.
                - "sequence_part_tokens": [B, T, 16, 256] unflattened spatiotemporal tokens.
                - "tokens": [B, T, 16, 256] alias for sequence_part_tokens.
                - "feature_maps": [B, 512, T, H', W'] backbone feature maps.
        """
        sils = self._standardize_input(silhouettes)

        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        b, c, t, h, w = out4.shape
        frame_maps = out4.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        frame_parts = self.hpp(frame_maps).permute(0, 2, 1).contiguous()
        sequence_part_tokens = self.part_proj(frame_parts).view(b, t, self.num_parts, self.embed_dim)

        if visible_mask is not None:
            sequence_part_tokens = sequence_part_tokens * visible_mask[:, :, None, None].float()

        pooled_maps = self._temporal_pool(out4)
        pooled_parts = self.hpp(pooled_maps)
        pooled_part_embedding = self.fcs(pooled_parts)
        global_embedding = F.normalize(pooled_part_embedding.mean(dim=-1), dim=-1)

        return {
            "tokens": sequence_part_tokens,
            "sequence_part_tokens": sequence_part_tokens,
            "pooled_part_embedding": pooled_part_embedding,
            "global_embedding": global_embedding,
            "feature_maps": out4,
        }

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatiotemporal part tokens flattened to [B, T * 16, 256].
        """
        out = self.forward(x, visible_mask=None)
        tokens = out["sequence_part_tokens"]
        return tokens.view(tokens.shape[0], -1, tokens.shape[-1])

    def forward_features(self, x: torch.Tensor, pool: Union[bool, str] = True) -> torch.Tensor:
        """
        Main feature extraction interface, exactly matching DeepGaitV2Hybrid.forward_features().

        Args:
            x: Silhouette sequence [B, T, 1, 64, 44].
            pool: Pooling type:
                - True or "global": Normalized global embedding [B, 256].
                - False or "parts": Part embeddings [B, 256, 16].
                - "tokens": Flattened part tokens [B, T * 16, 256].
                - "unflattened_tokens": Part tokens [B, T, 16, 256].
                - "max": Max-pooled over parts [B, 256].

        Returns:
            Extracted feature tensor.
        """
        out = self.forward(x, visible_mask=None)
        part_embeddings = out["pooled_part_embedding"]
        if pool is False:
            return part_embeddings
        if pool is True:
            return out["global_embedding"]

        selected_pool = str(pool).lower()
        if selected_pool == "parts":
            return part_embeddings
        if selected_pool == "tokens":
            return self.forward_tokens(x)
        if selected_pool == "unflattened_tokens":
            return out["sequence_part_tokens"]
        if selected_pool == "max":
            return F.normalize(part_embeddings.max(dim=-1).values, dim=-1)
        return out["global_embedding"]
