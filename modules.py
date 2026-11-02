from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_count(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(channels: int, norm: str = "gn") -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm == "in":
        return nn.InstanceNorm3d(channels, affine=True)
    return nn.GroupNorm(get_group_count(channels), channels)


def resize_to(x: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    if x.shape[2:] == size:
        return x
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)


class ConvNormAct3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        norm: str = "gn",
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = make_norm(out_channels, norm)
        self.act = nn.GELU() if activate else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_norm = make_norm(in_channels, norm)
        self.depthwise_act = nn.GELU()
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_norm = make_norm(out_channels, norm)
        self.pointwise_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_act(self.depthwise_norm(self.depthwise(x)))
        x = self.pointwise_act(self.pointwise_norm(self.pointwise(x)))
        return x


class SqueezeExcite3d(nn.Module):

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.pool(x)
        weight = self.act(self.fc1(weight))
        weight = self.gate(self.fc2(weight))
        return x * weight


class ResidualConvBlock3D(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, norm: str = "gn"
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            norm=norm,
        )
        self.conv2 = ConvNormAct3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=norm,
            activate=False,
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvNormAct3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                norm=norm,
                activate=False,
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResidualDSBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, norm: str = "gn") -> None:
        super().__init__()
        self.block1 = DepthwiseSeparableConv3d(in_channels, out_channels, norm=norm)
        self.block2 = DepthwiseSeparableConv3d(out_channels, out_channels, norm=norm)
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = ConvNormAct3d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                norm=norm,
                activate=False,
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block2(self.block1(x)) + self.shortcut(x))


class RandomViewDepthwiseBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        active_ratio: float = 0.5,
        branch_keep_prob: float = 0.75,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        active_channels = max(1, min(channels, int(round(channels * active_ratio))))
        self.active_channels = active_channels
        self.branch_keep_prob = branch_keep_prob
        self.pre = ConvNormAct3d(
            active_channels, active_channels, kernel_size=1, padding=0, norm=norm
        )
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        active_channels,
                        active_channels,
                        kernel_size=(3, 3, 1),
                        padding=(1, 1, 0),
                        groups=active_channels,
                        bias=False,
                    ),
                    make_norm(active_channels, norm),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        active_channels,
                        active_channels,
                        kernel_size=(3, 1, 3),
                        padding=(1, 0, 1),
                        groups=active_channels,
                        bias=False,
                    ),
                    make_norm(active_channels, norm),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        active_channels,
                        active_channels,
                        kernel_size=(1, 3, 3),
                        padding=(0, 1, 1),
                        groups=active_channels,
                        bias=False,
                    ),
                    make_norm(active_channels, norm),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv3d(
                        active_channels,
                        active_channels,
                        kernel_size=3,
                        padding=2,
                        dilation=2,
                        groups=active_channels,
                        bias=False,
                    ),
                    make_norm(active_channels, norm),
                    nn.GELU(),
                ),
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv3d(active_channels, active_channels, kernel_size=1, bias=False),
            make_norm(active_channels, norm),
            nn.GELU(),
        )
        self.se = SqueezeExcite3d(active_channels)
        self.post = ConvNormAct3d(
            channels, channels, kernel_size=1, padding=0, norm=norm, activate=False
        )
        self.act = nn.GELU()

    def _mix_branches(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [branch(x) for branch in self.branches]
        if not self.training:
            return sum(outputs) / float(len(outputs))
        keep_mask = torch.rand(len(outputs), device=x.device) < self.branch_keep_prob
        if not bool(keep_mask.any().item()):
            keep_mask[torch.randint(0, len(outputs), (1,), device=x.device)] = True
        mixed = 0.0
        count = 0
        for keep, out in zip(keep_mask, outputs):
            if bool(keep.item()):
                mixed = mixed + out
                count += 1
        return mixed / float(count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        active = self.pre(x[:, : self.active_channels])
        bypass = x[:, self.active_channels :]
        active = self.se(self.mix(self._mix_branches(active)))
        if bypass.numel() > 0:
            x = torch.cat([active, bypass], dim=1)
        else:
            x = active
        return self.act(self.post(x) + residual)


class ModalityAwareAffine(nn.Module):

    def __init__(self, num_modalities: int, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_modalities, channels))
        self.bias = nn.Parameter(torch.zeros(num_modalities, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.scale.view(
            1, self.scale.size(0), self.scale.size(1), 1, 1, 1
        )
        bias = self.bias.view(1, self.bias.size(0), self.bias.size(1), 1, 1, 1)
        return x * scale + bias


class StemBlock3D(nn.Module):

    def __init__(
        self, in_channels: int, stem_channels: int, out_channels: int, norm: str = "gn"
    ) -> None:
        super().__init__()
        self.stage1 = ResidualConvBlock3D(in_channels, stem_channels, norm=norm)
        self.stage2 = ResidualConvBlock3D(
            stem_channels, out_channels, stride=2, norm=norm
        )
        self.stage2_refine = ResidualDSBlock3D(out_channels, out_channels, norm=norm)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.stage1(x)
        x2 = self.stage2_refine(self.stage2(x1))
        return (x1, x2)


class SharedEncoderStage3D(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, depth: int = 2, norm: str = "gn"
    ) -> None:
        super().__init__()
        blocks: List[nn.Module] = [
            ResidualConvBlock3D(in_channels, out_channels, stride=2, norm=norm)
        ]
        for _ in range(depth):
            blocks.append(RandomViewDepthwiseBlock(out_channels, norm=norm))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class MaskAwareModalFusion(nn.Module):

    def __init__(self, channels: int, num_modalities: int = 4) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.score = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.modality_bias = nn.Parameter(torch.zeros(1, num_modalities, 1))

    def forward(self, feats: torch.Tensor, mask: torch.Tensor):
        pooled = feats.mean(dim=(3, 4, 5))
        scores = self.score(pooled) + self.modality_bias[:, : feats.size(1)]
        scores = scores.squeeze(-1).masked_fill(mask <= 0, -10000.0)
        weights = torch.softmax(scores, dim=1)
        fused = (feats * weights[:, :, None, None, None, None]).sum(dim=1)
        return (fused, weights)


class DMSF3D(nn.Module):

    def __init__(self, c3: int, c4: int, c5: int, norm: str = "gn") -> None:
        super().__init__()
        self.x3_self = ConvNormAct3d(c3, c3, kernel_size=1, padding=0, norm=norm)
        self.x4_to_3 = ConvNormAct3d(c4, c3, kernel_size=1, padding=0, norm=norm)
        self.x5_to_3 = ConvNormAct3d(c5, c3, kernel_size=1, padding=0, norm=norm)
        self.x3_to_4 = ConvNormAct3d(c3, c4, kernel_size=1, padding=0, norm=norm)
        self.x4_self = ConvNormAct3d(c4, c4, kernel_size=1, padding=0, norm=norm)
        self.x5_to_4 = ConvNormAct3d(c5, c4, kernel_size=1, padding=0, norm=norm)
        self.x3_to_5 = ConvNormAct3d(c3, c5, kernel_size=1, padding=0, norm=norm)
        self.x4_to_5 = ConvNormAct3d(c4, c5, kernel_size=1, padding=0, norm=norm)
        self.x5_self = ConvNormAct3d(c5, c5, kernel_size=1, padding=0, norm=norm)
        self.refine3 = ResidualDSBlock3D(c3, c3, norm=norm)
        self.refine4 = ResidualDSBlock3D(c4, c4, norm=norm)
        self.refine5 = ResidualDSBlock3D(c5, c5, norm=norm)

    def forward(
        self, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y3 = self.x3_self(x3)
        y3 = y3 + resize_to(self.x4_to_3(x4), x3.shape[2:])
        y3 = y3 + resize_to(self.x5_to_3(x5), x3.shape[2:])
        y4 = self.x4_self(x4)
        y4 = y4 + resize_to(self.x3_to_4(x3), x4.shape[2:])
        y4 = y4 + resize_to(self.x5_to_4(x5), x4.shape[2:])
        y5 = self.x5_self(x5)
        y5 = y5 + resize_to(self.x3_to_5(x3), x5.shape[2:])
        y5 = y5 + resize_to(self.x4_to_5(x4), x5.shape[2:])
        return (self.refine3(y3), self.refine4(y4), self.refine5(y5))


class LowRankTokenMixer3D(nn.Module):

    def __init__(
        self, channels: int, pooled_size: int = 4, rank: int = 16, norm: str = "gn"
    ) -> None:
        super().__init__()
        num_tokens = pooled_size**3
        self.pooled_size = pooled_size
        self.pre = ConvNormAct3d(
            channels, channels, kernel_size=1, padding=0, norm=norm
        )
        self.token_down = nn.Linear(num_tokens, rank, bias=False)
        self.token_up = nn.Linear(rank, num_tokens, bias=False)
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(channels, channels * 2, kernel_size=1, bias=False),
            make_norm(channels * 2, norm),
            nn.GELU(),
            nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False),
            make_norm(channels, norm),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(self.pre(x), output_size=(self.pooled_size,) * 3)
        tokens = pooled.flatten(2)
        tokens = tokens + self.token_up(self.token_down(tokens))
        pooled = tokens.view_as(pooled)
        pooled = self.channel_mlp(pooled)
        pooled = resize_to(pooled, x.shape[2:])
        return x + self.res_scale * pooled


class BFCMicroSSM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        pooled_size: int = 4,
        rank: int = 16,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.project = ConvNormAct3d(
            in_channels, bottleneck_channels, kernel_size=1, padding=0, norm=norm
        )
        self.local_refine = ResidualDSBlock3D(
            bottleneck_channels, bottleneck_channels, norm=norm
        )
        self.token_mixer = LowRankTokenMixer3D(
            bottleneck_channels, pooled_size=pooled_size, rank=rank, norm=norm
        )
        self.out = ResidualDSBlock3D(
            bottleneck_channels, bottleneck_channels, norm=norm
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        x = self.local_refine(x)
        x = self.token_mixer(x)
        return self.out(x)


class PriorHead3D(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, norm: str = "gn") -> None:
        super().__init__()
        hidden = max(in_channels // 2, num_classes * 4)
        self.refine = ResidualDSBlock3D(in_channels, hidden, norm=norm)
        self.head = nn.Conv3d(hidden, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.refine(x))


class MultiSourceFusionGate3D(nn.Module):

    def __init__(
        self,
        guide_channels: int,
        skip_channels: int,
        out_channels: int,
        num_classes: int,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.guide_proj = ConvNormAct3d(
            guide_channels, out_channels, kernel_size=1, padding=0, norm=norm
        )
        self.skip_proj = ConvNormAct3d(
            skip_channels, out_channels, kernel_size=1, padding=0, norm=norm
        )
        self.prior_proj = ConvNormAct3d(
            num_classes, out_channels, kernel_size=1, padding=0, norm=norm
        )
        hidden = max(out_channels, 32)
        self.gate_mlp = nn.Sequential(
            nn.Linear(out_channels * 3 + 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_channels * 3),
        )
        self.refine = ResidualDSBlock3D(out_channels, out_channels, norm=norm)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))
        self.prior_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        guide: torch.Tensor,
        skip: torch.Tensor,
        prior: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        guide = self.guide_proj(guide)
        skip = self.skip_proj(skip)
        prior = self.prior_proj(resize_to(prior, skip.shape[2:]))
        descriptor = torch.cat(
            [
                guide.mean(dim=(2, 3, 4)),
                skip.mean(dim=(2, 3, 4)),
                prior.mean(dim=(2, 3, 4)),
                mask.float(),
            ],
            dim=1,
        )
        guide_gate, skip_gate, prior_gate = self.gate_mlp(descriptor).chunk(3, dim=1)
        guide_gate = torch.sigmoid(guide_gate).view(
            guide.size(0), guide.size(1), 1, 1, 1
        )
        skip_gate = torch.sigmoid(skip_gate).view(skip.size(0), skip.size(1), 1, 1, 1)
        prior_gate = torch.sigmoid(prior_gate).view(
            prior.size(0), prior.size(1), 1, 1, 1
        )
        fused = guide_gate * guide
        fused = fused + self.skip_scale * skip_gate * skip
        fused = fused + self.prior_scale * prior_gate * prior
        return self.refine(fused)


class DUCLRSUpBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_classes: int,
        norm: str = "gn",
    ) -> None:
        super().__init__()
        self.up_proj = ConvNormAct3d(
            in_channels, out_channels, kernel_size=1, padding=0, norm=norm
        )
        self.fusion = MultiSourceFusionGate3D(
            guide_channels=out_channels,
            skip_channels=skip_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            norm=norm,
        )
        self.post = ResidualDSBlock3D(out_channels, out_channels, norm=norm)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        prior: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = resize_to(x, skip.shape[2:])
        residual = self.up_proj(x)
        fused = self.fusion(residual, skip, prior, mask)
        return self.post(fused) + self.res_scale * residual


class SegHead3D(nn.Module):

    def __init__(self, in_channels: int, num_classes: int, norm: str = "gn") -> None:
        super().__init__()
        self.refine = ResidualDSBlock3D(in_channels, in_channels, norm=norm)
        self.head = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, output_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        logits = self.head(self.refine(x))
        logits = resize_to(logits, output_size)
        return torch.softmax(logits, dim=1)
