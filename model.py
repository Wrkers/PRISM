from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils

try:
    from .modules import (
        BFCMicroSSM,
        DMSF3D,
        DUCLRSUpBlock,
        MaskAwareModalFusion,
        ModalityAwareAffine,
        PriorHead3D,
        SegHead3D,
        SharedEncoderStage3D,
        StemBlock3D,
    )
except ImportError:
    from modules import (
        BFCMicroSSM,
        DMSF3D,
        DUCLRSUpBlock,
        MaskAwareModalFusion,
        ModalityAwareAffine,
        PriorHead3D,
        SegHead3D,
        SharedEncoderStage3D,
        StemBlock3D,
    )


class PRISMFuse(nn.Module):

    def __init__(
        self,
        num_classes: int = 4,
        num_modalities: int = 4,
        base_channels: int = 16,
        bottleneck_channels: int = 128,
        latent_rank: int = 16,
        pooled_size: int = 4,
        norm: str = "gn",
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 6
        c5 = base_channels * 8
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.is_training = True
        self.use_checkpoint = use_checkpoint
        stem_count = num_modalities
        self.stems = nn.ModuleList(
            [StemBlock3D(1, c1, c2, norm=norm) for _ in range(stem_count)]
        )
        self.stage3 = SharedEncoderStage3D(c2, c3, depth=1, norm=norm)
        self.stage4 = SharedEncoderStage3D(c3, c4, depth=1, norm=norm)
        self.stage5 = SharedEncoderStage3D(c4, c5, depth=2, norm=norm)
        affine_factory = ModalityAwareAffine
        self.affine3 = affine_factory(num_modalities, c3)
        self.affine4 = affine_factory(num_modalities, c4)
        self.affine5 = affine_factory(num_modalities, c5)
        self.fuse1 = MaskAwareModalFusion(c1, num_modalities=num_modalities)
        self.fuse2 = MaskAwareModalFusion(c2, num_modalities=num_modalities)
        self.fuse3 = MaskAwareModalFusion(c3, num_modalities=num_modalities)
        self.fuse4 = MaskAwareModalFusion(c4, num_modalities=num_modalities)
        self.fuse5 = MaskAwareModalFusion(c5, num_modalities=num_modalities)
        self.dmsf = DMSF3D(c3, c4, c5, norm=norm)
        self.bottleneck = BFCMicroSSM(
            c5,
            bottleneck_channels=bottleneck_channels,
            pooled_size=pooled_size,
            rank=latent_rank,
            norm=norm,
        )
        self.prior_head = PriorHead3D(
            bottleneck_channels, num_classes=num_classes, norm=norm
        )
        self.up4 = DUCLRSUpBlock(
            in_channels=bottleneck_channels,
            skip_channels=c4,
            out_channels=c4,
            num_classes=num_classes,
            norm=norm,
        )
        self.up3 = DUCLRSUpBlock(
            in_channels=c4,
            skip_channels=c3,
            out_channels=c3,
            num_classes=num_classes,
            norm=norm,
        )
        self.up2 = DUCLRSUpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            num_classes=num_classes,
            norm=norm,
        )
        self.up1 = DUCLRSUpBlock(
            in_channels=c2,
            skip_channels=c1,
            out_channels=c1,
            num_classes=num_classes,
            norm=norm,
        )
        self.aux_head4 = SegHead3D(c4, num_classes=num_classes, norm=norm)
        self.aux_head3 = SegHead3D(c3, num_classes=num_classes, norm=norm)
        self.final_head = SegHead3D(c1, num_classes=num_classes, norm=norm)

    @staticmethod
    def _stack_modalities(features) -> torch.Tensor:
        return torch.stack(features, dim=1)

    @staticmethod
    def _flatten_modalities(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, m, c, h, w, z = x.shape
        return (x.view(b * m, c, h, w, z), (b, m))

    @staticmethod
    def _unflatten_modalities(x: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        b, m = shape
        _, c, h, w, z = x.shape
        return x.view(b, m, c, h, w, z)

    def _run_shared_stage(
        self, x: torch.Tensor, stage: nn.Module, affine: nn.Module
    ) -> torch.Tensor:
        x, shape = self._flatten_modalities(x)
        x = self._checkpoint_module(stage, x)
        x = self._unflatten_modalities(x, shape)
        return affine(x)

    def _checkpoint_module(self, module: nn.Module, *args):
        if self.use_checkpoint and self.training:
            return checkpoint_utils.checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        mask = mask.float()
        spatial_size = x.shape[2:]
        stage1_feats = []
        stage2_feats = []
        for modal_idx in range(self.num_modalities):
            stem = self.stems[modal_idx]
            modal_input = x[:, modal_idx : modal_idx + 1]
            modal_input = (
                modal_input * mask[:, modal_idx : modal_idx + 1, None, None, None]
            )
            x1, x2 = self._checkpoint_module(stem, modal_input)
            stage1_feats.append(x1)
            stage2_feats.append(x2)
        x1 = self._stack_modalities(stage1_feats)
        x2 = self._stack_modalities(stage2_feats)
        x3 = self._run_shared_stage(x2, self.stage3, self.affine3)
        x4 = self._run_shared_stage(x3, self.stage4, self.affine4)
        x5 = self._run_shared_stage(x4, self.stage5, self.affine5)
        x1_fused, w1 = self.fuse1(x1, mask)
        x2_fused, w2 = self.fuse2(x2, mask)
        x3_fused, w3 = self.fuse3(x3, mask)
        x4_fused, w4 = self.fuse4(x4, mask)
        x5_fused, w5 = self.fuse5(x5, mask)
        x3_ctx, x4_ctx, x5_ctx = self.dmsf(x3_fused, x4_fused, x5_fused)
        bottleneck = self._checkpoint_module(self.bottleneck, x5_ctx)
        prior_logits = self.prior_head(bottleneck)
        prior_low = F.softmax(prior_logits, dim=1)
        prior_pred = F.softmax(
            F.interpolate(
                prior_logits, size=spatial_size, mode="trilinear", align_corners=False
            ),
            dim=1,
        )
        if self.use_checkpoint and self.training:
            d4 = checkpoint_utils.checkpoint(
                lambda a, b, c, d: self.up4(a, b, c, d),
                bottleneck,
                x4_ctx,
                prior_low,
                mask,
                use_reentrant=False,
            )
            d3 = checkpoint_utils.checkpoint(
                lambda a, b, c, d: self.up3(a, b, c, d),
                d4,
                x3_ctx,
                prior_low,
                mask,
                use_reentrant=False,
            )
            d2 = checkpoint_utils.checkpoint(
                lambda a, b, c, d: self.up2(a, b, c, d),
                d3,
                x2_fused,
                prior_low,
                mask,
                use_reentrant=False,
            )
            d1 = checkpoint_utils.checkpoint(
                lambda a, b, c, d: self.up1(a, b, c, d),
                d2,
                x1_fused,
                prior_low,
                mask,
                use_reentrant=False,
            )
        else:
            d4 = self.up4(bottleneck, x4_ctx, prior_low, mask)
            d3 = self.up3(d4, x3_ctx, prior_low, mask)
            d2 = self.up2(d3, x2_fused, prior_low, mask)
            d1 = self.up1(d2, x1_fused, prior_low, mask)
        aux4 = self.aux_head4(d4, spatial_size)
        aux3 = self.aux_head3(d3, spatial_size)
        final_pred = self.final_head(d1, spatial_size)
        if not self.is_training:
            return final_pred
        return {
            "final_pred": final_pred,
            "prior_pred": prior_pred,
            "aux_preds": [aux4, aux3],
            "modal_weights": {"x1": w1, "x2": w2, "x3": w3, "x4": w4, "x5": w5},
        }
