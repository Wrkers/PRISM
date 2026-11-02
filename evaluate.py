import os
from typing import Optional, Sequence

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

try:
    from .engine import core_model
    from .losses import dice_loss, softmax_weighted_loss
except ImportError:
    from engine import core_model
    from losses import dice_loss, softmax_weighted_loss


PATCH_SIZE = 128


def softmax_output_dice_class4(output: torch.Tensor, target: torch.Tensor):
    eps = 1e-8
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps
    denominator1 = torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps
    denominator2 = torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps
    denominator3 = torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice = intersect3 / denominator3

    o4 = o3 * 0.0 if torch.sum(o3) < 500 else o3
    intersect4 = torch.sum(2 * (o4 * t3), dim=(1, 2, 3)) + eps
    denominator4 = torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    enhancing_dice_post = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    whole_inter = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    whole_den = (
        torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    )
    dice_whole = whole_inter / whole_den

    o_core = o1 + o3
    t_core = t1 + t3
    core_inter = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    core_den = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = core_inter / core_den

    separate = torch.stack([ncr_net_dice, edema_dice, enhancing_dice], dim=1)
    evaluation = torch.stack(
        [dice_whole, dice_core, enhancing_dice, enhancing_dice_post], dim=1
    )
    return separate.cpu().numpy(), evaluation.cpu().numpy()


def softmax_output_dice_class5(output: torch.Tensor, target: torch.Tensor):
    eps = 1e-8
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    o4 = (output == 4).float()
    t4 = (target == 4).float()

    necrosis = (torch.sum(2 * (o1 * t1), dim=(1, 2, 3)) + eps) / (
        torch.sum(o1, dim=(1, 2, 3)) + torch.sum(t1, dim=(1, 2, 3)) + eps
    )
    edema = (torch.sum(2 * (o2 * t2), dim=(1, 2, 3)) + eps) / (
        torch.sum(o2, dim=(1, 2, 3)) + torch.sum(t2, dim=(1, 2, 3)) + eps
    )
    non_enhancing = (torch.sum(2 * (o3 * t3), dim=(1, 2, 3)) + eps) / (
        torch.sum(o3, dim=(1, 2, 3)) + torch.sum(t3, dim=(1, 2, 3)) + eps
    )
    enhancing = (torch.sum(2 * (o4 * t4), dim=(1, 2, 3)) + eps) / (
        torch.sum(o4, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    )
    o5 = o4 * 0.0 if torch.sum(o4) < 500 else o4
    enhancing_post = (torch.sum(2 * (o5 * t4), dim=(1, 2, 3)) + eps) / (
        torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t4, dim=(1, 2, 3)) + eps
    )

    whole = (
        torch.sum(2 * ((o1 + o2 + o3 + o4) * (t1 + t2 + t3 + t4)), dim=(1, 2, 3)) + eps
    ) / (
        torch.sum(o1 + o2 + o3 + o4, dim=(1, 2, 3))
        + torch.sum(t1 + t2 + t3 + t4, dim=(1, 2, 3))
        + eps
    )
    core = (torch.sum(2 * ((o1 + o3 + o4) * (t1 + t3 + t4)), dim=(1, 2, 3)) + eps) / (
        torch.sum(o1 + o3 + o4, dim=(1, 2, 3))
        + torch.sum(t1 + t3 + t4, dim=(1, 2, 3))
        + eps
    )

    separate = torch.stack([necrosis, edema, non_enhancing, enhancing], dim=1)
    evaluation = torch.stack([whole, core, enhancing, enhancing_post], dim=1)
    return separate.cpu().numpy(), evaluation.cpu().numpy()


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_softmax(
    test_loader,
    model,
    dataname: str = "BRATS2023",
    feature_mask: Optional[Sequence[bool]] = None,
    compute_loss: bool = True,
    save_masks: bool = False,
    save_dir: Optional[str] = None,
    patch_size: int = PATCH_SIZE,
):
    model_core = core_model(model)
    model_core.is_training = False
    model.eval()

    if dataname in ["BRATS2023", "BRATS2021", "BRATS2020", "BRATS2019", "BRATS2018"]:
        num_classes = 4
    elif dataname == "BRATS2015":
        num_classes = 5
    else:
        raise ValueError(f"Unsupported dataset name: {dataname}")

    vals_eval = AverageMeter()
    loss_meter = 0.0
    device = next(model.parameters()).device
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size, device=device)

    for data in test_loader:
        x = data[0].to(device, non_blocking=True)
        target = data[1].to(device, non_blocking=True)
        yo = data[3].to(device, non_blocking=True)
        names = data[-1]

        if feature_mask is not None:
            mask = (
                torch.tensor(np.array(feature_mask), device=device)
                .unsqueeze(0)
                .repeat(len(names), 1)
            )
        else:
            mask = data[2].to(device, non_blocking=True)

        _, _, h, w, z = x.shape
        stride = max(int(patch_size * 0.5), 1)
        h_idx = list(range(0, max(h - patch_size, 0) + 1, stride))
        w_idx = list(range(0, max(w - patch_size, 0) + 1, stride))
        z_idx = list(range(0, max(z - patch_size, 0) + 1, stride))
        if not h_idx or h_idx[-1] != h - patch_size:
            h_idx.append(h - patch_size)
        if not w_idx or w_idx[-1] != w - patch_size:
            w_idx.append(w - patch_size)
        if not z_idx or z_idx[-1] != z - patch_size:
            z_idx.append(z - patch_size)

        weight = torch.zeros(len(names), num_classes, h, w, z, device=device)
        pred = torch.zeros(len(names), num_classes, h, w, z, device=device)

        for hs in h_idx:
            for ws in w_idx:
                for zs in z_idx:
                    weight[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        zs : zs + patch_size,
                    ] += one_tensor
                    patch = x[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        zs : zs + patch_size,
                    ]
                    pred_patch = model(patch, mask)
                    pred[
                        :,
                        :,
                        hs : hs + patch_size,
                        ws : ws + patch_size,
                        zs : zs + patch_size,
                    ] += pred_patch

        pred = pred / weight.clamp_min(1.0)

        if compute_loss:
            seg_cross = softmax_weighted_loss(pred, yo, num_classes=num_classes)
            seg_dice = dice_loss(pred, yo, num_classes=num_classes)
            loss_meter += (seg_cross + seg_dice).item()

        hard_pred = torch.argmax(pred, dim=1)
        if num_classes == 4:
            _, scores_eval = softmax_output_dice_class4(hard_pred, target)
        else:
            _, scores_eval = softmax_output_dice_class5(hard_pred, target)

        for case_idx, case_name in enumerate(names):
            vals_eval.update(scores_eval[case_idx])
            if save_masks and save_dir is not None:
                if nib is None:
                    raise ImportError("nibabel is required when save_masks=True.")
                os.makedirs(save_dir, exist_ok=True)
                case_mask = mask[case_idx].bool().detach().cpu().numpy().tolist()
                flag = "".join("1" if item else "0" for item in case_mask)
                out_name = f"{case_name}_{flag}.nii.gz"
                out_path = os.path.join(save_dir, out_name)
                affine = np.diag([1.0, 1.0, 1.0, 1.0])
                nib.save(
                    nib.Nifti1Image(
                        hard_pred[case_idx].detach().cpu().numpy().astype(np.uint8),
                        affine,
                    ),
                    out_path,
                )

    if compute_loss:
        return vals_eval.avg, loss_meter / max(len(test_loader), 1)
    return vals_eval.avg
