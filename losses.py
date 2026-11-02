from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


def dice_loss(
    output: torch.Tensor, target: torch.Tensor, num_classes: int = 4, eps: float = 1e-07
) -> torch.Tensor:
    target = target.float()
    dice = 0.0
    for class_idx in range(num_classes):
        pred_class = output[:, class_idx]
        target_class = target[:, class_idx]
        inter = torch.sum(pred_class * target_class)
        pred_sum = torch.sum(pred_class)
        target_sum = torch.sum(target_class)
        dice = dice + 2.0 * inter / (pred_sum + target_sum + eps)
    return 1.0 - dice / float(num_classes)


def softmax_weighted_loss(
    output: torch.Tensor, target: torch.Tensor, num_classes: int = 4
) -> torch.Tensor:
    target = target.float()
    _, _, h, w, z = output.size()
    loss = 0.0
    for class_idx in range(num_classes):
        pred_class = output[:, class_idx]
        target_class = target[:, class_idx]
        weight = 1.0 - torch.sum(target_class, dim=(1, 2, 3)) / torch.sum(
            target, dim=(1, 2, 3, 4)
        ).clamp_min(1.0)
        weight = weight.view(-1, 1, 1, 1).repeat(1, h, w, z)
        loss = (
            loss
            + (
                -weight
                * target_class
                * torch.log(torch.clamp(pred_class, min=1e-07, max=1.0))
            ).mean()
        )
    return loss


def segmentation_loss(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cross = softmax_weighted_loss(pred, target, num_classes=num_classes)
    dice = dice_loss(pred, target, num_classes=num_classes)
    return (cross + dice, cross, dice)


def compute_prism_loss(
    outputs: Dict[str, torch.Tensor],
    target: torch.Tensor,
    num_classes: int,
    aux_weight: float = 0.3,
    prior_weight: float = 0.4,
    consistency_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    final_loss, final_cross, final_dice = segmentation_loss(
        outputs["final_pred"], target, num_classes=num_classes
    )
    prior_loss, prior_cross, prior_dice = segmentation_loss(
        outputs["prior_pred"], target, num_classes=num_classes
    )
    aux_losses: List[torch.Tensor] = []
    aux_cross_losses: List[torch.Tensor] = []
    aux_dice_losses: List[torch.Tensor] = []
    for aux_pred in outputs.get("aux_preds", []):
        aux_seg_loss, aux_cross, aux_dice = segmentation_loss(
            aux_pred, target, num_classes=num_classes
        )
        aux_losses.append(aux_seg_loss)
        aux_cross_losses.append(aux_cross)
        aux_dice_losses.append(aux_dice)
    if aux_losses:
        aux_loss = sum(aux_losses) / float(len(aux_losses))
        aux_cross = sum(aux_cross_losses) / float(len(aux_cross_losses))
        aux_dice = sum(aux_dice_losses) / float(len(aux_dice_losses))
    else:
        zero = final_loss.new_zeros(())
        aux_loss = zero
        aux_cross = zero
        aux_dice = zero
    consistency = F.mse_loss(outputs["prior_pred"], outputs["final_pred"])
    total = (
        final_loss
        + prior_weight * prior_loss
        + aux_weight * aux_loss
        + consistency_weight * consistency
    )
    stats = {
        "loss": total.detach(),
        "final_loss": final_loss.detach(),
        "final_cross": final_cross.detach(),
        "final_dice": final_dice.detach(),
        "prior_loss": prior_loss.detach(),
        "prior_cross": prior_cross.detach(),
        "prior_dice": prior_dice.detach(),
        "aux_loss": aux_loss.detach(),
        "aux_cross": aux_cross.detach(),
        "aux_dice": aux_dice.detach(),
        "consistency": consistency.detach(),
    }
    return (total, stats)
