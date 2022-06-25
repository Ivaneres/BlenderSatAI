import torch
from torch import nn

from lidarNet.utils.ml_utils import crop_masks


def dice_loss(inputs, targets, eps=1):
    assert inputs.shape == targets.shape
    inputs = inputs.reshape(-1)
    targets = targets.reshape(-1)
    intersection = (inputs * targets).sum()
    dice = (2 * intersection + eps) / (inputs.sum() + targets.sum() + eps)
    return 1 - dice


def val_loss(mdl, val_loader, device, criterion):
    running_loss = 0.0
    mdl.eval()
    with torch.no_grad():
        for i, d in enumerate(val_loader):
            inputs, masks = d
            inputs = inputs.to(device)
            masks = masks.to(device).squeeze(dim=1)

            outputs = mdl(inputs).squeeze(dim=1)
            masks = crop_masks(masks, outputs.shape[1:3])
            loss = criterion(outputs, masks)

            running_loss += loss.item()
    return running_loss


def l1_loss_custom(inputs, targets):
    targets[targets < 0] = inputs[targets < 0]
    return nn.L1Loss()(inputs, targets)


def rmse_loss_custom(inputs, targets):
    targets[targets < 0] = inputs[targets < 0]
    return torch.sqrt(nn.MSELoss()(inputs, targets))
