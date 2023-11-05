import torch
from torch import Tensor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, auc

def threshold_prob(pred_prob, threshold=0.5):
    pred_prob_cloned = torch.clone(pred_prob)
    mask1 = pred_prob_cloned > threshold
    mask2 = torch.isclose(pred_prob_cloned, torch.FloatTensor([threshold]).cuda(), rtol=1e-05, atol=1e-04, equal_nan=False)
    
    mask = torch.logical_or(mask1, mask2)
    pred_prob_cloned[mask] = 1
    pred_prob_cloned[torch.logical_not(mask)] = 0
    return pred_prob_cloned

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def iou_coef(input: Tensor, target: Tensor, mask = None, smooth=1e-6, per_instance=False):
    mask = mask if mask is not None else torch.ones_like(input)
    dim_to_sum = tuple(range(1,len(input.shape)))
    intersection = torch.sum(torch.abs(input * target * mask), dim=dim_to_sum)
    union = torch.sum(target * mask, dim=dim_to_sum)+torch.sum(input * mask,dim=dim_to_sum)-intersection
    if per_instance:
        return (intersection / (union + smooth))
    iou = torch.mean(intersection / (union + smooth), axis=0)
    return iou

def dice_coef(input: Tensor, target: Tensor, mask = None, smooth=1e-6, per_instance=False):
    mask = mask if mask is not None else torch.ones_like(input)
    intersection = torch.sum(input * target * mask, dim=(1,2,3))
    union = torch.sum(target * mask,dim=(1,2,3)) + torch.sum(input * mask,dim=(1,2,3))
    if per_instance:
        return ((2. * intersection + smooth)/(union + smooth))
    dice = torch.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice

def dice_loss(input: Tensor, target: Tensor, mask : Tensor = None, per_instance: bool =False, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # assert input.size() == target.size()
    # fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - dice_coef(input, target, mask=mask, per_instance=per_instance)

def map_score(input, target):
    return average_precision_score(target.reshape(-1,), input.reshape(-1,))

def precision_recall(input, target):
    #calculate precision and recall
    return precision_recall_curve(target.reshape(-1,), input.reshape(-1,))

def get_roc_auc_score(input, target):
    return roc_auc_score(target, input)

def get_auc(recall, precision):
    return auc(recall, precision)