import torch
import torch.nn.functional as F

def iou(gt_masks, pred_masks):
    pred_masks_thresh = (pred_masks > 0.5).type(torch.int)
    gt_masks = (gt_masks > 0.5).type(torch.int)
    return torch.mean((pred_masks_thresh & gt_masks).type(torch.float))\
           / torch.mean((pred_masks_thresh | gt_masks).type(torch.float))

def visualize(images, masks):
    scale = int(images.shape[2] / masks.shape[2])
    masks_thresh = (F.upsample(masks, scale_factor=scale, mode='bilinear') > 0.5).type(torch.int)
    segmented_images = images.clone()
    segmented_images[masks_thresh.repeat(1,3,1,1) == 0] = 0
    return segmented_images
