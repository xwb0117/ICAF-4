import torch.nn.functional as F
import torch

def compute_miou_loss(pred_seg_per_point, gt_seg_onehot):
    dot = torch.sum(pred_seg_per_point * gt_seg_onehot, axis=1)
    denominator = torch.sum(pred_seg_per_point, axis=1) + torch.sum(gt_seg_onehot, axis=1) - dot
    mIoU = dot / (denominator + 1e-10)
    return torch.mean(1.0 - mIoU)

def compute_coorindate_loss(pred_coordinate_per_point, gt_coordinate_per_point, num_parts, gt_seg_onehot):
    loss_coordinate = 0.0
    coordinate_splits = torch.split(pred_coordinate_per_point, split_size_or_sections=3, dim=2)
    mask_splits = torch.split(gt_seg_onehot, split_size_or_sections=1, dim=2)
    for i in range(num_parts):
        diff_l2 = torch.norm(coordinate_splits[i] - gt_coordinate_per_point, dim=2)
        loss_coordinate += torch.mean(mask_splits[i][:, :, 0] * diff_l2, axis = 1)
    return torch.mean(loss_coordinate, axis=0)

def compute_reg_loss(pred_reg, gt_reg):
    gt_reg = gt_reg.view(pred_reg.shape[0], 2048, -1)
    loss_reg = F.mse_loss(pred_reg, gt_reg, reduction='mean')
    return  loss_reg

def compute_align_loss(pred, gt, grasp_cls, align_mask):
    pred = pred.view(gt.shape)
    diff_l2 = torch.norm(pred - gt, dim=3)
    loss_align_mask = diff_l2[grasp_cls.unsqueeze(-1).expand_as(diff_l2).bool()]
    loss_align_mask = torch.mean(loss_align_mask, dim=0)
    loss = torch.mean(loss_align_mask)
    return loss

def compute_vect_loss(pred_vect_per_point, gt_vect_per_point, mask):
    if pred_vect_per_point.shape[2] == 1:
        pred_vect_per_point = torch.squeeze(pred_vect_per_point, dim=2)
        diff_l2 = torch.abs(pred_vect_per_point - gt_vect_per_point) * mask
    else:
        diff_l2 = torch.norm(pred_vect_per_point - gt_vect_per_point, dim=2) * mask

    return torch.mean(torch.mean(diff_l2, axis=1), axis=0)