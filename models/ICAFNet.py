import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import PointNet2
from models import loss

class get_model(nn.Module):
    def __init__(self, num_parts, L):
        super().__init__()

        # Define the shared PN++
        self.backbone = PointNet2()

        # segmentation branch
        self.seg_layer = nn.Conv1d(128, num_parts, kernel_size=1, padding=0)
        # grasp seg branch
        self.grasp_seg_layer = nn.Conv1d(128, 2, kernel_size=1, padding=0)
        # NPCS branch
        self.npcs_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0),
            nn.Conv1d(128, 3 * num_parts, kernel_size=1, padding=0),
        )
        # NAOCS branch
        self.naocs_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0),
            nn.Conv1d(128, 3 * num_parts, kernel_size=1, padding=0),
        )
        # Anchor Candidates branch
        self.candidates_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0),
            nn.Conv1d(128, 4 * L, kernel_size=1, padding=0),
        )
        # D_quats branch
        self.D_quats_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0),
            nn.Conv1d(128, 4 * L, kernel_size=1, padding=0),
        )
        # anchor_seg branch
        self.anchor_seg_layer = nn.Conv1d(128, L, kernel_size=1, padding=0)
        # Joint parameters
        self.joint_feature_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        # Joint UNitVec, heatmap, joint_cls
        self.axis_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
        self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
        self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)
        self.joint_cls_layer = nn.Conv1d(
            128, num_parts, kernel_size=1, padding=0
        )

    def forward(self, input):
        features = self.backbone(input)
        pred_seg_per_point = self.seg_layer(features).transpose(1, 2)
        pred_npcs_per_point = self.npcs_layer(features).transpose(1, 2)
        pred_naocs_per_point = self.naocs_layer(features).transpose(1, 2)

        pred_grasp_seg_per_point = self.grasp_seg_layer(features).transpose(1,2)
        pred_Anchor_Candidates_per_point = self.candidates_layer(features).transpose(1,2)
        pred_D_quats_per_point = self.D_quats_layer(features).transpose(1,2)
        pred_anchor_seg_per_point = self.anchor_seg_layer(features).transpose(1,2)

        joint_features = self.joint_feature_layer(features)
        pred_axis_per_point = self.axis_layer(joint_features).transpose(1, 2)
        pred_unitvec_per_point = self.unitvec_layer(joint_features).transpose(1, 2)
        pred_heatmap_per_point = self.heatmap_layer(joint_features).transpose(1, 2)
        pred_joint_cls_per_point = self.joint_cls_layer(joint_features).transpose(1, 2)

        # Process the predicted things
        pred_seg_per_point = F.softmax(pred_seg_per_point, dim=2)
        pred_npcs_per_point = F.sigmoid(pred_npcs_per_point)
        pred_grasp_seg_per_point = F.softmax(pred_grasp_seg_per_point, dim=2)
        pred_Anchor_Candidates_per_point = F.tanh(pred_Anchor_Candidates_per_point)
        pred_D_quats_per_point = F.tanh(pred_D_quats_per_point)
        pred_anchor_seg_per_point = F.softmax(pred_anchor_seg_per_point)
        pred_heatmap_per_point = F.sigmoid(pred_heatmap_per_point)
        pred_unitvec_per_point = F.tanh(pred_unitvec_per_point)
        pred_axis_per_point = F.tanh(pred_axis_per_point)
        pred_joint_cls_per_point = F.softmax(pred_joint_cls_per_point, dim=2)

        pred = {
            "seg_per_point": pred_seg_per_point,
            "npcs_per_point": pred_npcs_per_point,
            "naocs_per_point": pred_naocs_per_point,
            "grasp_seg_per_point": pred_grasp_seg_per_point,
            "anchor_Candidates_per_point": pred_Anchor_Candidates_per_point,
            "D_quats_per_point": pred_D_quats_per_point,
            "anchor_seg_per_point": pred_anchor_seg_per_point,
            "heatmap_per_point": pred_heatmap_per_point,
            "unitvec_per_point": pred_unitvec_per_point,
            "axis_per_point": pred_axis_per_point,
            "joint_cls_per_point": pred_joint_cls_per_point
            }

        return pred

    def losses(self, pred, gt, L):
        # The returned loss is a value
        num_parts = pred["seg_per_point"].shape[2]
        # Convert the gt['seg_per_point'] into gt_seg_onehot B*N*K
        gt_seg_onehot = F.one_hot(gt["seg_per_point"].long(), num_classes=num_parts)

        # Convert the gt['grasp_seg_per_point'] into gt_grasp_seg_onehot B*N*K
        gt_grasp_seg_onehot = F.one_hot(gt["grasp_cls_per_point"].long(), num_classes=2)

        # pred['seg_per_point']: B*N*K, gt_seg_onehot: B*N*K
        seg_loss = loss.compute_miou_loss(pred["seg_per_point"], gt_seg_onehot)

        # pred['grasp_seg_per_point']: B*N*K, gt_grasp_seg_onehot: B*N*K
        grasp_seg_loss = loss.compute_miou_loss(pred["grasp_seg_per_point"], gt_grasp_seg_onehot)

        gt_grasp_seg_per_point = gt["grasp_cls_per_point"]
        gt_anchor_seg_per_point = gt["anchor_seg_per_point"]
        pred_anchor_seg_per_point = pred["anchor_seg_per_point"]
        segmented_pred_anchor_seg_per_point = pred_anchor_seg_per_point[gt_grasp_seg_per_point.unsqueeze(-1).expand_as(pred_anchor_seg_per_point).bool()].reshape(-1, L)
        segmented_gt_anchor_seg_per_point = gt_anchor_seg_per_point[gt_grasp_seg_per_point == 1]
        # pred_anchor_seg_per_point = pred["Alignment_mask"].contiguous().view(-1, L)
        # gt_anchor_seg_per_point = gt["anchor_seg_per_point"].view(-1, 1)[:, 0].long()
        anchor_seg_loss = F.cross_entropy(segmented_pred_anchor_seg_per_point, segmented_gt_anchor_seg_per_point)

        # pred['pred_Anchor_Candidates']: B*N*L*4, gt["Anchor_Candidates"]: B*N*L*4
        anchor_Candidates_loss = loss.compute_reg_loss(pred["anchor_Candidates_per_point"], gt["anchor_candidates_per_point"])

        # pred['pred_Alignment']: B*N*L*4, gt['pred_Alignment']: B*N*L*4
        D_quats_loss = loss.compute_align_loss(pred['D_quats_per_point'], gt['D_quats_per_point'], gt["grasp_cls_per_point"], gt['anchor_seg_per_point'])

        # pred['npcs_per_point']: B*N*3K, gt['npcs_per_point']: B*N*3, gt_seg_onehot: B*N*K
        npcs_loss = loss.compute_coorindate_loss(
            pred["npcs_per_point"],
            gt["npcs_per_point"],
            num_parts=num_parts,
            gt_seg_onehot=gt_seg_onehot,
        )

        # pred['naocs_per_point']: B*N*3K, gt['naocs_per_point']: B*N*3, gt_seg_onehot: B*N*K
        naocs_loss = loss.compute_coorindate_loss(
            pred["naocs_per_point"],
            gt["naocs_per_point"],
            num_parts=num_parts,
            gt_seg_onehot=gt_seg_onehot,
        )

        # Get the useful joint mask, gt['joint_cls_per_point'] == 0 means that
        # the point doesn't have a corresponding joint
        # B*N
        gt_joint_mask = (gt["joint_cls_per_point"] > 0).float()
        # Get the heatmap and unitvec map, the loss should only be calculated for revolute joint
        gt_revolute_mask = torch.zeros_like(gt["joint_cls_per_point"]) == 1
        revolute_index = torch.where(gt["joint_type"][0] == 1)[0]
        assert (gt["joint_type"][:, 0] == -1).all() == True
        for i in revolute_index:
            gt_revolute_mask = torch.logical_or(gt_revolute_mask, (gt["joint_cls_per_point"] == i))
        gt_revolute_mask = gt_revolute_mask.float()
        # pred['heatmap_per_point']: B*N*1, gt['heatmap_per_point']: B*N, gt_revolute_mask: B*N

        heatmap_loss = loss.compute_vect_loss(
            pred["heatmap_per_point"], gt["heatmap_per_point"], mask=gt_revolute_mask
        )
        # pred['unitvec_per_point']: B*N*3, gt['unitvec_per_point']: B*N*3, gt_revolute_mask: B*N
        unitvec_loss = loss.compute_vect_loss(
            pred["unitvec_per_point"], gt["unitvec_per_point"], mask=gt_revolute_mask
        )
        # pred['axis_per_point]: B*N*3, gt['axis_per_point']: B*N*3, gt_joint_mask: B*N
        axis_loss = loss.compute_vect_loss(
            pred["axis_per_point"], gt["axis_per_point"], mask=gt_joint_mask
        )

        # Conver the gt['joint_cls_per_point'] into gt_joint_cls_onehot B*N*K
        gt_joint_cls_onehot = F.one_hot(
            gt["joint_cls_per_point"].long(), num_classes=num_parts
        )
        joint_loss = loss.compute_miou_loss(
            pred["joint_cls_per_point"], gt_joint_cls_onehot
        )

        loss_dict = {
            "seg_loss": seg_loss,
            "npcs_loss": npcs_loss,
            "naocs_loss": naocs_loss,
            "heatmap_loss": heatmap_loss,
            "unitvec_loss": unitvec_loss,
            "axis_loss": axis_loss,
            "joint_loss": joint_loss,
            "grasp_seg_loss": grasp_seg_loss,
            "D_quats_loss": D_quats_loss,
            "anchor_seg_loss": anchor_seg_loss,
            "anchor_candidates_loss": anchor_Candidates_loss
            }

        return loss_dict

    def loss_weight(self):
        loss_weight = {
            "seg_loss": 1,
            "npcs_loss": 10,
            "naocs_loss": 10,
            "heatmap_loss": 1,
            "unitvec_loss": 1,
            "axis_loss": 1,
            "joint_loss": 1,
            "grasp_seg_loss": 1,
            "D_quats_loss": 10,
            "anchor_seg_loss": 1,
            "anchor_candidates_loss": 10
                       }
        return loss_weight

class AvgRecorder(object):
    """
    Average and current value recorder
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

