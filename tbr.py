
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ttn import TemporalTransformNetwork
from utils import (batch_iou,
                   bbox_se_transform_batch, bbox_xw_transform_batch,
                   bbox_se_transform_inv, bbox_xw_transform_inv)

class StartEndRegression(nn.Module):
    def __init__(self, start_sample_num, end_sample_num, feat_dim):
        super(StartEndRegression, self).__init__()
        self.start_sample_num = start_sample_num
        self.end_sample_num = end_sample_num
        self.temporal_len = self.start_sample_num + self.end_sample_num
        self.feat_dim = feat_dim
        self.prop_boundary_ratio = 0.5
        self.hidden_dim_1d = 128

        self.start_reg_conv = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=self.start_sample_num // 4),
        )
        self.end_reg_conv = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 1, kernel_size=self.end_sample_num // 4),
        )

    def forward(self, starts_feature, ends_feature):
        start_reg = self.start_reg_conv(starts_feature)
        end_reg = self.end_reg_conv(ends_feature)
        se_reg = torch.cat([start_reg, end_reg], dim=1).squeeze(2)
        return se_reg


class CenterWidthRegression(nn.Module):
    def __init__(self, start_sample_num, end_sample_num, action_sample_num, feat_dim):
        super(CenterWidthRegression, self).__init__()
        self.temporal_len = action_sample_num + start_sample_num + end_sample_num
        self.feat_dim = feat_dim
        self.hidden_dim_1d = 512
        self.reg_1d_b = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4, stride=2),
            nn.ReLU(inplace=True)
        )
        self.reg_1d_o = nn.Sequential(
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=self.temporal_len // 4, padding=0, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, 3, kernel_size=1),
        )

    def forward(self, x):
        rbf = self.reg_1d_b(x)
        regression = self.reg_1d_o(rbf)
        return regression


class TemporalBoundaryRegressor(nn.Module):
    def __init__(self, opt):
        super(TemporalBoundaryRegressor, self).__init__()
        start_sample_num = opt['start_sample_num']
        end_sample_num = opt['end_sample_num']
        action_sample_num = opt['action_sample_num']
        prop_boundary_ratio = opt['prop_boundary_ratio']
        temporal_interval = opt['temporal_interval']
        self.hidden_dim_1d = 512
        self.reg1se = StartEndRegression(start_sample_num, end_sample_num, self.hidden_dim_1d)
        self.reg1xw = CenterWidthRegression(start_sample_num, end_sample_num, action_sample_num, self.hidden_dim_1d)
        self.ttn = TemporalTransformNetwork(prop_boundary_ratio,
                                            action_sample_num,
                                            start_sample_num,
                                            end_sample_num,
                                            temporal_interval, norm_mode='padding')

    def forward(self, proposals, features, video_sec, gt_boxes, iou_thres, training):
        proposals1 = proposals[:, 0:2]
        starts_feature1, actions_feature1, ends_feature1 = self.ttn(proposals1, features, video_sec)
        reg1se = self.reg1se(starts_feature1, ends_feature1)
        features1xw = torch.cat([starts_feature1, actions_feature1, ends_feature1], dim=2)
        reg1xw = self.reg1xw(features1xw).squeeze(2)
        preds_iou1 = reg1xw[:, 2].sigmoid()
        reg1xw = reg1xw[:, :2]

        if training:
            proposals2xw = bbox_xw_transform_inv(proposals1, reg1xw, 0.1, 0.2)
            proposals2se = bbox_se_transform_inv(proposals1, reg1se, 1.0)

            iou1 = batch_iou(proposals1, gt_boxes)
            targets1se = bbox_se_transform_batch(proposals1, gt_boxes)
            targets1xw = bbox_xw_transform_batch(proposals1, gt_boxes)
            rloss1se = self.regression_loss(reg1se, targets1se, iou1, iou_thres)
            rloss1xw = self.regression_loss(reg1xw, targets1xw, iou1, iou_thres)
            rloss1 = rloss1se + rloss1xw
            iloss1 = self.iou_loss(preds_iou1, iou1, iou_thres=iou_thres)
        else:
            proposals2xw = bbox_xw_transform_inv(proposals1, reg1xw, 0.1, 0.2)
            proposals2se = bbox_se_transform_inv(proposals1, reg1se, 0.2)
            rloss1 = 0
            iloss1 = 0
        proposals2 = (proposals2se + proposals2xw) / 2.0
        proposals2 = torch.clamp(proposals2, min=0.)
        return preds_iou1, proposals2, rloss1, iloss1

    def regression_loss(self, regression, targets, iou_with_gt, iou_thres):
        weight = (iou_with_gt >= iou_thres).float().unsqueeze(1)
        reg_loss = F.smooth_l1_loss(regression, targets, reduction='none')
        if torch.sum(weight) > 0:
            reg_loss = torch.sum(weight * reg_loss) / torch.sum(weight)
        else:
            reg_loss = torch.sum(weight * reg_loss)
        return reg_loss

    def iou_loss(self, preds_iou, match_iou, iou_thres):
        preds_iou = preds_iou.view(-1)
        u_hmask = (match_iou > iou_thres).float()
        u_mmask = ((match_iou <= iou_thres) & (match_iou > 0.3)).float()
        u_lmask = (match_iou <= 0.3).float()

        num_h = torch.sum(u_hmask)
        num_m = torch.sum(u_mmask)
        num_l = torch.sum(u_lmask)

        r_m = num_h / (num_m)
        r_m = torch.min(r_m, torch.Tensor([1.0]).cuda())[0]
        u_smmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
        u_smmask = u_smmask * u_mmask
        u_smmask = (u_smmask > (1. - r_m)).float()

        r_l = num_h / (num_l)
        r_l = torch.min(r_l, torch.Tensor([1.0]).cuda())[0]
        u_slmask = torch.Tensor(np.random.rand(u_hmask.size()[0])).cuda()
        u_slmask = u_slmask * u_lmask
        u_slmask = (u_slmask > (1. - r_l)).float()

        iou_weights = u_hmask + u_smmask + u_slmask
        iou_loss = F.smooth_l1_loss(preds_iou, match_iou, reduction='none')
        if torch.sum(iou_weights) > 0:
            iou_loss = torch.sum(iou_loss * iou_weights) / torch.sum(iou_weights)
        else:
            iou_loss = torch.sum(iou_loss * iou_weights)
        return iou_loss
