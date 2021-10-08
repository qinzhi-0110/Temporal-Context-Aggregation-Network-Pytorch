
import torch
import torch.nn as nn
from lgte import LocalGlobalTemporalEncoder
from tbr import TemporalBoundaryRegressor


class TCANet(nn.Module):
    def __init__(self, mode, opt):
        super(TCANet, self).__init__()
        self.input_dim = opt['feat_dim']
        self.mode = mode
        self.lgte_num = opt['lgte_num']
        self.hidden_dim_1d = 512
        self.x_1d_b_f = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
        )
        self.tbr1 = TemporalBoundaryRegressor(opt)
        self.tbr2 = TemporalBoundaryRegressor(opt)
        self.tbr3 = TemporalBoundaryRegressor(opt)

        self.lgtes = nn.ModuleList(
            [LocalGlobalTemporalEncoder(self.hidden_dim_1d, 0.1, opt['temporal_scale'], opt['window_size']) for i in range(self.lgte_num)])

    def forward(self, x):
        features, video_second, proposals, gt_boxes, temporal_mask = x
        training = self.mode in 'training'
        return self.process(features, gt_boxes, proposals, video_second, training)

    def process(self, features, gt_boxes, proposals, video_sec, training):
        features = self.x_1d_b_f(features)
        for layer in self.lgtes:
            features = layer(features)

        batch_size = proposals.size(0)
        proposals_num = proposals.size(1)
        for i in range(batch_size):
            proposals[i, :, 2] = i
        proposals = proposals.view(batch_size * proposals_num, 3)
        proposals_select = proposals[:, 0:2].sum(dim=1) > 0
        proposals = proposals[proposals_select, :]

        batch_idx = proposals[:, 2].type(torch.long)
        features = features[batch_idx]
        video_sec = video_sec[batch_idx].float()
        if training:
            gt_boxes = gt_boxes.view(batch_size * proposals_num, 2)
            gt_boxes = gt_boxes[proposals_select, :]

        preds_iou1, proposals1, rloss1, iloss1 = self.tbr1(proposals, features, video_sec, gt_boxes, 0.5, training)
        preds_iou2, proposals2, rloss2, iloss2 = self.tbr2(proposals1, features, video_sec, gt_boxes, 0.6, training)
        preds_iou3, proposals3, rloss3, iloss3 = self.tbr3(proposals2, features, video_sec, gt_boxes, 0.7, training)

        if training:
            loss_meta = {"rloss1": rloss1, "rloss2": rloss2, "rloss3": rloss3,
                         "iloss1": iloss1, "iloss2": iloss2, "iloss3": iloss3,
                         "total_loss": rloss1 + rloss2 + rloss3 + iloss1 + iloss2 + iloss3}
            if torch.isnan(loss_meta["total_loss"]):
                from ipdb import set_trace
                set_trace()
            return loss_meta
        else:
            preds_meta = {"proposals1": proposals1, "proposals2": proposals2, "proposals3": proposals3,
                          "iou1": preds_iou1.view(-1), "iou2": preds_iou2.view(-1), "iou3": preds_iou3.view(-1)}
            return preds_meta



