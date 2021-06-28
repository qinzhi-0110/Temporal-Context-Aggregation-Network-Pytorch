# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle
import json
import torch.utils.data as data
import torch
import random
import math
from ipdb import set_trace
from tqdm import tqdm


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
    return json_data


def load_pickle_feature(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def collect_gt(video_info, gt_list):
    video_labels = video_info['annotations']  # the measurement is second, not frame
    video_second = float(video_info['duration'])
    for j in range(len(video_labels)):
        tmp_info = video_labels[j]
        tmp_start = (tmp_info['segment'][0] / video_second)
        tmp_end = (tmp_info['segment'][1] / video_second)
        if tmp_start < tmp_end:
            gt_list.append([tmp_start, tmp_end])


class TCADataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        mode_map = {"train": "training",
                    "validation": "validation",
                    "test": "testing"}
        self.path_prefix = mode_map[subset]
        self.temporal_scale = opt["temporal_scale"]  # 100
        self.temporal_gap = 1. / self.temporal_scale
        self.subset = subset
        self.mode = opt["mode"]
        self.feature_path = opt["feature_path"]
        self.video_anno_path = opt["video_anno"]
        if opt["mode"] in 'training':
            self.proposals_path = opt["train_proposals_path"]
        elif opt["mode"] in 'inference':
            self.proposals_path = opt["test_proposals_path"]

        self._getDatasetDict()
        self._splitPart(opt)

    def _getDatasetDict(self):
        ignore_videos = ['S8GtH2Zayds', 'saZkh1Xacp0', 'uRCf7b3qk0I', 'ukyFvye2yK0', 'VsZiOEzQqyI', 'KhkQyn-WblM']
        anno_database = load_json(self.video_anno_path)
        anno_database = anno_database['database']
        self.video_dict = {}
        self.dirty_instance_cnt = 0
        self.dirty_video_cnt = 0
        gt_list = []
        for video_name, anno in tqdm(anno_database.items(), total=len(anno_database)):
            video_subset = anno['subset']
            if 'train' in video_subset:
                collect_gt(anno, gt_list)
            if self.subset not in video_subset:
                continue
            if video_name in ignore_videos:
                self.dirty_video_cnt += 1
                continue
            if self.mode in "training":
                video_info = self._filter_dirty_data(anno)
            else:
                video_info = anno
            if video_info is None:
                self.dirty_video_cnt += 1
                continue
            self.video_dict[video_name] = video_info
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d, drop instance: %d drop video: %d" % (self.subset,
                                                                                 len(self.video_list),
                                                                                 self.dirty_instance_cnt,
                                                                                 self.dirty_video_cnt))
        return gt_list

    def _splitPart(self, opt):
        if opt["part_idx"] >= 0:
            part_num = 4
            assert part_num > opt["part_idx"]
            self.video_list.sort()
            avg_num = math.ceil(len(self.video_list) / part_num)
            self.video_list = self.video_list[opt["part_idx"]*avg_num:(opt["part_idx"]+1)*avg_num]
            print("partidx:{}  total count:{}".format(opt["part_idx"], len(self.video_list)))
        else:
            print("not split part!")
    def _filter_dirty_data(self, anno):
        new_anno = {"annotations": [],
                    "duration": anno["duration"],
                    "subset": anno["subset"]}
        for a in anno["annotations"]:
            if (a['segment'][1] - a['segment'][0]) > 1:
                new_anno["annotations"].append(a)
            else:
                self.dirty_instance_cnt += 1
        if len(new_anno["annotations"]) > 0:
            return new_anno
        else:
            return None

    def __getitem__(self, index):
        video_data, video_duration, video_name = self._load_feature(index)
        xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax = self._load_proposals(video_name, video_duration)
        meta = {}
        if self.mode in 'training':
            topk = 64
        else:
            topk = xmin.shape[0]
            # topk = 100
        feature, proposals, gt_boxes, feature_len, temporal_mask = self._sample_data_for_tcanet(video_data,
                                                                                                 xmin, xmax,
                                                                                                 topk,
                                                                                                 video_duration,
                                                                                                 gt_xmin, gt_xmax)
        meta["features"] = feature
        meta["gt_boxes"] = gt_boxes
        meta["proposals"] = proposals
        meta["feature_len"] = torch.tensor([feature_len])
        meta["video_duration"] = torch.tensor([video_duration])
        meta['temporal_mask'] = temporal_mask
        if self.mode not in 'training':
            meta["score"] = score
        return meta

    def _load_feature(self, index):
        video_name = self.video_list[index]
        video_info = self.video_dict[video_name]
        video_duration = float(video_info['duration'])
        feature_path_list = self.feature_path.split(',')
        slow_feature_list = []
        fast_feature_list = []
        feature_frame_list = []
        for feature_path in feature_path_list:
            if 'hacs_segments_features' in feature_path:
                feature = np.load(os.path.join(feature_path, self.path_prefix, video_name + ".npy"))
                video_df = {}
                video_df['slow_feature'] = torch.from_numpy(feature).permute(1, 0).unsqueeze(0)
                video_df['fast_feature'] = torch.zeros(1, 0)
                video_df['feature_frames'] = 0
            else:
                video_df = load_pickle_feature(os.path.join(feature_path, self.path_prefix, video_name + ".pkl"))
            slow_feature_list.append(video_df['slow_feature'])
            fast_feature_list.append(video_df['fast_feature'])
            feature_frame_list.append(video_df["feature_frames"])
        slow_feature = torch.cat(slow_feature_list, dim=1)
        fast_feature = torch.cat(fast_feature_list, dim=1)
        feature_frame = list(set(feature_frame_list))[0]
        feature_list = []
        if slow_feature.numel() > 0:
            feature_list += [slow_feature]
        if fast_feature.numel() > 0:
            feature_list += [fast_feature]
        video_data = torch.cat(feature_list, dim=1)
        return video_data, video_duration, video_name

    def _load_proposals(self, video_name, duration):
        mode = 1
        if mode == 1:
            df = pd.read_csv(os.path.join(self.proposals_path, video_name + ".csv"))
            xmin = df.xmin.values[:]
            xmax = df.xmax.values[:]
            score = df.score.values[:]
            iou = df.iou.values[:]
            ioa = df.ioa.values[:]
            if self.mode in "training":
                gt_xmin = df.gt_xmin.values[:]
                gt_xmax = df.gt_xmax.values[:]
            else:
                gt_xmin = np.zeros_like(iou)
                gt_xmax = np.zeros_like(iou)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax
        elif mode == 2:
            xmin = self.anchors[:, 0] * duration
            xmax = self.anchors[:, 1] * duration
            score = np.ones_like(xmin)
            iou = np.zeros_like(xmin)
            ioa = np.zeros_like(xmin)
            gt_xmin = np.zeros_like(xmin)
            gt_xmax = np.zeros_like(xmin)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax
        elif mode == 3:
            df = pd.read_csv(os.path.join(self.proposals_path, video_name + ".csv"))
            xmin = df.xmin.values[:]
            xmax = df.xmax.values[:]
            score = df.score.values[:]
            xmin_score = df.xmin_score.values[:]
            xmax_score = df.xmax_score.values[:]
            score = np.stack([score, xmin_score, xmax_score], axis=1)
            iou = df.iou.values[:]
            ioa = df.ioa.values[:]
            if self.mode in "training":
                gt_xmin = df.gt_xmin.values[:]
                gt_xmax = df.gt_xmax.values[:]
            else:
                gt_xmin = np.zeros_like(iou)
                gt_xmax = np.zeros_like(iou)
            return xmin, xmax, score, iou, ioa, gt_xmin, gt_xmax

    def _sample_data_for_tcanet(self, feature, xmin, xmax, topk, video_duration, gt_xmin, gt_xmax):
        if topk > xmin.shape[0]:
            rel_topk = xmin.shape[0]
        else:
            rel_topk = topk
        feature_len = feature.size(2)
        # if self.mode in 'training':
        t_max = self.temporal_scale
        if feature.size(2) <= t_max:
            full_feature = torch.zeros((feature.size(1), t_max))
            assert full_feature.size(1) >= feature.size(2)
        else:
            full_feature = torch.zeros((feature.size(1), feature.size(2)))
        temporal_mask = torch.ones((full_feature.size(1),), dtype=torch.bool)
        full_feature[:, :feature.size(2)] = feature[0, :, :]
        temporal_mask[:feature.size(2)] = False

        proposals = torch.zeros((topk, 3))
        gt_boxes = torch.zeros((topk, 2))
        proposals[:rel_topk, 0] = torch.from_numpy(xmin[:rel_topk])
        proposals[:rel_topk, 1] = torch.from_numpy(xmax[:rel_topk])
        gt_boxes[:rel_topk, 0] = torch.from_numpy(gt_xmin[:rel_topk])
        gt_boxes[:rel_topk, 1] = torch.from_numpy(gt_xmax[:rel_topk])
        return full_feature, proposals, gt_boxes, feature_len, temporal_mask

    def __len__(self):
        return len(self.video_list)



