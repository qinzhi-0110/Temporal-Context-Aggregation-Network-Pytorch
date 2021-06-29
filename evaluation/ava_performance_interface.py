import numpy as np
import torch
import sys
sys.path.append('./')
from .ava import object_detection_evaluation
from .ava import standard_fields
from tqdm import tqdm
import pickle
from ipdb import set_trace
def get_labelmap(actions_list):
    labelmap = []
    class_ids = set()
    for i, action in enumerate(actions_list):
        class_id = i + 1
        labelmap.append({"id": class_id, "name": action})
        class_ids.add(class_id)
    return labelmap, class_ids


def prepare_evaluator(actions_list, gt_dict=None):
    categories, class_whitelist = get_labelmap(actions_list)
    pascal_evaluator = object_detection_evaluation.PascalDetectionEvaluator(
        categories)
    if gt_dict is not None:
        add_all_ground_truth(pascal_evaluator, gt_dict)
    return pascal_evaluator


def add_one_ground_truth(pascal_evaluator, image_key, boxes, labels):
    pascal_evaluator.add_single_ground_truth_image_info(
        image_key, {
            standard_fields.InputDataFields.groundtruth_boxes:
                np.array(boxes, dtype=float),
            standard_fields.InputDataFields.groundtruth_classes:
                np.array(labels, dtype=int),
            standard_fields.InputDataFields.groundtruth_difficult:
                np.zeros(len(boxes), dtype=bool)
        })


def add_one_detection(pascal_evaluator, image_key, boxes, labels, scores):
    pascal_evaluator.add_single_detected_image_info(
        image_key, {
            standard_fields.DetectionResultFields.detection_boxes:
                np.array(boxes, dtype=float),
            standard_fields.DetectionResultFields.detection_classes:
                np.array(labels, dtype=int),
            standard_fields.DetectionResultFields.detection_scores:
                np.array(scores, dtype=float)
        })


def split_gt_box_labels(gt_box_list):
    box_list = []
    labels_list = []
    for box in gt_box_list:
        _box = box[:4]
        x1, y1, x2, y2 = _box
        labels = box[4:]
        for label in labels:
            box_list += [[y1, x1, y2, x2]]
            labels_list += [int(label)]
    return box_list, labels_list


def add_all_ground_truth(pascal_evaluator, gt_dict):
    image_key_list = list(gt_dict.keys())
    print("get all ground truth!!")
    for image_key in tqdm(image_key_list, total=len(image_key_list)):
        gt_box_list = gt_dict[image_key]
        box_list, labels_list = split_gt_box_labels(list(gt_box_list))
        add_one_ground_truth(pascal_evaluator, image_key, box_list, labels_list)


def nms_batch(scores, rois, nms_thresh, nms_func=None):
    return_list = []
    for label in range(scores.shape[1]):
        label_score = scores[:, label]
        if label_score.max() < 0.01:
            continue
        s_keep = label_score > 0.01
        boxes = rois[s_keep, :]
        label_score = label_score[s_keep]
        boxes = np.concatenate([boxes, label_score.reshape(-1, 1)], axis=1)
        if nms_func is not None:
            keep = nms_func(boxes, thresh=nms_thresh)
            boxes = boxes[keep, :]
        for j in range(boxes.shape[0]):
            box_list = list(boxes[j, :4])
            return_list += [tuple([boxes[j, 4], label+1] + box_list)]
    return return_list


def add_one_batch_gt(pascal_evaluator, image_key, gt_boxes, bin_labels):
    keep = bin_labels.sum(1)>0
    gt_boxes = gt_boxes[keep, :]
    bin_labels = bin_labels[keep, :]
    gt_box_list = []
    for i in range(gt_boxes.shape[0]):
        bin_label = bin_labels[i, :]
        labels = np.where(bin_label > 0)[0]
        gt_box_list += [list(gt_boxes[i]) + list(labels + 1)]
    box_list, labels_list = split_gt_box_labels(gt_box_list)
    add_one_ground_truth(pascal_evaluator, image_key, box_list, labels_list)


def train_eval_handle_batch(pascal_evaluator, image_key, proposals, scores, gt_boxes, bin_labels):
    capacity=50
    if isinstance(proposals, torch.Tensor):
        proposals = proposals.numpy()
        scores = scores.numpy()
        gt_boxes = gt_boxes.numpy()
        bin_labels = bin_labels.numpy()
    add_one_batch_gt(pascal_evaluator, image_key, gt_boxes, bin_labels)
    if len(proposals) == 0:
        return
    pro_list = nms_batch(scores, proposals[:, 1:5], 0, nms_func=None)
    pro_list = sorted(pro_list, key=lambda tup: -tup[0])
    pro_list = pro_list[:capacity]
    boxes = []
    labels = []
    scores = []
    for item in pro_list:
        score, action_id, x1, y1, x2, y2 = item
        boxes.append([y1, x1, y2, x2])
        labels.append(action_id)
        scores.append(score)
    if len(scores)>0 :
        try:
            add_one_detection(pascal_evaluator, image_key, boxes, labels, scores)
        except:
            set_trace()
            print("stop!!")


def train_eval_handle_batch_faster_rcnn(pascal_evaluator, image_key, proposals, scores, labels, gt_boxes, bin_labels):
    if isinstance(proposals, torch.Tensor):
        proposals = proposals.numpy()
        scores = scores.numpy()
        gt_boxes = gt_boxes.numpy()
        bin_labels = bin_labels.numpy()
    add_one_batch_gt(pascal_evaluator, image_key, gt_boxes, bin_labels)
    pro_list = [[scores[i], labels[i], *tuple(proposals[i, :])] for i in range(proposals.shape[0])]
    pro_list = sorted(pro_list, key=lambda tup: -tup[0])
    boxes = []
    labels = []
    scores = []
    for item in pro_list:
        score, action_id, x1, y1, x2, y2 = item
        boxes.append([y1, x1, y2, x2])
        labels.append(action_id)
        scores.append(score)
    if len(scores) > 0:
        try:
            add_one_detection(pascal_evaluator, image_key, boxes, labels, scores)
        except:
            set_trace()
            print("stop!!")


def train_eval_collect_results(pascal_evaluator, frame_key_tensor, proposals, scores, gt_boxes, bin_labels):
    batch_size = len(frame_key_tensor[0])
    image_key = []
    for i, video_name in enumerate(frame_key_tensor[0]):
        image_key += [(video_name, frame_key_tensor[1][i].item())]
    for i, ik in enumerate(image_key):
        train_eval_handle_batch(pascal_evaluator, ik, proposals[i, :, :], scores[i, :, :], gt_boxes[i, :, :], bin_labels[i, :, :])

def train_eval_collect_results_person_image(pascal_evaluator, image_key, proposals_list, scores_list, \
                                            gt_boxes_list, bin_labels_list):
    for i, ik in enumerate(image_key):
        train_eval_handle_batch(pascal_evaluator, tuple(ik), proposals_list[i], scores_list[i], \
                                gt_boxes_list[i], bin_labels_list[i])

def train_eval_collect_results_faster_rcnn(pascal_evaluator, frame_key_tensor, proposals_list, scores_list,
                                           labels_list, gt_boxes, bin_labels):
    image_key = []
    for i, video_name in enumerate(frame_key_tensor[0]):
        image_key += [(video_name, frame_key_tensor[1][i].item())]
    for i, ik in enumerate(image_key):
        train_eval_handle_batch_faster_rcnn(pascal_evaluator, ik, proposals_list[i], scores_list[i], labels_list[i],
                                            gt_boxes[i, :, :], bin_labels[i, :, :])

def collect_one_results(pascal_evaluator, image_key, proposals, scores, gt_boxes, bin_labels):
    train_eval_handle_batch(pascal_evaluator, image_key, proposals, scores, gt_boxes, bin_labels)

def train_eval_get_results(pascal_evaluator, write_log=None, metrics_path=None, epoch=None):
    metrics = pascal_evaluator.evaluate()
    ap_dict = {}
    mAP = 0
    for name, ap in metrics.items():
        if 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/' in name:
            name = name.replace('PascalBoxes_PerformanceByCategory/AP@0.5IOU/', '')
        if 'mAP' in name:
            mAP = ap
            continue
        ap_dict[name] = ap
    actions_list = list(ap_dict.keys())
    actions_list.sort()
    log = 'epoch:{}  frame_map: {} \n'.format(epoch, mAP)
    for action in actions_list:
        log += '{}: {}\n'.format(action, ap_dict[action])
    if write_log is not None:
        write_log(log)
        print("frame_map:{} ,log write ok!".format(mAP))
    else:
        print(log)
    if metrics_path is not None:
        with open(metrics_path + "metrics-%.3f.pkl" % mAP, 'wb') as f:
            pickle.dump(metrics, f)
    return mAP



if __name__ == '__main__':
    pass

