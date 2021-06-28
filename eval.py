# -*- coding: utf-8 -*-
import sys

from evaluation.eval_proposal import ANETproposal
from evaluation.eval_detection import ANETdetection
import matplotlib.pyplot as plt
import numpy as np
import json
from ipdb import set_trace


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def write_results(log):
    with open("./ap_result.txt", "w") as f:
        f.write(log)


def run_proposal_evaluation(ground_truth_filename, proposal_filename,
                            max_avg_nr_proposals=100,
                            tiou_thresholds=np.linspace(0.5, 0.95, 10),
                            subset='validation', assign_class=None):
    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False,
                                 assign_class=assign_class)
    anet_proposal.evaluate()

    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    auc_rate = anet_proposal.auc_rate

    return (average_nr_proposals, average_recall, recall, auc_rate)


def run_detection_evaluation(ground_truth_filename, detection_filename,
                             tiou_thresholds=np.linspace(0.5, 0.95, 10),
                             subset='validation', assign_class=None):
    anet_detection = ANETdetection(ground_truth_filename, detection_filename,
                                   tiou_thresholds=tiou_thresholds,
                                   subset=subset, verbose=True, check_status=False,
                                   assign_class=assign_class)
    anet_detection.evaluate()


def plot_metric(opt, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)

    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2 * idx, :], color=colors[idx + 1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2 * idx] * 100) / 100.),
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(
                int(np.trapz(average_recall, average_nr_proposals) * 100) / 100.),
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')

    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    plt.savefig(opt["save_fig_path"])


def evaluation_proposal(opt):
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid, auc_rate = run_proposal_evaluation(
        opt["video_anno"],
        opt["proposals_result_file"],
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')

    # plot_metric(opt,uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    print("AR@1 is \t", np.mean(uniform_recall_valid[:, 0]))
    print("AR@5 is \t", np.mean(uniform_recall_valid[:, 4]))
    print("AR@10 is \t", np.mean(uniform_recall_valid[:, 9]))
    print("AR@100 is \t", np.mean(uniform_recall_valid[:, -1]))
    return auc_rate


def evaluation_detection(opt, assign_class=None):
    run_detection_evaluation(opt["video_anno"],
                             opt["detection_result_file"],
                             subset='validation',
                             assign_class=assign_class)

