import json
import urllib.request as urllib2

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from evaluation.eval_utils import get_blocked_videos
from evaluation.eval_utils import interpolated_prec_rec
from evaluation.eval_utils import segment_iou
from evaluation.eval_utils import wrapper_segment_iou

class ANETproposal(object):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PROPOSAL_FIELDS = ['results']

    def __init__(self, ground_truth_filename=None, proposal_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 proposal_fields=PROPOSAL_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 max_avg_nr_proposals=None,
                 subset='validation', verbose=False,
                 check_status=True,
                 assign_class=None):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not proposal_filename:
            raise IOError('Please input a valid proposal file.')
        self.assign_class = assign_class
        if assign_class is not None:
            print("assign_class:{} for proposals".format(assign_class))
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.max_avg_nr_proposals = max_avg_nr_proposals
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = proposal_fields
        self.recall = None
        self.avg_recall = None
        self.proposals_per_video = None
        self.check_status = check_status
        # Retrieve blocked videos from server.
        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()
        # Import ground truth and proposals.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename)
        self.proposal = self._import_proposal(proposal_filename)

        if self.verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.proposal)
            print('\tNumber of proposals: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            if self.assign_class is not None and v['annotations'][0]['label'] != self.assign_class:
                continue
            '''
            # qzw
            if float(v['duration']) >= 20:
                continue
            '''
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                '''
                # qzw
                temporal_len = float(ann['segment'][1]) - float(ann['segment'][0])
                if temporal_len >= 42.666/2.:
                    continue
                '''
                '''
                temporal_len = float(ann['segment'][1]) - float(ann['segment'][0])
                if temporal_len / float(v['duration']) < 0.1:
                    continue
                '''
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        return ground_truth, activity_index

    def _import_proposal(self, proposal_filename):
        """Reads proposal file, checks if it is well formatted, and returns
           the proposal instances.

        Parameters
        ----------
        proposal_filename : str
            Full path to the proposal json file.

        Outputs
        -------
        proposal : df
            Data frame containing the proposal instances.
        """
        with open(proposal_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid proposal file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        score_lst = []
        ground_truth_video_list = list(self.ground_truth['video-id'])
        for videoid, v in data['results'].items():
            videoid = videoid[2:]
            if videoid in self.blocked_videos:
                continue
            for result in v:
                '''
                # qzw
                if (float(result['segment'][1]) - float(result['segment'][0])) > 42.6666/2.:
                    continue
                '''
                if videoid not in ground_truth_video_list:
                    continue
                if self.assign_class is not None and videoid not in ground_truth_video_list:
                    continue
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                score_lst.append(result['score'])
        proposal = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'score': score_lst})
        return proposal

    def evaluate(self):
        """Evaluates a proposal file. To measure the performance of a
        method for the proposal task, we computes the area under the 
        average recall vs average number of proposals per video curve.
        """
        recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
                self.ground_truth, self.proposal,
                max_avg_nr_proposals=self.max_avg_nr_proposals,
                tiou_thresholds=self.tiou_thresholds)

        area_under_curve = np.trapz(avg_recall, proposals_per_video)
        self.auc_rate = 100.*float(area_under_curve)/proposals_per_video[-1]
        if self.verbose:
            print('[RESULTS] Performance on ActivityNet proposal task.')
            print('\tArea Under the AR vs AN curve: {}%'.format(100.*float(area_under_curve)/proposals_per_video[-1]))
        '''
        from ipdb import set_trace
        set_trace()
        '''
        self.recall = recall
        self.avg_recall = avg_recall
        self.proposals_per_video = proposals_per_video
        # print("recall:{}  avg_recall:{}  proposals_per_video:{}".format(recall, avg_recall, proposals_per_video))

def average_recall_vs_avg_nr_proposals(ground_truth, proposals,
                                       max_avg_nr_proposals=None,
                                       tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """ Computes the average recall given an average number 
        of proposals per video.
    
    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    proposal : df
        Data frame containing the proposal instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        array with tiou thresholds.
        
    Outputs
    -------
    recall : 2darray
        recall[i,j] is recall at ith tiou threshold at the jth average number of average number of proposals per video.
    average_recall : 1darray
        recall averaged over a list of tiou threshold. This is equivalent to recall.mean(axis=0).
    proposals_per_video : 1darray
        average number of proposals per video.
    """

    # Get list of videos.
    video_lst = ground_truth['video-id'].unique()

    if not max_avg_nr_proposals:
        max_avg_nr_proposals = float(proposals.shape[0])/video_lst.shape[0]

    ratio = max_avg_nr_proposals*float(video_lst.shape[0])/proposals.shape[0]

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')
    proposals_gbvn = proposals.groupby('video-id')

    # For each video, computes tiou scores among the retrieved proposals.
    score_lst = []
    total_nr_proposals = 0
    
    from ipdb import set_trace
    '''
    success_iou = [0.5, 0.6, 0.7, 0.8, 0.9]
    success_cnt = [[], [], [], [], [], []]
    gt_cnt = []
    '''
    temporal_interval = [[0, 5], [5, 10], [10, 50], [50, 200]]
    gt_temporal_cnt = [0 for i in range(len(temporal_interval))]
    recall_success_temporal_cnt = [0 for i in range(len(temporal_interval))]
    for videoid in video_lst:
        # Get ground-truth instances associated to this video.
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)
        this_video_ground_truth = ground_truth_videoid.loc[:,['t-start', 't-end']].values

        # Get proposals for this video.
        try:
            proposals_videoid = proposals_gbvn.get_group(videoid)
        except:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        this_video_proposals = proposals_videoid.loc[:, ['t-start', 't-end']].values

        if this_video_proposals.shape[0] == 0:
            n = this_video_ground_truth.shape[0]
            score_lst.append(np.zeros((n, 1)))
            continue

        # Sort proposals by score.
        sort_idx = proposals_videoid['score'].argsort()[::-1]
        this_video_proposals = this_video_proposals[sort_idx, :]

        if this_video_proposals.ndim != 2:
            this_video_proposals = np.expand_dims(this_video_proposals, axis=0)
        if this_video_ground_truth.ndim != 2:
            this_video_ground_truth = np.expand_dims(this_video_ground_truth, axis=0)

        nr_proposals = np.minimum(int(this_video_proposals.shape[0] * ratio), this_video_proposals.shape[0])
        total_nr_proposals += nr_proposals
        this_video_proposals = this_video_proposals[:nr_proposals, :]
        # Compute tiou scores.
        tiou = wrapper_segment_iou(this_video_proposals, this_video_ground_truth)
        score_lst.append(tiou)
        if tiou.size == 0:
            set_trace()
        gt_temporal_length = this_video_ground_truth[:, 1] - this_video_ground_truth[:, 0]
        for idx, (inter_start, inter_end) in enumerate(temporal_interval):
            gt_mask = (gt_temporal_length > inter_start) * (gt_temporal_length <= inter_end)
            gt_temporal_cnt[idx] += gt_mask.sum()
            recall_iou = tiou.max(axis=1)
            recall_mask = recall_iou > 0.5
            recall_success_temporal_cnt[idx] += (gt_mask * recall_mask).sum()
        '''
        gt_cnt += (this_video_ground_truth[:, 1] - this_video_ground_truth[:, 0]).tolist()
        if tiou.size == 0:
            continue
        for ridx, iou in enumerate(success_iou):
            recall_iou = tiou.max(axis=1)
            recall_mask = recall_iou > iou
            gt_success = this_video_ground_truth[recall_mask, :]
            gt_length = (gt_success[:, 1] - gt_success[:, 0]).tolist()
            success_cnt[ridx] += gt_length
        '''
    for idx, (inter_start, inter_end) in enumerate(temporal_interval):
        recall_rate = recall_success_temporal_cnt[idx] / gt_temporal_cnt[idx]
        print("temporal interval ({}, {}] recall_rate:{}".format(inter_start, inter_end, recall_rate))
    print("total recall rate:{}".format(sum(recall_success_temporal_cnt)/sum(gt_temporal_cnt)))
    # plot_recall_dict(success_iou, success_cnt, gt_cnt)

    # Given that the length of the videos is really varied, we 
    # compute the number of proposals in terms of a ratio of the total 
    # proposals retrieved, i.e. average recall at a percentage of proposals 
    # retrieved per video.
    # Computes average recall.
    pcn_lst = np.arange(1, 101) / 100.0 *(max_avg_nr_proposals*float(video_lst.shape[0])/total_nr_proposals)
    matches = np.empty((video_lst.shape[0], pcn_lst.shape[0]))
    positives = np.empty(video_lst.shape[0])
    recall = np.empty((tiou_thresholds.shape[0], pcn_lst.shape[0]))
    # Iterates over each tiou threshold.
    for ridx, tiou in enumerate(tiou_thresholds):

        # Inspect positives retrieved per video at different 
        # number of proposals (percentage of the total retrieved).
        for i, score in enumerate(score_lst):
            # Total positives per video.
            positives[i] = score.shape[0]
            # Find proposals that satisfies minimum tiou threshold.
            true_positives_tiou = score >= tiou
            # Get number of proposals as a percentage of total retrieved.
            pcn_proposals = np.minimum((score.shape[1] * pcn_lst).astype(np.int), score.shape[1])

            for j, nr_proposals in enumerate(pcn_proposals):
                # Compute the number of matches for each percentage of the proposals
                matches[i, j] = np.count_nonzero((true_positives_tiou[:, :nr_proposals]).sum(axis=1))

        # Computes recall given the set of matches per video.
        recall[ridx, :] = matches.sum(axis=0) / positives.sum()

    # Recall is averaged.
    avg_recall = recall.mean(axis=0)

    # Get the average number of proposals per video.
    proposals_per_video = pcn_lst * (float(total_nr_proposals) / video_lst.shape[0])

    return recall, avg_recall, proposals_per_video


def plot_recall_dict(success_iou, success_cnt, gt_cnt):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('Agg')
    from ipdb import set_trace
    def collect_gt_frequency(gt_cnt, bins):
        gt_cnt = np.array(gt_cnt)
        gt_frequency = []
        for s in range(len(bins) - 1):
            mask = (gt_cnt >= bins[s]) * (gt_cnt < bins[s+1])
            gt_frequency += [mask.sum()]
        return gt_frequency
    def trans_bins(bins):
        center_list = []
        for s in range(len(bins) - 1):
            center_list += [(bins[s] + bins[s+1])/2]
        return center_list

    def plot_statistics_result(data_list, gt_cnt, x_label, y_label, title, save_path):
        n, bins, patches = plt.hist(x=data_list, bins=20, color='#0504aa', alpha=0.7, rwidth=0.85)
        gt_frequency = collect_gt_frequency(gt_cnt, bins)
        plt.close()
        rate = (n / np.array(gt_frequency)).tolist()
        center_list = trans_bins(bins)
        plt.bar(center_list, rate, width=10)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

    for i, iou in enumerate(success_iou):
        plot_statistics_result(success_cnt[i], 
                               gt_cnt,
                               x_label='temporal length',
                               y_label='recall success frequency',
                               title='HACS temporal length recall with iou :{}'.format(iou),
                               save_path='/mnt/data/FlylineData/flq/dataset/HACS/videos/recall_with_temporal_length/bmn/IOU-{}.png'.format(iou))



