import sys
from tca_dataset import TCADataSet
import os
import json
import math
import torch
import torch.nn.parallel
import torch.optim as optim
import numpy as np
import opts
from tqdm import tqdm
from tcanet import TCANet
import time
import pandas as pd
from post_processing import TCA_post_processing
from eval import evaluation_proposal, evaluation_detection
from ipdb import set_trace

sys.dont_write_bytecode = True


def train_TCANet(data_loader, model, optimizer, epoch, warm_scheduler=None):
    model.train()
    epoch_loss_sum = 0
    iou_loss1_sum = 0
    iou_loss2_sum = 0
    iou_loss3_sum = 0

    reg_loss1_sum = 0
    reg_loss2_sum = 0
    reg_loss3_sum = 0

    total_iter = len(data_loader)
    log_period = 200
    time_now = time.time()
    for n_iter, meta in enumerate(data_loader):
        features = meta["features"]
        gt_boxes = meta["gt_boxes"]
        proposals = meta["proposals"]
        video_second = meta["video_duration"]
        temporal_mask = meta['temporal_mask']
        if warm_scheduler is not None:
            warm_scheduler.step()
        features = features.cuda()
        gt_boxes = gt_boxes.cuda()
        proposals = proposals.cuda()
        video_second = video_second.cuda()
        input = (features, video_second, proposals, gt_boxes, temporal_mask)
        loss_meta = model(input)
        n_list = list(loss_meta.keys())
        for n in n_list:
            loss_meta[n] = loss_meta[n].mean()
        total_loss = loss_meta["total_loss"]
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss_sum += total_loss.cpu().detach().numpy()
        iou_loss1_sum += loss_meta["iloss1"].cpu().detach().numpy()
        iou_loss2_sum += loss_meta["iloss2"].cpu().detach().numpy()
        iou_loss3_sum += loss_meta["iloss3"].cpu().detach().numpy()

        reg_loss1_sum += loss_meta["rloss1"].cpu().detach().numpy()
        reg_loss2_sum += loss_meta["rloss2"].cpu().detach().numpy()
        reg_loss3_sum += loss_meta["rloss3"].cpu().detach().numpy()


        if n_iter % log_period == 0:
            derta = time.time() - time_now
            avg_time = derta / (n_iter + 1)
            end_time = time.time() + avg_time * (total_iter - n_iter + 1)
            print(
                "TCA training iter %d/%d lr:%s  total loss(epoch %d):%.03f iloss1:%.03f iloss2:%.03f iloss3:%.03f rloss1:%0.3f rloss2:%.03f rloss3:%.03f until:%s" % (
                    n_iter + 1, total_iter, str(["%.06f" % group['lr'] for group in optimizer.param_groups]),
                    epoch,
                    epoch_loss_sum / (n_iter + 1),
                    iou_loss1_sum / (n_iter + 1),
                    iou_loss2_sum / (n_iter + 1),
                    iou_loss3_sum / (n_iter + 1),
                    reg_loss1_sum / (n_iter + 1),
                    reg_loss2_sum / (n_iter + 1),
                    reg_loss3_sum / (n_iter + 1),
                    time.asctime(time.localtime(end_time))
                ))


def test_TCANet(data_loader, model, epoch):
    model.eval()
    best_loss = 1e10
    epoch_loss_sum = 0
    iou_loss1_sum = 0
    iou_loss2_sum = 0
    iou_loss3_sum = 0

    reg_loss1_sum = 0
    reg_loss2_sum = 0
    reg_loss3_sum = 0

    log_period = 200
    time_now = time.time()
    total_iter = len(data_loader)
    for n_iter, meta in enumerate(data_loader):
        features = meta["features"]
        gt_boxes = meta["gt_boxes"]
        proposals = meta["proposals"]
        video_second = meta["video_duration"]
        temporal_mask = meta['temporal_mask']
        features = features.cuda()
        gt_boxes = gt_boxes.cuda()
        proposals = proposals.cuda()
        video_second = video_second.cuda()
        input = (features, video_second, proposals, gt_boxes, temporal_mask)
        loss_meta = model(input)
        n_list = list(loss_meta.keys())
        for n in n_list:
            loss_meta[n] = loss_meta[n].mean()
        total_loss = loss_meta["total_loss"]

        epoch_loss_sum += total_loss.cpu().detach().numpy()
        iou_loss1_sum += loss_meta["iloss1"].cpu().detach().numpy()
        iou_loss2_sum += loss_meta["iloss2"].cpu().detach().numpy()
        iou_loss3_sum += loss_meta["iloss3"].cpu().detach().numpy()

        reg_loss1_sum += loss_meta["rloss1"].cpu().detach().numpy()
        reg_loss2_sum += loss_meta["rloss2"].cpu().detach().numpy()
        reg_loss3_sum += loss_meta["rloss3"].cpu().detach().numpy()

        if n_iter % log_period == 0:
            derta = time.time() - time_now
            avg_time = derta / (n_iter + 1)
            end_time = time.time() + avg_time * (total_iter - n_iter + 1)
            print(
                "TCA testing iter %d/%d total loss(epoch %d):%.03f iloss1:%.03f iloss2:%.03f iloss3:%.03f rloss1:%0.3f rloss2:%.03f rloss3:%.03f until:%s" % (
                    n_iter + 1, total_iter,
                    epoch,
                    epoch_loss_sum / (n_iter + 1),
                    iou_loss1_sum / (n_iter + 1),
                    iou_loss2_sum / (n_iter + 1),
                    iou_loss3_sum / (n_iter + 1),
                    reg_loss1_sum / (n_iter + 1),
                    reg_loss2_sum / (n_iter + 1),
                    reg_loss3_sum / (n_iter + 1),
                    time.asctime(time.localtime(end_time))
                ))
    print(
        "TCA test loss(epoch %d): %.03f" % (
            epoch,
            epoch_loss_sum / (n_iter + 1)))
    print(
        "TCA testing total loss(epoch %d):%.03f iloss1:%.03f iloss2:%.03f iloss3:%.03f rloss1:%0.3f rloss2:%.03f rloss3:%.03f until:%s" % (
            epoch,
            epoch_loss_sum / (n_iter + 1),
            iou_loss1_sum / (n_iter + 1),
            iou_loss2_sum / (n_iter + 1),
            iou_loss3_sum / (n_iter + 1),
            reg_loss1_sum / (n_iter + 1),
            reg_loss2_sum / (n_iter + 1),
            reg_loss3_sum / (n_iter + 1),
            time.asctime(time.localtime(end_time))
        ))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, os.path.join(opt["checkpoint_path"], "TCA_checkpoint.pth.tar"))
    if epoch_loss_sum < best_loss:
        best_loss = epoch_loss_sum
        torch.save(state, os.path.join(opt["checkpoint_path"], "TCA_best.pth.tar"))


def save_check_point(model, optimizer, epoch, opt):
    if epoch < 5:
        return
    if torch.cuda.device_count() > 1:
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()
    optimizer_dict = optimizer.state_dict()
    checkpoint_path = os.path.join(opt["checkpoint_path"], "epoch-{}.pth.tar".format(epoch))
    checkpoint_dict = {"epoch": epoch,
                       "model_state": model_dict,
                       "optimizer_dict": optimizer_dict}
    torch.save(checkpoint_dict, checkpoint_path)
    print("checkpoint_path:{} has been saved!".format(checkpoint_path))


def data2cuda(state_dict):
    keys = list(state_dict)
    for key in keys:
        if type(state_dict[key]) is dict:
            data2cuda(state_dict[key])
        elif type(state_dict[key]) is torch.Tensor:
            if state_dict[key].device.type == 'cpu':
                state_dict[key] = state_dict[key].cuda()


def load_resume_epoch(opt, model, optimizer):
    if opt['continue_epoch'] > 0:
        checkpoint = torch.load(os.path.join(opt["checkpoint_path"], "epoch-{}.pth.tar".format(opt['continue_epoch'])))
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        data2cuda(optimizer.state)
        return checkpoint['epoch']
    else:
        return -1


def TCA_Train(opt):
    model = TCANet('training', opt)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt["training_lr"],
                            weight_decay=opt["weight_decay"])
    c_epoch = load_resume_epoch(opt, model, optimizer) + 1
    model = torch.nn.DataParallel(model).cuda()

    train_loader = torch.utils.data.DataLoader(TCADataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=opt["num_workers"], pin_memory=True)

    test_loader = torch.utils.data.DataLoader(TCADataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"] // 2, shuffle=False,
                                              num_workers=opt["num_workers"], pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    test_TCANet(test_loader, model, -1)
    for epoch in range(opt["train_epochs"]):
        scheduler.step()
        if epoch < c_epoch:
            # scheduler.current_iter = (epoch-1)*len(train_loader)
            continue
        if epoch == 0:
            from lr_scheduler import CosLrWarmupScheduler
            warm_scheduler = CosLrWarmupScheduler(optimizer, 1000)
        else:
            warm_scheduler = None
        train_TCANet(train_loader, model, optimizer, epoch, warm_scheduler)
        test_TCANet(test_loader, model, epoch)
        save_check_point(model, optimizer, epoch, opt)


def TCA_inference(opt):
    model = TCANet('inference', opt)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(os.path.join(opt["checkpoint_path"], "TCA_best.pth.tar"))
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except Exception as e:
        print("{}  continue?".format(e))
        model.module.load_state_dict(checkpoint['model_state'])
    model.eval()
    test_loader = torch.utils.data.DataLoader(TCADataSet(opt, subset=opt["inference_dataset"]),
                                              batch_size=1, shuffle=False,
                                              num_workers=opt["num_workers"], pin_memory=True, drop_last=False)
    total_iter = len(test_loader)
    log_period = 20
    time_start = time.time()
    max_proposals = 200 // 2
    gpu_count = torch.cuda.device_count()

    with torch.no_grad():
        for n_iter, meta in tqdm(enumerate(test_loader), total=len(test_loader)):
            features = meta["features"]
            proposals = meta["proposals"]
            video_second = meta["video_duration"]
            temporal_mask = meta['temporal_mask']
            score = meta["score"]
            video_name = test_loader.dataset.video_list[n_iter]
            score = score.squeeze(0)
            features = features.cuda()
            proposals = proposals.cuda()
            video_second = video_second.cuda()
            temporal_mask = temporal_mask.cuda()
            iter_num = math.ceil(proposals.size(1) / max_proposals)
            proposals2_list = []
            iou2_list = []
            proposals3_list = []
            iou3_list = []
            proposals4_list = []
            iou4_list = []
            for p_num in range(iter_num):
                input = (features, video_second, proposals[:, p_num * max_proposals:(p_num + 1) * max_proposals, :], None, temporal_mask)
                preds_meta = model(input)
                proposals2_list.append(preds_meta["proposals1"].cpu())
                iou2_list.append(preds_meta["iou1"].cpu())
                proposals3_list.append(preds_meta["proposals2"].cpu())
                iou3_list.append(preds_meta["iou2"].cpu())
                proposals4_list.append(preds_meta["proposals3"].cpu())
                iou4_list.append(preds_meta["iou3"].cpu())
            all_proposals2 = torch.cat(proposals2_list, dim=0)
            all_ious2 = torch.cat(iou2_list, dim=0).numpy()
            all_proposals3 = torch.cat(proposals3_list, dim=0)
            all_ious3 = torch.cat(iou3_list, dim=0).numpy()
            all_proposals4 = torch.cat(proposals4_list, dim=0)
            all_ious4 = torch.cat(iou4_list, dim=0).numpy()
            all_proposals = all_proposals4
            all_ious = all_ious4
            video_duration = meta["video_duration"].item()

            proposals = all_proposals.numpy() / video_duration
            proposals4 = all_proposals4.numpy() / video_duration
            proposals3 = all_proposals3.numpy() / video_duration
            proposals2 = all_proposals2.numpy() / video_duration
            score = score.numpy()
            # all_ious = all_ious.numpy()
            #########################################################################
            if len(score.shape) > 1:
                xmin_score = score[:, 1]
                xmax_score = score[:, 2]
                score = score[:, 0]
            else:
                xmin_score = score
                xmax_score = score
            new_score = score * all_ious
            # new_props = np.stack([proposals[:, 0], proposals[:, 1], new_score, score, all_ious], axis=1)
            new_props = np.stack(
                [proposals4[:, 0], proposals4[:, 1], proposals3[:, 0], proposals3[:, 1], proposals2[:, 0],
                 proposals2[:, 1], new_score, score, xmin_score, xmax_score,
                 all_ious, all_ious3, all_ious2], axis=1)

            col_name = ["xmin", "xmax", "xmin3", "xmax3", "xmin2", "xmax2", "score", "ori_score", "xmin_score",
                        "xmax_score", "pred_iou", "preds_iou3", "preds_iou2"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv(
                os.path.join(opt["checkpoint_path"], "TCA_results", video_name + ".csv"),
                index=False)
            current_iter = n_iter
            if current_iter % log_period == 0 and False:
                derta = time.time() - time_start
                avg_time = derta / (current_iter + 1)
                end_time = time.time() + avg_time * (total_iter - current_iter + 1)
                print(
                    "TCA inference iter %d/%d   until:%s" % (
                        current_iter + 1, total_iter,
                        time.asctime(time.localtime(end_time))
                    ))


def main(opt):
    if opt["mode"] == "train":
        TCA_Train(opt)
    elif opt["mode"] == "inference":
        result_path = os.path.join(opt['checkpoint_path'], "TCA_results")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if opt['part_idx'] >= 0:
            TCA_inference(opt)
        if opt['part_idx'] > 0:
            return
        print("Post processing start")
        TCA_post_processing(opt)
        print("Post processing finished")
        if 'test' not in opt['inference_dataset']:
            # evaluation_proposal(opt)
            if opt["output_detection_result"] != "False":
                evaluation_detection(opt)
    else:
        raise ValueError("mode {} not implated!".format(opt["mode"]))


if __name__ == '__main__':
    from time import sleep

    # print("start sleep!!")
    # sleep(int(60*60*6))
    opt = opts.parse_opt()
    opt = vars(opt)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if opt['part_idx'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt['part_idx'])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    opt_file = open(os.path.join(opt["checkpoint_path"], "opts.json"), "w")
    json.dump(opt, opt_file)
    opt_file.close()
    if not os.path.exists(os.path.join(opt["checkpoint_path"], "TCA_results")):
        os.makedirs(os.path.join(opt["checkpoint_path"], "TCA_results"))
    opt['proposals_result_file'] = os.path.join(opt["checkpoint_path"],
                                                opt['proposals_result_file'].format(opt['inference_dataset']))
    opt['detection_result_file'] = os.path.join(opt["checkpoint_path"],
                                                opt['detection_result_file'].format(opt['inference_dataset']))

    main(opt)



