import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--mode',
        type=str,
        default='train')

    parser.add_argument(
        '--inference_dataset',
        type=str,
        default='validation')

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/mnt/data/FlylineData/flq/model/train_snap/hacs/hacs.tca.debug/')
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5)

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4)

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=9)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16)
    parser.add_argument(
        '--step_size',
        type=int,
        default=7)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    # Overall Dataset settings
    parser.add_argument(
        '--video_anno',
        type=str,
        default="/mnt/data/FlylineData/flq/dataset/HACS/HACS-dataset/HACS_v1.1.1/HACS_segments_v1.1.1.json")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=1000)
    parser.add_argument(
        '--window_size',
        type=int,
        default=9)

    parser.add_argument(
        '--frames_fps',
        type=float,
        default=15.0)

    parser.add_argument(
        '--feature_path',
        type=str,
        default="/home/flq/dataset/HACS/features/slowfast101.epoch9.87.52.finetune.pool.t.keep.t.s8/")

    parser.add_argument(
        '--proposals_path',
        type=str,
        default="/mnt/data/FlylineData/flq/model/train_snap/hacs/hacs.bmn.pem.slowfast101.t200.wd1e-5.warmup/pem_input_100/")

    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2304)

    parser.add_argument(
        '--lgte_num',
        type=int,
        default=2)

    parser.add_argument(
        '--use_bn',
        type=str,
        default="False")

    # pem rcnn settings
    parser.add_argument(
        '--action_sample_num',
        type=int,
        default=64)

    parser.add_argument(
        '--start_sample_num',
        type=int,
        default=32)

    parser.add_argument(
        '--end_sample_num',
        type=int,
        default=32)

    parser.add_argument(
        '--temporal_interval',
        type=float,
        default=8./15.)

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=16)
    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4)
    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.0)
    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.0)

    parser.add_argument(
        '--classifier_result',
        type=str,
        default="/mnt/data/FlylineData/flq/result/hacs.classifier/{}94.32.json",
    )

    parser.add_argument(
        '--output_detection_result',
        type=str,
        default="False")

    parser.add_argument(
        '--proposals_result_file',
        type=str,
        default="{}_result_proposals.json")

    parser.add_argument(
        '--detection_result_file',
        type=str,
        default="{}_result_detections.json")

    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="evaluation_result.jpg")

    parser.add_argument(
        '--assign_class',
        type=str,
        default="")

    parser.add_argument(
        '--gpu',
        type=str,
        default="0,1,2,3")

    parser.add_argument(
        '--part_idx',
        type=int,
        default=-1)

    parser.add_argument(
        '--continue_epoch',
        type=int,
        default=-1)

    args = parser.parse_args()
    print(args)

    return args

