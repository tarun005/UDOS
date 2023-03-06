r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3

Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
from models.projection import *
import os
import time
import copy

import torch
import torch.utils.data
# import torchvison
# import torchvision.models.detection
# from models import mask_rcnn
import models

from coco_utils import get_coco_ss,get_coco_ssn, get_coco_kp, get_coco
# from coco_utils_joint_load import get_coco

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# from engine import train_one_epoch
# from maskGeneric import train_one_epoch
from maskEnd2End import train_one_epoch
from genericEval import evaluate

import presets
import utils

import warnings
warnings.filterwarnings("ignore")


def get_dataset(name, image_set, transform, data_path, toBinary, subset):
    paths = {
        "coco": (data_path, get_coco, 2),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, toBinary=toBinary, subset=subset)
    return ds


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=None, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument('--num-classes', type=int, default=91, help='Number of classes. 91 for CoCo, 2 for CAS')
    parser.add_argument('--data-split-train', type=str, default="all", help='Which COCO split to use.',
                                                choices=["all", "voc", "coco"])
    parser.add_argument('--data-split-test', type=str, default="all", help='Which COCO split to use.',
                                                choices=["all", "voc", "coco"])
    parser.add_argument('--sim_mode', type=str, help='sim mode in group loss',
                                                choices=["binary_concat" , "binary_sum" , "contrastive" , "prototype"])
    parser.add_argument('--thres', required=True, type=float)
    parser.add_argument('--proj_layer', required=True, type=str, choices=["conv" , "mlp"])
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--glevels', type=int, default=0)

    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--toBinary",
        dest="toBinary",
        help="convert labels to binary for class agnostic segmentation",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    if not args.debug:
        utils.init_distributed_mode(args)
    else:
        args.distributed = 0
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset = get_dataset(args.dataset, "train", get_transform(True, args.data_augmentation),
                                       args.data_path, args.toBinary, subset=args.data_split_train)
    dataset_test  = get_dataset(args.dataset, "val", get_transform(False, args.data_augmentation), 
                                       args.data_path, args.toBinary, subset=args.data_split_test)
    
    if args.toBinary:
        assert args.num_classes == 2

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    # if args.toBinary and args.test_only:
    if args.test_only:
        # kwargs["box_nms_thresh"] = 0.75
        # kwargs["box_score_thresh"] = 0.
        kwargs["box_detections_per_img"] = 100#1000
    else:
        kwargs["box_detections_per_img"] = 700
        kwargs["rpn_pre_nms_top_n_train"] = 4000
        kwargs["rpn_post_nms_top_n_train"] = 4000

    model = models.__dict__[args.model](num_classes=args.num_classes, pretrained=args.pretrained, **kwargs)
    model.to(device)

    # if args.resume:
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     print("Resuming from checkpoint ...")

    try:
        backbone = copy.deepcopy(model.module.backbone)
        roi_pool = model.module.roi_heads.mask_roi_pool
    except:
        backbone = copy.deepcopy(model.backbone)
        roi_pool = model.roi_heads.mask_roi_pool

    # if args.sim_mode == "binary_concat":
    #     projection = ProjectionLinear(output_dim*2)
    #     args.proj_layer = "mlp"
    # elif args.sim_mode == "binary_sum":
    #     # projection = ProjectionLinear(output_dim)
    #     projection = MaskEmbeddings(backbone, roiAlign=roi_pool, pos_encoding=True)
    #     args.proj_layer = "mlp"
    # elif args.sim_mode == "contrastive" or args.sim_mode == "prototype":
    #     if args.proj_layer == "conv":
    #         # projection = ProjectionConv(256,[256,256])
    #         projection = MaskEmbeddings(backbone, roiAlign=roi_pool)
    #     elif args.proj_layer == "mlp":
    #         # projection = ProjectionLinearEmbedding(output_dim)
    #         projection = MaskEmbeddings()
    #     else:
    #         raise NotImplementedError
    # else:
    #     raise NotImplementedError
    projection = MaskEmbeddings(backbone, roiAlign=roi_pool)

    projection.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    projection_without_ddp = projection
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

        projection = torch.nn.parallel.DistributedDataParallel(projection, device_ids=[args.gpu])
        projection_without_ddp = projection.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    projection_params = [p for p in projection.parameters() if p.requires_grad]
    projection_optimizer = torch.optim.SGD(
        projection_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # projection_optimizer = torch.optim.Adam(
    #     projection_params, lr=args.lr, weight_decay=args.weight_decay)
    

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(projection_optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(projection_optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))
    
    if os.path.exists(os.path.join(args.output_dir , "checkpoint.pth")):
        args.resume = os.path.join(args.output_dir , "checkpoint.pth")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resuming from checkpoint ...")
        optimizer.load_state_dict(checkpoint['optimizer'])
        if "projection" in checkpoint:
            print("Loading projection networks")
            projection_without_ddp.load_state_dict(checkpoint["projection"], strict=True)
            projection_optimizer.load_state_dict(checkpoint['projection_optimizer'])
            args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model,projection, data_loader_test, device=device, toBinary=args.toBinary, thres=args.thres, glevels=args.glevels)
        return 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, projection, projection_optimizer, data_loader, device, epoch, args.print_freq, args)
        # train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),

                'projection': projection_without_ddp.state_dict(),
                'projection_optimizer': projection_optimizer.state_dict(),
                'args': args,
                'epoch': epoch
            }
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        # if epoch % 3 == 0 or epoch == args.epochs-1:
        #     # evaluate after every epoch
        #     evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
