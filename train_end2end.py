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
# from models.projection import *
import os
import time
import copy

import torch
import torch.utils.data
# import torchvison
# import torchvision.models.detection
# from models import mask_rcnn
import models

# from coco_utils_joint_load import get_coco
# from lvis_utils_joint_load import get_coco
# from uvo_utils_joint_load import get_coco
from datasets.joint_loader import get_coco, get_lvis, get_uvo
from datasets.val_dataset_utils import get_coco_test, get_lvis_test, get_uvo_test, get_openimages_test, get_ade20k_test, get_cityscapes_test
# from lvis_utils import get_coco

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
# from engine import train_one_epoch
# from maskGeneric import train_one_epoch
from train_one_epoch import train_one_epoch
from test_end2nd import evaluate

import presets
import utils

import warnings
warnings.filterwarnings("ignore")


def get_dataset(name, image_set, transform, data_path, toBinary, subset, spp):
    paths = {
        "coco": (data_path, get_coco, 2),
        "uvo" : (data_path, get_uvo, 2),
        "lvis": (data_path, get_lvis, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, toBinary=toBinary, subset=subset, spp=spp)
    return ds

def get_dataset_test(name, image_set, transform, data_path, toBinary, subset):
    paths = {
        "coco": (data_path, get_coco_test, 2),
        "lvis": (data_path, get_coco_test, 2),
        "uvo": (data_path, get_uvo_test, 2),
        "openimages" : (data_path, get_openimages_test, 2),
        "ade20k" : (data_path, get_ade20k_test, 2),
        "cityscapes" : (data_path, get_cityscapes_test, 2)
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
    parser.add_argument('--dataset', default='coco', choices=["uvo","coco","openimages","ade20k"], help='dataset')
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
    parser.add_argument('--print-freq', default=500, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--data-augmentation', default="hflip", 
                                            help='data augmentation policy (default: hflip)')
    parser.add_argument('--num-classes', type=int, default=2, 
                                            help='Number of classes. 2 is default for class-agnostic segmentation.')
    parser.add_argument('--data-split-train', type=str, default="all", help='Which COCO split to use.',
                                                choices=["all", "voc", "coco"])
    parser.add_argument('--data-split-test', type=str, default="all", help='Which COCO split to use.',
                                                choices=["all", "voc", "coco"])
    parser.add_argument('--delta', type=int, default=15)
    # parser.add_argument('--thres', type=float, default=0.5)
    # parser.add_argument('--load_model', default='')
    # parser.add_argument('--test_niter', type=int)
    parser.add_argument('--lambda_2' , type=float)
    parser.add_argument('--lambda_3' , type=float)
    parser.add_argument('--first_stage_scoring',type=int,default=1,choices=[0,1])
    parser.add_argument('--second_stage_scoring',type=int,default=1,choices=[0,1])
    parser.add_argument('--spp', default="ss", choices=["ss","grid","mcg","ssn"])
    # parser.add_argument('--pos', default=None, choices=["sine","grid","none"])
    # parser.add_argument('--maskrcnn', default="false", choices=["true", "false"])
    parser.add_argument('--detections', default=300, type=int)
    parser.add_argument('--iou_overlap_thres', type=float, default=0.5)

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


def train(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    if not args.debug:
        utils.init_distributed_mode(args)
    else:
        args.distributed = 0
    print(args)

    device = torch.device(args.device)

    # args.maskrcnn = (args.maskrcnn == "true")

    if args.dataset == "coco":
        args.shared = False
    else:
        args.shared = True

    print("Creating model")
    kwargs = {
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "lambda_l2": args.lambda_2,
        "lambda_l3": args.lambda_3,
        # "n_iter_test": args.test_niter,
        "delta": args.delta,
        "first_stage_scoring": args.first_stage_scoring,
        "second_stage_scoring": args.second_stage_scoring,
        # "pos_encoding": args.pos,
        # "baseline": args.maskrcnn,
        "shared" : args.shared,
        "iou_overlap" : args.iou_overlap_thres
    }

    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    
    # kwargs["box_nms_thresh"] = 0.5
    kwargs["box_detections_per_img"] = args.detections
    kwargs["box_score_thresh"] = 0.0
    # kwargs["rpn_pre_nms_top_n_test"] = 2000
    # kwargs["rpn_post_nms_top_n_test"] = 2000
    

    model = models.__dict__[args.model](num_classes=args.num_classes, pretrained=args.pretrained, **kwargs)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        optimizer_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # Data loading code
    
    print("Loading data")

    if 0:#args.debug:
        image_set = "val"
    else:
        image_set = "train"

    if not args.test_only:
        dataset = get_dataset(args.dataset, image_set, get_transform(True, args.data_augmentation),
                                        args.data_path, args.toBinary, subset=args.data_split_train, spp=args.spp)
        
        if args.toBinary:
            assert args.num_classes == 2

        print("Creating data loaders")
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)

        if args.aspect_ratio_group_factor >= 0:
            group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=utils.collate_fn)
        

    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        args.start_epoch = checkpoint["epoch"] + 1


    if args.load_model:
        checkpoint = torch.load(args.load_model, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        print("Resuming from pretrained model ...")

    if args.test_only:
        test(args, model)
        return 0

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, args)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch
            }

            if 1:#epoch in [4,6,8]:
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_dir, 'checkpoint.pth'))

        # if epoch > 4 or epoch == args.epochs-1:
        #     test(args, model)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    test(args, model)

    print('Training time {}'.format(total_time_str))

def test(args, model):

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test  = get_dataset_test(args.dataset, "val", presets.DetectionPresetEval(),
                                       args.data_path, args.toBinary, subset=args.data_split_test)
    
    if args.toBinary:
        assert args.num_classes == 2

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    evaluate(model, data_loader_test, device=device, args=args, toBinary=args.toBinary)
    return 0

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    train(args)
    # if args.test_only:
    #     assert args.resume is not None
    #     test(args)
    # else:
    #     train(args)
