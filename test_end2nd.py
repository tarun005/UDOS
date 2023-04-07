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
import torch
import torch.utils.data
# import torchvison
# import torchvision.models.detection
import models

from datasets.val_dataset_utils import get_coco_test, get_lvis_test, get_uvo_test

import presets
import time

import warnings
warnings.filterwarnings("ignore")

from datasets.val_dataset_utils import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
import utils

from util.grouping_utils import generate_grid

def get_dataset(name, image_set, transform, data_path, toBinary, subset):
    paths = {
        "coco": (data_path, get_coco_test, 2),
        "lvis": (data_path, get_lvis_test, 2),
        "uvo": (data_path, get_uvo_test, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, toBinary=toBinary, subset=subset)
    return ds

def get_transform():
    return presets.DetectionPresetEval()


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--rpn-score-thresh', default=None, type=float, help='rpn score threshold for faster-rcnn')
    parser.add_argument('--num-classes', type=int, default=91, help='Number of classes. 91 for CoCo, 2 for CAS')
    parser.add_argument('--data-split-test', type=str, default="all", help='Which COCO split to use.',
                                                choices=["all", "voc", "coco" , "rare" , "common"])
    parser.add_argument('--niter', type=int)
    parser.add_argument('--first_stage_scoring',type=int,default=1,choices=[0,1])
    parser.add_argument('--second_stage_scoring',type=int,default=1,choices=[0,1])

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
    if not args.debug:
        utils.init_distributed_mode(args)
    else:
        args.distributed = 0
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset_test  = get_dataset(args.dataset, "val", get_transform(), 
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

    print("Creating model")
    kwargs = {
        "first_stage_scoring": args.first_stage_scoring,
        "second_stage_scoring": args.second_stage_scoring,
    }
    if "rcnn" in args.model:
        if args.rpn_score_thresh is not None:
            kwargs["rpn_score_thresh"] = args.rpn_score_thresh
    
    # if args.toBinary and args.test_only:
    # kwargs["box_nms_thresh"] = 0.75
    kwargs["box_score_thresh"] = 0.
    kwargs["box_detections_per_img"] = 1000
    kwargs["n_iter"] = args.niter

    model = models.__dict__[args.model](num_classes=args.num_classes, pretrained=False, **kwargs)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # for k,v in checkpoint["model"].items(): 
        #     if k in model_without_ddp.state_dict():
        #         pass
        #     else:
        #         print("Not loading", k)
        print("Resuming from checkpoint ...")

    evaluate(model, data_loader_test, device=device, args=args, toBinary=args.toBinary)
    return 0

@torch.no_grad()
def evaluate(model, data_loader, device, args, toBinary=False):
    n_threads = torch.get_num_threads()
    toBinary = True
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = ["bbox" , "segm"]

    split=args.data_split_test
    if args.dataset == "coco":
        from VOCtoCOCO import mapping
        if split == "voc":
            indices = mapping._common_with_voc
        elif split == "coco":
            indices = mapping._exclusive_to_coco
        else:
            indices = mapping._all # Or all
    else:
        indices = [1]

    # coco = get_coco_api_from_dataset(data_loader.dataset.dataset.det_gt, set=args.dataset)
    coco = get_coco_api_from_dataset(data_loader.dataset, set=args.dataset, indices=indices)
    # iou_types = _get_iou_types(model)
        
    coco_evaluator = CocoEvaluator(coco, iou_types, toBinary=toBinary)
    for iou_type in iou_types:
        coco_evaluator.coco_eval[iou_type].params.maxDets = [10,50,100,300,500]

    it = 0
    for images, targets_gt in metric_logger.log_every(data_loader, 100, header):
        it += 1
        # 
        # print(images[0]["image_id"])
        # if targets_gt[0]["image_id"] != 139:
        #     continue
        # if it>200:
        #     break
        images = list(img.to(device) for img in images)
        # targets_spp = [{k: v.to(device) for k, v in t.items()} for t in targets_spp]
        targets_gt = [{k: v.to(device) for k, v in t.items()} for t in targets_gt]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        # for tsp in targets_spp:
        #     tsp["scores"] = torch.ones(len(tsp["boxes"]))
        #     tsp["masks"] = tsp["masks"][:,None]
        # outputs = targets_spp
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for t in outputs:
            # t["labels"] = (t["labels"] > 0).long()
            assert(torch.allclose(t["labels"] , torch.ones_like(t["labels"])))
            assert not t["scores"].mean().isnan()
        model_time = time.time() - model_time
        # import pdb; pdb.set_trace()
        # score = ss_cue(outputs[0]["boxes"] , targets_spp[0]["masks"]).view(-1).clamp(min=0)
        # outputs[0]["scores"] = score.view(-1)

        res = {target["image_id"].item(): output for target, output in zip(targets_gt, outputs)}
        
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()
    print("#"*10)
    # print("Average Recall @100:{:.4f}".format(stats['segm'][2]))
    print("#"*10)
    torch.set_num_threads(n_threads)
    return coco_evaluator


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
