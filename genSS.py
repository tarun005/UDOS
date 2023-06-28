from datasets.val_dataset_utils import get_coco_test as get_coco
import numpy as np
import selectivesearch as ss
import json, os
import argparse
from PIL import Image
import utils
import transforms
import torch
import tqdm
import torch.distributed as dist


def get_parser():

    parser = argparse.ArgumentParser("Convert coco images to SS labels")
    parser.add_argument("--split", default="train")
    parser.add_argument("--root", default="/datasets01/COCO/022719/")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--subset', default="voc", choices=["voc", "coco", "all"])

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    ## Compute selective search labels for each image in the json file and store as a separate json file.
    args = get_parser()

    utils.init_distributed_mode(args)

    filename = os.path.join(args.root, "annotations", "instances_%s2017.json"%args.split)
    dataset = get_coco(args.root, args.split, transforms.Compose([transforms.ToTensor()]), subset=args.subset)
    # 

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    batch_sampler = torch.utils.data.BatchSampler(sampler, 1, drop_last=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=batch_sampler, num_workers=2,
        collate_fn=utils.collate_fn)

    maskId = 1
    annotations= []

    with open(args.input_file) as fh:
        all_image_ids = [int(ig.strip()) for ig in fh.readlines()]

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for image , target in metric_logger.log_every(data_loader, 100, header):

        img = np.asarray(image[0].permute(1,2,0))
        image_id = target[0]["image_id"].item()

        gt_masks = target[0]["masks"].squeeze(1)

        if image_id not in all_image_ids:
            continue;
        anno_part, maskId = ss.getMasks(img, maskId, image_id, dest="coco_ann")
        annotations.extend(anno_part)

    metric_logger.synchronize_between_processes()
    aout = [None for _ in range(args.world_size)]
    dist.all_gather_object(aout, annotations)
    annotations = [a for sublist in aout for a in sublist]
    if args.gpu == 0:
        print("\nSaving annotations so far ... {}".format(len(annotations)))
        info = {}
        info["annotations"]= annotations
        with open(args.output_file , "w") as fh:
            json.dump(info, fh, indent=4)

# python -m torch.distributed.launch --nproc_per_node=8 --use_env generateSSlabels.py --split train --output_file part_cropped_train2017.json
