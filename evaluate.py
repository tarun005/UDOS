import torch
import torch.utils.data
import time

import warnings
warnings.filterwarnings("ignore")

from datasets.val_dataset_utils import get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
import utils

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
        indices = [1] ## class agnostic evaluation. all labels are 1.

    coco = get_coco_api_from_dataset(data_loader.dataset, set=args.dataset, indices=indices)
        
    coco_evaluator = CocoEvaluator(coco, iou_types, toBinary=toBinary)
    for iou_type in iou_types:
        coco_evaluator.coco_eval[iou_type].params.maxDets = [10,50,100,300,500]

    for images, targets_gt in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets_gt = [{k: v.to(device) for k, v in t.items()} for t in targets_gt]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        for t in outputs:
            assert(torch.allclose(t["labels"] , torch.ones_like(t["labels"])))
            assert not t["scores"].mean().isnan()
        model_time = time.time() - model_time

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
    torch.set_num_threads(n_threads)
    return coco_evaluator
