import math
import sys
import utils
import torch
import torch.distributed as dist

from util.grouping_utils import generate_grid

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, args):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    i = 0
    for images, targets_spp, targets_gt in metric_logger.log_every(data_loader, print_freq, header):
        i+=1
        # metric_logger.update(loss=0.3)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # if i > 10:
        #     break;

        images = list(image.to(device) for image in images)
        targets_spp = [{k: v.to(device) for k, v in t.items()} for t in targets_spp]
        targets_gt = [{k: v.to(device) for k, v in t.items()} for t in targets_gt]

        if args.spp == "grid":
            for tgt in targets_spp:
                boxes_spp = tgt["boxes"]
                H,W = tgt["masks"].shape[-2:]
                boxes, masks = generate_grid(H,W)
                tgt["boxes"] = boxes.to(boxes_spp)
                tgt["masks"] = masks.to(boxes_spp)
                tgt["labels"] = torch.ones(len(boxes), dtype=torch.int64, device=device)

        loss_dict = model(images, targets_spp, targets_gt)
        
        # aout = [None for _ in range(8)]
        # dist.all_gather_object(aout, image_id)
        # image_id = [a for sublist in aout for a in sublist]
        # torch.save(image_id , "image_id.pth") 

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value) or torch.isnan(losses_reduced):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


# def _get_iou_types(model):
#     model_without_ddp = model
#     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
#         model_without_ddp = model.module
#     iou_types = ["bbox"]
#     if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
#         iou_types.append("segm")
#     if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
#         iou_types.append("keypoints")
#     return iou_types


# @torch.no_grad()
# def evaluate(model, data_loader, device, toBinary=False):
#     n_threads = torch.get_num_threads()
#     toBinary = True
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = 'Test:'

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     # iou_types = _get_iou_types(model)
#     iou_types = ["bbox" , "segm"]
#     coco_evaluator = CocoEvaluator(coco, iou_types, toBinary=toBinary)
#     for iou_type in iou_types:
#         coco_evaluator.coco_eval[iou_type].params.maxDets = [300,500,1000]

#     it = 0
#     for images, targets_gt in metric_logger.log_every(data_loader, 100, header):
#         import pdb; pdb.set_trace()
#         it += 1
#         # if it>100:
#         #     break
#         images = list(img.to(device) for img in images)
#         # targets_spp = [{k: v.to(device) for k, v in t.items()} for t in targets_spp]
#         targets_gt = [{k: v.to(device) for k, v in t.items()} for t in targets_gt]

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)
#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         for t in outputs:
#             # t["labels"] = (t["labels"] > 0).long()
#             assert(torch.allclose(t["labels"] , torch.ones_like(t["labels"])))
#         model_time = time.time() - model_time
#         # import pdb; pdb.set_trace()

#         res = {target["image_id"].item(): output for target, output in zip(targets_gt, outputs)}
        
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     print("Averaged stats:", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     stats = coco_evaluator.summarize()
#     print("#"*10)
#     print("Average Recall @100:{:.4f}".format(stats['segm'][8]))
#     print("#"*10)
#     torch.set_num_threads(n_threads)
#     return coco_evaluator
