# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn, Tensor
import warnings
from typing import Tuple, List, Dict, Optional, Union
from .grouping import merge_op
from util.grouping_utils import remove_overlaps

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Args:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform, roi_heads_stage2=None):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads ## On Superpixels
        # self.projection = projection
        if roi_heads_stage2:
            self.roi_heads_stage2 = roi_heads_stage2 ## On original annotations
        else:
            self.roi_heads_stage2 = None
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    # def forward(self, images, targets_spp=None, targets_gt=None):
            
    #     outputs = self.forward(images, targets_spp, targets_gt)
    #     return outputs

    def forward(self, images, targets_spp=None, targets_gt=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """

        if self.training and (targets_gt is None or targets_spp is None):
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets_gt is not None and targets_spp is not None

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images_copy = tuple(images)
        images, targets_spp = self.transform(images, targets_spp)
        images_copy, targets_gt = self.transform(images_copy, targets_gt)
        if targets_spp is not None:
            self.check_targets(targets_spp)
            self.check_targets(targets_gt)

        if targets_spp is not None:
            targets_spp = remove_overlaps(targets_spp, targets_gt, self.iou_overlap)

        ## Add full and part masks to gt
        if targets_spp is not None:
            targets_spp_concat = []
            for image_id in range(len(targets_gt)):
                concatenated_gt = {}
                concatenated_gt["masks"] = torch.cat([targets_spp[image_id]["masks"] , targets_gt[image_id]["masks"]])
                concatenated_gt["boxes"] = torch.cat([targets_spp[image_id]["boxes"] , targets_gt[image_id]["boxes"]])
                concatenated_gt["labels"] = torch.cat([targets_spp[image_id]["labels"] , targets_gt[image_id]["labels"]])
                targets_spp_concat.append(concatenated_gt)
        else:
            targets_spp_concat = None

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets_spp_concat)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets_spp_concat,stage1=True)

        # Resize masks to transformed image size.
        detections = self.transform.postprocess_partial(detections, images.image_sizes)
        boxes = [d["boxes"].detach().cuda() for d in detections]

        pairwise_similarities = self.projection(features, boxes, images.image_sizes).detach()

        if self.roi_heads.second_stage_scoring or self.roi_heads.first_stage_scoring:
            class_loss=False
        else:
            class_loss=True

        ## extract similarities within each sample in a batch.
        ## TODO: extend for bsz>2
        ps = [pairwise_similarities[:len(boxes[0]),:len(boxes[0])] ,pairwise_similarities[len(boxes[0]):,len(boxes[0]):] ]

        losses = {}
        if self.training:
            merged_boxes = []
            for idx, bx in enumerate(boxes):
                merged_boxes = merge_op(ps[idx], bx, masks=None, scores=None, aff_thres=0.5)[0]
                boxes[idx] = torch.cat([boxes[idx],merged_boxes.detach()]) 
            
            if self.roi_heads_stage2:
                _, detector_losses_s2 = self.roi_heads_stage2(features, boxes, images.image_sizes, targets_gt, class_logits_loss=class_loss)
            else:
                _, detector_losses_s2 = self.roi_heads(features, boxes, images.image_sizes, targets_gt, class_logits_loss=class_loss)

            losses.update(detector_losses)
            losses.update(proposal_losses)
            
            detector_losses_s2 = {k+"_stage2":v*self.lambda_l3 for k,v in detector_losses_s2.items()}
            losses.update(detector_losses_s2)
        else:
            union = []
            boxes = torch.cat(boxes, dim=0)
            union.append(boxes)

            assert len(detections) == 1 , "Test bsz needs to be 1."
            
            # for _ in range(self.niter_test)
            with torch.no_grad():
                boxes = merge_op(pairwise_similarities, boxes, masks=None, scores=None, aff_thres=0.5)[0]
                pairwise_similarities = self.projection(features, [boxes], images.image_sizes).detach()

                ## Also collect the aggregate of all boxes
                union.append(boxes)
            if self.roi_heads_stage2:
                detections, _ = self.roi_heads_stage2(features, [torch.cat(union, dim=0)], images.image_sizes, class_logits_loss=class_loss, nms=False)
            else:
                detections, _ = self.roi_heads(features, [torch.cat(union, dim=0)], images.image_sizes, class_logits_loss=class_loss, nms=False)

            detections = [{k: v.cpu() for k, v in t.items()} for t in detections]
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        ##########################################
        ##########################################

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    def check_targets(self, targets):
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError("Expected target boxes to be a tensor"
                                        "of shape [N, 4], got {:}.".format(
                                            boxes.shape))
            else:
                raise ValueError("Expected target boxes to be of type "
                                    "Tensor, got {:}.".format(type(boxes)))

        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width."
                                    " Found invalid box {} for target at index {}."
                                    .format(degen_bb, target_idx))


    # def forward_baseline(self, images, targets_gt=None):
    #     # type: (List[Tensor], Optional[List[Dict[str, Tensor]]], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    #     """
    #     Args:
    #         images (list[Tensor]): images to be processed
    #         targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    #     Returns:
    #         result (list[BoxList] or dict[Tensor]): the output from the model.
    #             During training, it returns a dict[Tensor] which contains the losses.
    #             During testing, it returns list[BoxList] contains additional fields
    #             like `scores`, `labels` and `mask` (for Mask R-CNN models).
    #     """

    #     if self.training and (targets_gt is None):
    #         raise ValueError("In training mode, targets should be passed")
    #     if self.training:
    #         assert targets_gt is not None

    #     original_image_sizes: List[Tuple[int, int]] = []
    #     for img in images:
    #         val = img.shape[-2:]
    #         assert len(val) == 2
    #         original_image_sizes.append((val[0], val[1]))

    #     images = tuple(images)
    #     images, targets_gt = self.transform(images, targets_gt)

    #     features = self.backbone(images.tensors)
    #     if isinstance(features, torch.Tensor):
    #         features = OrderedDict([('0', features)])
    #     proposals, proposal_losses = self.rpn(images, features, targets_gt)
    #     if self.roi_heads.second_stage_scoring or self.roi_heads.first_stage_scoring:
    #         class_loss = False
    #     else:
    #         class_loss = True

    #     detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets_gt, class_logits_loss=class_loss)

    #     if not self.training:
    #         detections = [{k: v.cpu() for k, v in t.items()} for t in detections]
    #         detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
    #     ##########################################
    #     ##########################################
    #     losses = {}
    #     losses.update(detector_losses)
    #     losses.update(proposal_losses)

    #     if torch.jit.is_scripting():
    #         if not self._has_warned:
    #             warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
    #             self._has_warned = True
    #         return losses, detections
    #     else:
    #         return self.eager_outputs(losses, detections)
