import torch.nn as nn

import torch

from util.grouping_utils import *
from sklearn.cluster import AgglomerativeClustering

__all__ = ["ProjectionLayer"]

torch.autograd.set_detect_anomaly(True)


def merge_op(pairwise_similarities, boxes, masks=None, scores=None, aff_thres=0.2, use_knn=True):
    """
    Args:
        Same as above
    """

    pairwise_similarities = pairwise_similarities.detach()

    assert len(boxes) == len(pairwise_similarities)

    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=(1-aff_thres))

    indices = pairwise_similarities.float()

    labels = clustering.fit((1-indices).cpu().numpy()).labels_
    groups = clusters(torch.Tensor(labels))
    merged_boxes = torch.stack([mergeBoxes(boxes[inds]) for inds in groups], dim=0)
    if masks is not None:
        merged_masks = torch.stack([masks[inds].any(0) for inds in groups], dim=0) ## Merge binary
    else:
        merged_masks = None
    if scores is not None:
        merged_scores = torch.stack([scores[inds].mean() for inds in groups])
    else:
        merged_scores = None

    return merged_boxes, merged_masks, merged_scores

class GroupingLayer(nn.Module):

    def __init__(self, roiAlign=None, delta=15):

        super().__init__()

        self.roiAlign = roiAlign
        self.delta=delta

    def forward(self, features, boxes, sizes):

        if not isinstance(features, dict):
            features = {"0":features}
        
        boxes = [b.detach() for b in boxes]

        ## Box align features
        boxes = [add_context(b, self.delta) for b in boxes]
        box_features = self.roiAlign(features, boxes, sizes).mean(-1).mean(-1)

        projection_features = torch.nn.functional.normalize(box_features, dim=1, p=2)
        batch_similarities = torch.matmul(projection_features , projection_features.T.clone().detach())
        
        return batch_similarities