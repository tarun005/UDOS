import torch
from util.box_ops import box_iou
from torchvision.ops.boxes import box_area
import numpy as np
from copy import deepcopy

def remove_overlaps(U, S, thres=0.9):
    """
        U is superpixel masks
        S is annotated masks
    """

    new_U = []
    for u_i, s_i in zip(U, S):
        u_masks = u_i["masks"][:,None]
        s_masks = s_i["masks"]

        pred_masks_split = u_masks.split(3,dim=0) ## Tune this according to your GPU mem
        iou = []
        for pms in pred_masks_split:
            iou_part = mask_iou(pms,s_masks)[0]
            iou.append(iou_part)
            
        iou_ = torch.cat(iou, dim=0)

        valid_inds = iou_.max(1).values < thres

        if valid_inds.sum() == 0:
            new_U_i = s_i
        else:
            new_U_i = deepcopy(u_i)
            for k in ["boxes", "labels", "masks"]:
                new_U_i[k] = u_i[k][valid_inds]

        new_U.append(new_U_i)
    
    return new_U 

def generate_grid(H,W,grid_size=8):

    """
        Given an image, generate superpixels as 
        regular sized grids.
    """

    xboxes = np.array(np.arange(0,W+1,(W)//grid_size))
    yboxes = np.array(np.arange(0,H+1,H//grid_size))

    toplefts_x = xboxes[:8].reshape(1,-1).repeat(grid_size,axis=0).reshape(-1,1).clip(max=W-1,min=0.)
    toplefts_y = yboxes[:8].reshape(-1,1).repeat(grid_size,axis=1).reshape(-1,1).clip(max=H-1,min=0.)
    bottomright_x = xboxes[-8:].reshape(1,-1).repeat(grid_size,axis=0).reshape(-1,1).clip(max=W-1,min=0.)
    bottomright_y = yboxes[-8:].reshape(-1,1).repeat(grid_size,axis=1).reshape(-1,1).clip(max=H-1,min=0.)

    boxes = torch.Tensor(np.hstack([toplefts_x, toplefts_y, bottomright_x, bottomright_y])).long()

    masks = []
    for xt,yt,xb,yb in boxes:
        m = torch.zeros(H,W)
        m[yt:yb,xt:xb] = 1.0
        masks.append(m)

    masks = torch.stack(masks)

    return boxes, masks


def get_box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = (inter+1e-6) / (union+1e-6)

    return iou


def clusters(indices):
    """Get the members in each cluster as a list of lists."""

    ids = indices.unique()
    clusters = []
    for ii in ids:
        members = [e for e,i in enumerate(indices) if i==ii]
        clusters.append(members)
    return clusters

def mergeBoxes(box_list):
    """
    Merge all input boxes to form a bigger bounding box.
    Input: (N,4)
    Output: (4,)
    """
    x_min = box_list[:,0].min()
    y_min = box_list[:,1].min()
    
    x_max = box_list[:,2].max()
    y_max = box_list[:,3].max()
    
    return torch.Tensor([x_min,y_min, x_max, y_max]).to(box_list.device)

def get_centroids(boxes, masks=None, type="boxes"):
    """
    Return centroids of each proposal in terms of boxes or masks.
    Args:
        boxes (torch.Tensor): [N,4]
        masks (torch.Tensor): [N,1,H,W]
    """

    if type == "boxes":
        centers_x = torch.mean(boxes[:,[0,2]], dim=1)
        centers_y = torch.mean(boxes[:,[1,3]], dim=1)
    elif type == "masks":
        ### TODO: parallelize the computation 
        assert masks is not None
        centers_x = []
        centers_y = []
        if masks.ndim == 4:
            masks = torch.Tensor(masks[:,0])
        for m in masks:
            inds = torch.nonzero(m)
            cx,cy = inds.mean(0)
            centers_x.append(cx)
            centers_y.append(cy)
        centers_x , centers_y = torch.cat(centers_x) , torch.cat(centers_y)
    
    centers = torch.stack([centers_x, centers_y], dim=1).view(-1,2)
    return centers

def load_from_checkpoint(path, model):
    checkpoint = torch.load(path, map_location='cpu')

    mdl = checkpoint["state_dict"]
    mdl = {k.partition("module.encoder_q.")[-1]:v for k,v in mdl.items()}
    mdl_state_dict = model.state_dict()
    for k,v in mdl.items():
        if k in mdl_state_dict:
            mdl_state_dict[k].copy_(v)
    return model

def mask_diagonal(simMatrix):

    inds = torch.arange(len(simMatrix))
    inv_z = torch.ones_like(simMatrix)
    inv_z[inds,inds] = 0
    return simMatrix*inv_z

def add_context(boxes, delta=5):

    assert 0 <= delta <= 100
    width = boxes[:,2] - boxes[:,0]
    height = boxes[:,3] - boxes[:,1]
    boxes_max_ = boxes.max(0)[0].view(1,-1)

    width , height = width*((delta/2)/100) , height*((delta/2)/100)
    add_strip = torch.stack([-1*width, -1*height, width, height], dim=1)
    boxes += add_strip
    boxes = torch.min(boxes.clamp(min=0), boxes_max_)

    return boxes

def nondiagonal_mean(tensor):

    if tensor.numel() == 1:
        return tensor.mean()
    
    assert tensor.ndim == 2
    x,y = torch.triu_indices(*tensor.shape , offset=1)
    off_diagonal_ele = tensor[x,y]
    return off_diagonal_ele.mean()

def mask_area(masks):
    """
    Compute area of binary masks
    Args:
        masks: (N,1,H,W)
    """
    masks = masks>0.5
    area = masks.sum(-1).sum(-1).squeeze(-1)
    return area

def mask_iou(pred, gt, iou=False):
    """
    Compute intersection area between two masks.
    """
    with torch.no_grad():
        mask1 = (pred>0.5).detach()
        mask2 = (gt>0.5).detach()
        intersection = torch.logical_and(mask1, mask2).sum(-1).sum(-1) 

        union = torch.logical_or(mask1, mask2).sum(-1).sum(-1)
        iou = (intersection+1e-6) / (union+1e-6)

    return iou, union, intersection

def remove_diagonal(tensor):
    assert tensor.ndim == 2 ## 2D matrix
    assert tensor.shape[0] == tensor.shape[1] ## square matrix
    n = tensor.shape[0]
    return tensor.flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)


def get_sim_indices(pred, gt, mode, thres, type="masks", return_weights=False):
    """
    For each predicted box, assign it to one gt box if more than thres% of the predicted box lies inside the gt box. 
    If a predbox overlaps more than 90% with two or more gt boxes, assign it to highest overlap.
    If it overlaps with none, then do not assign it to any box.

    Args:
        Mode: Can be binary, contrastive, prototype
        type: can be masks or boxes
    """


    ## For each prediction, compute its affinity to each ground truth box.
    ## affinity is the percentage of pred box that lies inside the gt box.

    if type == "masks":
        area = mask_area
        iou = mask_iou
    elif type == "boxes":
        area = box_area
        iou = box_iou

    ## Some of these could be zero, so do an early check
    assert len(gt) 
    assert len(pred)

    # area_of_pred = area(pred).view(-1,1) ## (P,)

    if type == "boxes":
        _ ,_ ,inter_area_pred_gt = iou(pred, gt) ## (P,G)
        area_of_pred = area(pred).view(-1,1) ## (P,1)
    elif type == "masks": ## Requires special care to handle memory explosion
        pred_masks_split = pred.split(3,dim=0) ## Tune this according to your GPU mem
        inter_area_pred_gt = []
        area_of_pred = []
        for pms in pred_masks_split:
            _,_, inter_area_pred_gt_part = iou(pms,gt)
            inter_area_pred_gt.append(inter_area_pred_gt_part)
            area_of_pred.append(area(pms).view(-1,1))
        inter_area_pred_gt = torch.cat(inter_area_pred_gt, dim=0)
        area_of_pred = torch.cat(area_of_pred, dim=0)

    overlap = inter_area_pred_gt/area_of_pred ## (P,G)
    instance_affs, instance_inds = overlap.max(axis=1) ## 0 for BG box
    instance_inds += 1

    bg_inds = instance_affs < thres ## If %overlap is less than t%, then it is probably a BG box.
    instance_inds[bg_inds] = 0 ## Assign 0 label to all BG preds.
    object_inds = torch.logical_not(bg_inds) ## not bg = object

    onehot_affinities = torch.eye(torch.max(instance_inds)+1)[instance_inds].to(pred.device) ## Row onehot (P,Nc)
    if mode == "prototype":
        return onehot_affinities, object_inds

    row_matrix = onehot_affinities.unsqueeze(-1).repeat(1,1,len(onehot_affinities)) ## (P,Nc,P)
    col_matrix = row_matrix.permute(2,1,0) ## (P,Nc,P)
    affinity_matrix = (row_matrix*col_matrix).sum(1) ## (P,P)

    if "binary" in mode:
        affinity_matrix = remove_diagonal(affinity_matrix) ## diagonals are always zero. do not count them in loss
        affinity_matrix = affinity_matrix[object_inds]#[:,object_inds]
        areas = mask_area(gt).view(-1,1).float()
        H,W = pred.shape[-2:]
        total_area = H*W
        bg_area = (total_area - areas.sum()).view(-1,1)
        areas = torch.cat([bg_area,areas], dim=0)
        spp_areas = areas[instance_inds]
        areas_product = remove_diagonal(torch.matmul(spp_areas, spp_areas.T))
        areas_product = areas_product[object_inds]
        if areas_product.numel() == 0: 
            weights = torch.zeros(1)
        elif areas_product.min() < 10:
            weights = torch.ones_like(affinity_matrix)
        else:
            areas_weight = 1./(areas_product)
            weights = areas_weight/areas_weight.sum()
        # weights = torch.ones_like(affinity_matrix)
        # save_dict = {
        #     "instance_inds" : instance_inds.cpu(),
        #     "image" : image.cpu().permute(1,2,0).numpy(),
        #     "pred" : pred.cpu().squeeze(1),
        #     "gt" : gt.cpu(),
        #     "labels" : affinity_matrix.cpu(),
        #     "obj_inds" : object_inds.cpu()
        # }
        # torch.save(save_dict , "save_dict.pth")
        # pdb.set_trace()
        if weights.mean().isnan():
            import pdb; pdb.set_trace()
        return affinity_matrix.view(-1,1), object_inds, weights.view(-1,1) ## (P',P) (P,)

    return affinity_matrix, object_inds ## (P,P)


def ss_cue(boxes, masks):
    """
    Take boxes (proposals) and superpixels from an image and predict "objectness" scores for each box (proposal).
    Args:
        boxes : (N,4)
        masks : (M,H,W)
    Output:
        ss_cue: from https://thomas.deselaers.de/publications/papers/alexe-cvpr10.pdf
    """
    if masks.ndim == 4:
        assert masks.shape[1] == 1
        masks = masks.squeeze(1)
    
    masks = masks.detach()
    boxes = boxes.detach()

    box_areas = box_area(boxes).view(1,-1) + 1e-4
    boxes = torch.clamp(boxes-1,min=0)
    masks = (masks>0.5).float()
    mask_areas = masks.sum(-1).sum(-1).view(-1,1)
    integral_images = masks.cumsum(1).cumsum(2)
    boxes = boxes.long()
    x1,y1,x2,y2 = boxes.split(1,dim=1)

    top_left = integral_images[:,y1,x1]
    top_right = integral_images[:,y1,x2]
    bottom_left = integral_images[:,y2,x1]
    bottom_right = integral_images[:,y2,x2]
    areas_inside_box = (top_left + bottom_right - top_right - bottom_left).squeeze(-1) # M,N
    areas_outside_box = mask_areas - areas_inside_box
    min_of_two_areas = torch.minimum(areas_inside_box, areas_outside_box)
    ss_cue = (1 - (min_of_two_areas.sum(0)/box_areas)).view(-1)
    ss_cue = ss_cue.clamp(max=1., min=0.)
    return ss_cue
