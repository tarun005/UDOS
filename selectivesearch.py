# -*- coding: utf-8 -*-
from __future__ import division

import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import skimage.measure as measure
import numpy
# import mask as maskUtils
import pycocotools.mask as maskUtils
import numpy as np
from itertools import groupby

# from detectron2.structures.instances import Instances


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
#
#  - Modified version with LBP extractor for texture vectorization


def _generate_segments(im_orig, scale, sigma, min_size):
    """
        segment smallest regions by the algorithm of Felzenswalb and
        Huttenlocher
    """

    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        calculate the sum of histogram intersection of colour
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        calculate the sum of histogram intersection of texture
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        calculate the size similarity over the image
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        calculate the fill similarity over the image
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_colour_hist(img):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

        extract HSV
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img):
    """
        calculate texture gradient for entire image

        The original SelectiveSearch algorithm proposed Gaussian derivative
        for 8 orientations, but we use LBP instead.

        output will be [height(*)][width(*)]
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):
    """
        calculate texture histogram for each region

        calculate the histogram of gradient for each colours
        the size of output histogram will be
            BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):

        # mask by the colour channel
        fd = img[:, colour_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)

    return hist


def _extract_regions(img):

    R = {}

    # get hsv image
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # pass 1: count pixel positions
    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):

            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate colour histogram of each region
    for k, v in list(R.items()):

        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        # R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        binary_mask = (img[:,:,3] == k).astype(np.uint8)
        maskRgn = maskUtils.encode(numpy.asfortranarray(binary_mask))
        R[k]["mask"] = maskRgn
        R[k]["size"] = maskUtils.area(maskRgn)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])


    return R


def _extract_neighbours(regions):

    # def intersect(a, b):
    #     if (a["min_x"] < b["min_x"] < a["max_x"]
    #             and a["min_y"] < b["min_y"] < a["max_y"]) or (
    #         a["min_x"] < b["max_x"] < a["max_x"]
    #             and a["min_y"] < b["max_y"] < a["max_y"]) or (
    #         a["min_x"] < b["min_x"] < a["max_x"]
    #             and a["min_y"] < b["max_y"] < a["max_y"]) or (
    #         a["min_x"] < b["max_x"] < a["max_x"]
    #             and a["min_y"] < b["min_y"] < a["max_y"]):
    #         return True
    #     return False

    def intersect(a , b):
        return maskUtils.area(maskUtils.merge([a["mask"] , b["mask"]] , intersect=True))

    R = list(regions.items())
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_mask = maskUtils.merge([r1["mask"] , r2["mask"]], intersect=False)
    # new_size = r1["size"] + r2["size"]
    new_size = maskUtils.area(new_mask)
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size, "mask":new_mask,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(
        im_orig, scales=[1.0], sigma=0.8, min_size=50):
    '''Selective Search

    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (left, top, width, height),
                    'labels': [...],
                    'size': component_size,
                    'mask': np.ndarray
                },
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # load image and get smallest regions

    regions = []
    for scale in scales:
        img = _generate_segments(im_orig, scale, sigma, min_size)

        if img is None:
            return None, {}

        imsize = img.shape[0] * img.shape[1]
        scale_region = _extract_regions(img)
        regions.append(scale_region)

    # merge initial region
    R = regions[0]
    if len(scales) > 1:
        for region in regions[1:]:
            max_idx = max(R.keys())
            for r_idx in region:
                R[max_idx + r_idx] = region[r_idx]

    # extract neighbouring information
    neighbours = _extract_neighbours(R)

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search
    while S != {}:

        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in list(S.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in list(R.items()):
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels'],
            'mask': r['mask']
        })

    return img, regions

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(int(len(list(elements))))
    return rle

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
        COCO mask object: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    Copied from https://github.com/waspinator/pycococreator/blob/master/pycococreatortools/pycococreatortools.py
    """
    polygons = []
    binary_mask = maskUtils.decode(mask)
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = [np.subtract(c,1) for c in contours]# np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons

# def _to_segm(mask):

#     """
#         Convert a fRpyobject mask into poly mask for coco annotation
#     """
#     mask = maskUtils.decode(mask)
#     rle_obj = binary_mask_to_rle(mask)
#     return rle_obj

def compress_mask(mask):
    if type(mask["counts"]) == bytes:
        mask["counts"] = mask["counts"].decode("ascii")
    return mask

def getMasks(img, maskId=None, image_id=None, dest="plot"):
    """
        dest: what mode to return results in.
            "plot": return masks and boxes.
            "coco_eval": return decoded binary masks for evaluation.
            "coco_ann": return in coco json annotation format
    """
    assert dest in ["plot", "coco_eval", "coco_ann"]
    img_lbl, regions = selective_search(
        img, scales=[50,100], sigma=0.8, min_size=10)

    candidates = set()
    idxSet = []
    # print("Selected {} regions".format(len(regions)))
    for idx , r in enumerate(regions):
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller and larger
        if r['size']<500 or r['size']>85000:
            continue
        # # # distorted rects
        x, y, w, h = r['rect']
        try:
            if max(h/w, w/h) > 8.0:
                continue
        except:
            continue;
        candidates.add(r['rect'])
        idxSet.append(idx)
    regions = [r for idx,r in enumerate(regions) if idx in idxSet ]
    # print("Filtered to {} regions".format(len(candidates)))

    # ## Remove large masks which significantly overlap with smaller masks. Over segmentation is preferred.
    # ## Large masks tend to span multiple objects and backgrounds
    # sorted_regions = sorted(regions, key=lambda i:i["size"])[::-1]
    # # import pdb; pdb.set_trace()
    # removeIdx = []
    # for ii , r in enumerate(sorted_regions):
    #     for jj in range(ii+1,len(sorted_regions)):
    #         assert r["size"] >= sorted_regions[jj]["size"]
    #         iou_score = maskUtils.iou([r["mask"]], [sorted_regions[jj]["mask"]], [False])
    #         if iou_score >= 0.75:
    #             removeIdx.append(ii)
    #             break;
    # # import pdb; pdb.set_trace()
    # regions = [r for idx,r in enumerate(sorted_regions) if idx not in removeIdx ]
    # # print("Filtered to {} regions after area NMS".format(len(regions)))

    if dest == "coco_eval":
        if not len(regions):
            return None
        ## Change to coco format
        import torch
        boxes = torch.cat([torch.Tensor(r["rect"]).view(1,-1) for r in regions])
        boxes[:, 2:] += boxes[:, :2]
        masks = torch.stack([torch.tensor(maskUtils.decode(r["mask"])) for r in regions]).unsqueeze(1)
        scores = torch.ones(len(regions))
        labels = torch.ones_like(scores)
        return [{
            "boxes":boxes,
            "labels":labels,
            "scores":scores,
            "masks":masks
        }]
    
    if dest == "coco_ann":

        assert maskId is not None and image_id is not None
        anno = []

        for r in regions:
            # import pdb; pdb.set_trace()
            anno.append({
                "segmentation": [compress_mask(r["mask"])],
                "area": int(r["size"]),
                "iscrowd" : 0,
                "image_id" : image_id,
                "bbox" : maskUtils.toBbox(r["mask"]).tolist(),
                "category_id":1,
                'id' : maskId
            })
            maskId += 1
        
        return anno, maskId
        
    return regions


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import argparse
    from PIL import Image
    import numpy as np
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.structures.instances import Instances

    parser = argparse.ArgumentParser("Selective Search")
    parser.add_argument("--input_img", type=str, help="Input Image")
    parser.add_argument("--output_img", type=str, help="Output path/savepath")

    args = parser.parse_args()
    if args.output_img is None:
        ext = args.input_img.split(".")[-1]
        output_img = args.input_img.split(".")[0] + "_out." + ext
    else:
        output_img = args.output_img

    print("Saving to {}".format(output_img))

    img = np.asarray(Image.open(args.input_img).convert('RGB'))

    ########################
    ### Use SS algorithm ###
    ########################
    
    regions = getMasks(img)
    print("Found {} instances".format(len(regions)))
    instanceMask = []
    boxes = []
    for r in regions[:100]:
        instanceMask.append(maskUtils.decode(r["mask"]))
        boxes.append(maskUtils.toBbox(r["mask"]))
    instanceMask = np.stack(instanceMask, axis=0) > 0.5
    boxes = np.stack(boxes, axis=0)

    ########################
    ########################
    ########################


    ########################
    # Use Open CV
    ########################

    # import cv2
    # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # ss.setBaseImage(img)
    # ss.switchToSelectiveSearchFast()
    # boxes = ss.process()
    # print("Found {} boxes".format(len(boxes)))
    # boxes = boxes[:30]

    ########################
    ########################
    ########################

    # print("Creating instance object ...")
    prediction = Instances(img.shape[:2])
    prediction.pred_masks = instanceMask
    # prediction.pred_boxes = boxes

    visualizer = Visualizer(img, MetadataCatalog.get("__unused"), instance_mode=ColorMode.IMAGE)
    vis_output = visualizer.draw_instance_predictions(predictions=prediction)
    vis_output.save(output_img)
