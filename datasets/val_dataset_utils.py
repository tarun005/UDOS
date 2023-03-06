import copy
import os
from PIL import Image

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from .DetectionLoader import Detection

import transforms as T


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return image, target
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
        except:
            mask = coco_mask.decode(polygons)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, indices) -> None:
        super().__init__()
        self.indices = indices

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        if len(anno) and "iscrowd" in anno[0]:
            anno = [obj for obj in anno if obj['iscrowd'] == 0]
        else: ## Hack for LVIS
            for obj in anno:
                obj["iscrowd"] = 0

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if len(anno) and "segmentation" in anno[0]:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)
        else:
            masks = torch.zeros((len(anno) , h, w), dtype=torch.uint8)


        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if self.indices and len(keep):
            keep = keep & torch.tensor([c in self.indices for c in classes] )
        
        # assert any(keep), f"Image id {image_id} does not have matching instances."
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        # for a in anno:
        #     if a["image_id"] not in [329703, 206685, 169701, 352612, 211141, 56599, 335503, 336113]:
        #         return False
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, Detection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds, eval_classes):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # print("{}/{}".format(img_idx+1, len(ds)), end="\r", flush=True)
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()
        if 'masks' in targets:
            masks = targets['masks']
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if 'keypoints' in targets:
            keypoints = targets['keypoints']
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            if labels[i] in eval_classes:
                ann["ignored_split"] = 0
            else:
                ann["ignored_split"] = 1
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['iscrowd'] = iscrowd[i]
            ann['id'] = ann_id
            if 'masks' in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if 'keypoints' in targets:
                ann['keypoints'] = keypoints[i]
                ann['num_keypoints'] = sum(k != 0 for k in keypoints[i][2::3])
            dataset['annotations'].append(ann)
            ann_id += 1
    print(categories)
    print(len(categories))
    # categories = set([1])
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    # # print(set(categories))
    # # print(len(set(categories)))
    # # print(len(dataset['annotations']))
    # import json
    # with open("voc_only_val.json" , "w") as fh:
    #     json.dump(dataset, fh)
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset, set, indices):
    for _ in range(10):
        if isinstance(dataset, Detection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if set not in ["coco", "obj365"]:
        if isinstance(dataset, Detection):
            return dataset.coco
    print("Creating dataset index .... ")
    return convert_to_coco_api(dataset, indices)


class CocoDetection(Detection):
    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertLabelsToBinary(object):
    def __call__(self, image, target):
        # import pdb; pdb.set_trace()
        target["labels"] = torch.ones_like(target["labels"], dtype=torch.int64)
        return image, target
        

def get_coco_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_{}2017.json"

    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    # from VOStoCOCO import mapping
    # if subset == "voc":
    #     indices = mapping._common_with_voc
    # elif subset == "coco":
    #     indices = mapping._exclusive_to_coco
    # else:
    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    # if toBinary:
    #     t.append(ConvertLabelsToBinary())
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset


def get_lvis_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all", spp=None):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "lvis_v1_{}_coco_filtered_0.5_oln_compatible_cocoval.json"

    PATHS = {
        "train": ("train2017", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("val2017", "/newfoundland2/tarun/datasets/LVIS/lvis_v1_val_singlelabel.json"),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    if toBinary:
        t.append(ConvertLabelsToBinary())
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("After", len(dataset))

    return dataset


def get_uvo_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_sparse.json"

    PATHS = {
        "train": ("all_UVO_frames", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("all_UVO_frames", os.path.join("annotations", anno_file_template.format("val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    # if image_set == "val":
    #     dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset

def get_openimages_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "instances_{}_V6_littlearea.json"

    PATHS = {
        "train": ("validation", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("validation", os.path.join("annotations", anno_file_template.format("val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    # if image_set == "val":
    #     dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset

def get_ade20k_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_instances_weiyao.json"
    # anno_file_template = "{}.json"

    PATHS = {
        "train": ("", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("", os.path.join("annotations", anno_file_template.format("val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    # if image_set == "val":
    #     dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset

def get_obj365_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "boxes_{}.json"

    PATHS = {
        "train": ("", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("", os.path.join("annotations", anno_file_template.format("val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    t.append(ConvertLabelsToBinary())
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    # if image_set == "val":
    #     dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset

def get_cityscapes_test(root, image_set, transforms, mode='instances', toBinary=False, subset="all"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "instancesonly_filtered_gtFine_{}.json"

    PATHS = {
        "train": ("", os.path.join("annotations", anno_file_template.format("train"))),
        "val": ("", os.path.join("annotations", anno_file_template.format("val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    indices = None
    assert isinstance(indices, (list , type(None)))

    t = [ConvertCocoPolysToMask(indices)]

    if transforms is not None:
        t.append(transforms)
    
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)
    print("Before", len(dataset))

    # if image_set == "val":
    #     dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)
    print("After", len(dataset))

    return dataset

if __name__ == "__main__":

    import presets, pdb
    from torch.utils.data import DataLoader
    from utils import collate_fn

    dataset_train = get_obj365_test("/newfoundland2/tarun/datasets/Objects365", "val", presets.DetectionPresetTrain("hflip"), subset="coco")
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=8, collate_fn=collate_fn)

    labels = []
    counts = 0
    for i , (imgs, target) in enumerate(data_loader_train):
        # import pdb; pdb.set_trace()
        print(f"{i+1}/{len(data_loader_train)}" , end="\r")
    #     io.write_video("video_1.mp4" , video, fps=30)
        allLabels = torch.cat([(t["labels"]) for t in target], dim=0)
        # counts += len(target[0]["labels"])
        labels.append(allLabels)
        
        if i%1000 == 0:
            print(torch.cat(labels, dim=0).unique())
    print()
    print(torch.cat(labels, dim=0).unique())
    print(counts)
