import copy
import os, sys
from PIL import Image
import random

import torch
import torch.utils.data
import torchvision

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from torchvision.transforms import functional as F

from .DetectionLoader import Detection

import transforms as T

def horizontalFlip(image, target):

    image = F.hflip(image)
    width, _ = F._get_image_size(image)
    target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)

    return image, target

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

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

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

    assert isinstance(dataset.det_gt, Detection)
    assert isinstance(dataset.det_spp, Detection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.det_gt.ids):
        ann_ids = dataset.det_gt.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.det_gt.coco.loadAnns(ann_ids)

        ann_ids_spp = dataset.det_spp.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno_spp = dataset.det_spp.coco.loadAnns(ann_ids_spp)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno) and _has_valid_annotation(anno_spp):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        print("{}/{}".format(img_idx+1, len(ds)), end="\r", flush=True)
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
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = 1# labels[i]
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
    categories = set([1])
    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset, set=None, indices=None):
    for _ in range(10):
        if isinstance(dataset, Detection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        return dataset.det_gt.coco
    # if 1:#isinstance(dataset, torchvision.datasets.CocoDetection):
    # return convert_to_coco_api(dataset.det_gt)


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

class CocoDetectionMerged(torch.utils.data.Dataset):
    def __init__(self, img_folder, ann_file_spp, ann_file_gt, transforms_gt, transforms_spp, split="train"):
        super().__init__()
        self.det_gt = CocoDetection(img_folder, ann_file_gt, transforms_gt)
        self.det_spp = CocoDetection(img_folder, ann_file_spp, transforms_spp)
        self.split = split

    def __getitem__(self, idx):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        img1, target_spp = self.det_spp.__getitem__(idx)
        img2, target_gt = self.det_gt.__getitem__(idx)

        if random.random() > 0.5 and self.split=="train": ## Avoid different augmentations for both loaders
            img1, target_spp = horizontalFlip(img1, target_spp)
            img2, target_gt = horizontalFlip(img2, target_gt)

        assert torch.allclose(img1, img2)
        return img1, target_spp, target_gt

    def __len__(self):
        return self.det_gt.__len__()

class ConvertLabelsToBinary(object):
    def __call__(self, image, target):
        target["labels"] = torch.ones_like(target["labels"], dtype=torch.int64)
        return image, target
        

def get_uvo(root, image_set, transforms, mode='instances', toBinary=False, subset="all", spp="new"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_{}2017.json"
 
    if os.path.exists("/checkpoint/trandu/OpenSegmentationProject"):
        gt_path = os.path.join("/checkpoint/trandu/OpenSegmentationProject/data/UVO/annotations/" , "train_sparse.json")
    else:
        gt_path = os.path.join(root , "annotations", "train_sparse_shorter.json" )

    if spp == "new":
        if os.path.exists("/checkpoint/trandu/OpenSegmentationProject"):
            spp_path = os.path.join("/checkpoint/trandu/OpenSegmentationProject/data/UVO/annotations/" , "train_sparse_part_cropped.json")
        else:
            spp_path = os.path.join(root , "annotations", "train_sparse_part_cropped.json" )

    else:
        if os.path.exists("/checkpoint/trandu/OpenSegmentationProject"):
            spp_path = os.path.join("/checkpoint/trandu/OpenSegmentationProject/data/UVO/annotations/" , "train_sparse_part.json")
        else:
            spp_path = os.path.join(root , "annotations", "train_sparse_part.json" )

    PATHS = {
        "train_gt": ("all_UVO_frames", gt_path),
        "train_spp": ("all_UVO_frames",spp_path)
    }

    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t_gt = [ConvertCocoPolysToMask(indices)]
    t_spp = [ConvertCocoPolysToMask(indices=None)]

    if transforms is not None:
        t_gt.append(transforms)
        t_spp.append(transforms)
    
    if toBinary:
        t_gt.append(ConvertLabelsToBinary())

    transforms_gt = T.Compose(t_gt)
    transforms_spp = T.Compose(t_spp)

    img_folder_gt, ann_file_gt = PATHS["train_gt"]
    img_folder_spp , ann_file_spp = PATHS["train_spp"]

    img_folder_gt = os.path.join(root, img_folder_gt)
        
    dataset = CocoDetectionMerged(img_folder_gt, ann_file_spp=ann_file_spp, ann_file_gt=ann_file_gt, transforms_gt=transforms_gt, transforms_spp=transforms_spp, split=image_set)
    print("Before", len(dataset))

    # if image_set == "train":
    dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)

    print("After", len(dataset))

    return dataset

def get_coco(root, image_set, transforms, mode='instances', toBinary=False, subset="all", spp="ss"):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_{}2017.json"

    if image_set == "train":
        if os.path.exists(os.path.join(root, "annotations/instances_train2017_MCG.json")):
            PATHS = {
                "train_gt": ("train2017", os.path.join("annotations", anno_file_template.format(mode, "train"))),
                "train_spp": ("train2017", os.path.join("annotations/instances_train2017_MCG.json")),
            }
        else:
            print("SuperPixel file not found.")
            sys.exit(0)
    else:
        PATHS = {
                    "train_gt": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
                    "train_spp" : ("val2017", os.path.join("annotations/instances_val2017_MCG_complete.json")),
                }


    from VOCtoCOCO import mapping
    if subset == "voc":
        indices = mapping._common_with_voc
        print("Clipping to VOC")
    elif subset == "coco":
        indices = mapping._exclusive_to_coco
        print("Clipping to N-VOC")
    else:
        indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t_gt = [ConvertCocoPolysToMask(indices)]
    t_spp = [ConvertCocoPolysToMask(indices=None)]

    if transforms is not None:
        t_gt.append(transforms)
        t_spp.append(transforms)
    
    if toBinary:
        t_gt.append(ConvertLabelsToBinary())
        t_spp.append(ConvertLabelsToBinary())

    transforms_gt = T.Compose(t_gt)
    transforms_spp = T.Compose(t_spp)

    img_folder_gt, ann_file_gt = PATHS["train_gt"]
    img_folder_spp , ann_file_spp = PATHS["train_spp"]
        
    img_folder = os.path.join(root, img_folder_gt)
    ann_file_spp = os.path.join(root, ann_file_spp)
    ann_file_gt = os.path.join(root, ann_file_gt)

    dataset = CocoDetectionMerged(img_folder, ann_file_spp=ann_file_spp, ann_file_gt=ann_file_gt, transforms_gt=transforms_gt, transforms_spp=transforms_spp, split=image_set)
    print("Before", len(dataset))

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)

    print("After", len(dataset))

    return dataset

def get_lvis(root, image_set, transforms, mode='instances', toBinary=False, subset="all", spp=None):
    """
        subset: all stands for complete CoCo subset
                voc stands for 20 classes in common between COCO and VOC.
                coco stands for 60 classes exclusive to coco
    """
    print("Clipping subset to:" , subset)
    anno_file_template = "{}_{}2017.json"

    PATHS = {
            # "train_gt": ("train2017", "/checkpoint/trandu/coco_ann/LVIS/"),
            # "train_spp": ("train2017", "/checkpoint/trandu/coco_ann/instances_train2017_MCG.json"),
            "train_gt" : ("train2017" , "/newfoundland2/tarun/datasets/LVIS/lvis_train.json"),
            "train_spp": ("train2017", "/newfoundland2/tarun/datasets/COCO/annotations/instances_train2017_MCG.json"),
        }


    indices = None # Or all
    assert isinstance(indices, (list , type(None)))

    t_gt = [ConvertCocoPolysToMask(indices)]
    t_spp = [ConvertCocoPolysToMask(indices=None)]

    if transforms is not None:
        t_gt.append(transforms)
        t_spp.append(transforms)
    
    if toBinary:
        t_gt.append(ConvertLabelsToBinary())

    transforms_gt = T.Compose(t_gt)
    transforms_spp = T.Compose(t_spp)

    img_folder_gt, ann_file_gt = PATHS["train_gt"]
    img_folder_spp , ann_file_spp = PATHS["train_spp"]
        
    dataset = CocoDetectionMerged(root, ann_file_spp=ann_file_spp, ann_file_gt=ann_file_gt, transforms_gt=transforms_gt, transforms_spp=transforms_spp, split=image_set)
    print("Before", len(dataset))

    dataset = _coco_remove_images_without_annotations(dataset, cat_list=indices)

    print("After", len(dataset))

    return dataset

if __name__ == "__main__":

    import presets, pdb
    from torch.utils.data import DataLoader
    from utils import collate_fn

    dataset_train = get_coco("/checkpoint/tarun05/data/UVO", "train", presets.DetectionPresetTrain("hflip"), subset="common")
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    labels = []
    counts = 0
    for i , (imgs, target_spp, target_gt) in enumerate(data_loader_train):
        # if any([len(t["boxes"]) == 0 for t in target_spp]): 
        pdb.set_trace()
        
        # print(f"{i+1}/{len(data_loader_train)}" , end="\r")
        # print(target_spp[0]["masks"].sum(-1).sum(-1).max())
        print(len(target_gt[0]["masks"]))
        print(len(target_spp[0]["masks"]))

    #     io.write_video("video_1.mp4" , video, fps=30)
        # pdb.set_trace()
        # allLabels = torch.cat([(t["labels"]) for t in target], dim=0)
        counts += len(target_gt[0]["labels"])
        # labels.append(allLabels)
        
        # if i%1000 == 0:
        #     print(torch.cat(labels, dim=0).unique())
    print()
    print(counts)
