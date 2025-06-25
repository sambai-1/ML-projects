import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class license_coco(Dataset):
    """
    COCO but returns data in Faster-R-CNN / RetinaNet-style
    (image  → Tensor[C,H,W], target → dict w/ boxes & labels).
    """

    def __init__(self, root, ann_file, transforms=None):
        """
        root       : directory that holds all images
        ann_file   : path to _annotations.coco.json
        transforms : any torchvision / albumentations transform pipeline
        """
        self.root = root
        self.transforms = transforms

        # ---- load JSON & build look-ups ----
        coco = json.load(open(ann_file))
        self.images   = {img["id"]: img for img in coco["images"]}
        self.cats     = {cat["id"]: cat["name"] for cat in coco["categories"]}

        # map image_id → list[anno]
        self.annos_per_img = {}
        for ann in coco["annotations"]:
            self.annos_per_img.setdefault(ann["image_id"], []).append(ann)

        # keep a list of image_ids so __getitem__ has a stable index
        self.ids = sorted(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info   = self.images[img_id]
        path   = os.path.join(self.root, info["file_name"])

        # -------- load & transform the image --------
        img = Image.open(path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        # -------- build target dict --------
        annos = self.annos_per_img.get(img_id, [])
        boxes   = []
        labels  = []
        for a in annos:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])       # to [x1,y1,x2,y2]
            labels.append(a["category_id"])

        target = {
            "boxes":  torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        return img, target

def collate_fn(batch):
    images, targets = zip(*batch)   # tuple-of-lists → two lists
    return list(images), list(targets)
