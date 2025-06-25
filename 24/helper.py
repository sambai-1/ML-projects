import os
from PIL import Image
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import torch


class resize:
    def __init__(self, size):
        # size = (width, height)
        self.size = size

    def __call__(self, image, target):
        orig_w, orig_h = image.size
        new_h, new_w = self.size

        image = F.resize(image, (new_h, new_w))

        scale_w = new_w / orig_w
        scale_h = new_h / orig_h
        for ann in target:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [
                x * scale_w,
                y * scale_h,
                w * scale_w,
                h * scale_h,
            ]

        return image, target


class transformPair:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class customCoco(CocoDetection):
    def __init__(self, root, annFile, size=(128, 128), transform=None):
        super().__init__(root, annFile, transforms=transformPair([resize(size)]))
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

def collate(batch):
    images, annotations = list(zip(*batch))
    
    targets = []
    for i, objs in enumerate(annotations):
        # objs is a list of dicts for image i
        # extract bboxes and labels
        bboxes = [i['bbox'] for i in objs]
        labels = [i['category_id'] for i in objs]
        # convert from [x,y,w,h] to [x_min,y_min,x_max,y_max]
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            bboxes[:, 2:] = bboxes[:, :2] + bboxes[:, 2:]
            #area = torch.tensor([o['area'] for o in objs], dtype=torch.float32)
            #iscrowd = torch.tensor([o['iscrowd'] for o in objs], dtype=torch.int64)
        else:
            bboxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
        
        targets.append({
            'boxes': bboxes,
            'labels': labels,
        })
    return list(images), targets