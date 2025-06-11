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
    def __init__(self, root, annFile, size=(32, 32), transform=None):
        super().__init__(root, annFile, transforms=transformPair([resize(size)]))
        self.transform = transform

    def __getitem__(self, index):
        image, target = super().__getitem__(index)

        if self.transform is not None:
            image = self.transform(image)

        return image, target

def collate(batch):
    images, annots = list(zip(*batch))
    
    targets = []
    for i, objs in enumerate(annots):
        # objs is a list of dicts for image i
        # extract bboxes and labels
        boxes  = torch.tensor([o['bbox'] for o in objs], dtype=torch.float32)
        labels = torch.tensor([o['category_id'] for o in objs], dtype=torch.int64)
        # convert from [x,y,w,h] to [x_min,y_min,x_max,y_max]
        boxes[:,2:] = boxes[:,:2] + boxes[:,2:]
        
        targets.append({
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([i]),      # optional
            'area': torch.tensor([o['area'] for o in objs]),
            'iscrowd': torch.tensor([o['iscrowd'] for o in objs]),
        })
    return list(images), targets