import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F

class customCoco(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile, transform=transform)
        self.transforms = transform

    def __getitem__(self, item):
        img, anns = super().__getitem__(item)

        boxes = [i['bbox'] for i in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]

        reg_h, reg_w = img.size
        scale_h, scale_w = 32/reg_h, 32/reg_w
        for i in range(len(boxes)):
            x, y, w, h = i
            boxes[i] = [
            x * scale_w,
            y * scale_h,
            w * scale_w,
            h * scale_h,
            ]
        

        tmp = [i.get('category_id', 1) for i in anns]
        labels = torch.as_tensor(tmp, dtype=torch.int64)

        image_id = torch.tensor([item])
        area = torch.as_tensor(
            [obj.get('area', (boxes[i,3]-boxes[i,1])*(boxes[i,2]-boxes[i,0]))
             for i, obj in enumerate(anns)],
            dtype=torch.float32
        )
        iscrowd = torch.as_tensor(
            [obj.get('iscrowd', 0) for obj in anns],
            dtype=torch.int64
        )

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        img = F.resize(img, (32, 32))

        return img, target

def collate(batch):
    return tuple(zip(*batch))