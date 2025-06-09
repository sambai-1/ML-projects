import torch
from torchvision.datasets import CocoDetection

class customCoco(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root=root, annFile=annFile, transform=transform)
        self.transforms = transform

    def __getitem__(self, item):
        img, anns = super().__getitem__(item)

        boxes = [i['bbox'] for i in anns]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]

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

        return img, target

def collate(batch):
    return tuple(zip(*batch))