from collections import defaultdict
import json
import os
from PIL import Image
import torch
from torchvision.datasets import CocoDetection

class license_coco(CocoDetection):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.transforms = transforms

        #connect annotations to images
        json_file = json.load(open(ann_file))
        self.images = {image["id"]: image for image in json_file["images"]}
        self.labels = {labels["id"]: labels["name"] for labels in json_file["categories"]}

        self.annotations = defaultdict(list)
        for item in json_file["annotations"]:
            self.annotations[item["image_id"]].append(item)

        self.index = sorted(self.images.keys())
    
    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        image_id = self.index[index]
        image_data = self.images[image_id]
        image_path = os.path.join(self.root, image_data["file_name"].replace("\\", "/"))
        print(f"Opening: {image_path}")
        img = Image.open(image_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)

        annotations = self.annotations.get(image_id, [])
        boxes   = []
        labels  = []
        for charactor in annotations:
            if charactor["category_id"] == 17:
                continue
            x, y, w, h = charactor["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(charactor["category_id"])

        target = {
            "boxes":  torch.as_tensor(boxes),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id])
        }

        return img, target 

def license_collate(batch):
    images, targets = zip(*batch)
    print(images, targets)

    images = torch.stack(images)

    # Flatten all target sequences and record their lengths
    flat_labels = []
    target_lengths = []

    for t in targets:
        label = t["labels"]
        flat_labels.extend(label.tolist())
        target_lengths.append(len(label))

    print(flat_labels)
    flat_targets = torch.tensor(flat_labels)
    target_lengths = torch.tensor(target_lengths)

    return images, flat_targets, target_lengths
