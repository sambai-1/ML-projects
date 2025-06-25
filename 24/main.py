import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from PIL import Image
import sys

DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

def load_models():
    plate_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    plate_model.load_state_dict(torch.load("plate_model.pth"))
    plate_model.to(DEVICE)
    plate_model.eval()
    return [plate_model]

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).to(DEVICE)

def largest_bbox(plate):
    boxes = plate['boxes']
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    max_idx = areas.argmax().item()
    return boxes[max_idx].tolist()

def run(image_path):
    models = load_models()
    plate_model = models[0]
    image = process_image(image_path)

    with torch.no_grad():
        plate_output = plate_model([image])[0]
    if len(plate_output['boxes']) == 0:
        print("No plate in picture")
        return
    else:
        max_bbox = largest_bbox(plate_output)
        print("Bounding box with max area:", max_bbox)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_max_bbox.py <image_path>")
    else:
        run(sys.argv[1])