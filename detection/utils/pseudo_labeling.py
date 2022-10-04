import json

import cv2
from glob import glob
from tqdm import tqdm
import torch
from torchvision.transforms import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import nms

thr = 0.5
device = "cuda"
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

files = glob("/home/evgenii/Desktop/ML_HW/car_plates/images/*")
t = transforms.Compose([transforms.ToPILImage(),
                        transforms.ToTensor()])

for path in tqdm(files):
    img_name = path.split("/")[-1]
    img_orig = cv2.imread(path)
    img = t(img_orig).to(device)
    with torch.no_grad():
        output = model(img.unsqueeze(0))[0]
    labels = output["labels"]
    boxes = output["boxes"]
    scores = output["scores"]
    car_idx = torch.where(labels == 3)
    boxes, scores = boxes[car_idx], scores[car_idx]
    score_idx = torch.where(scores > thr)
    boxes, scores = boxes[score_idx], scores[score_idx]
    boxes_idx = nms(boxes, scores, 0.2)
    boxes = boxes[boxes_idx].cpu().numpy()

    with open(path.replace(".png", ".json").replace("images", "ann_json")) as ann_f:
        ann = json.load(ann_f)

    for an in ann:
        box = an["box"]
        cv2.rectangle(img_orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    for box in boxes:
        box = list(map(int, box))
        ann.append({
            "class": 1,
            "box": box
        })
        cv2.rectangle(img_orig, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

    with open(path.replace(".png", ".json").replace("images", "ann_json_full"), "w") as ann_f:
        json.dump(ann, ann_f)

    cv2.imwrite("/home/evgenii/Desktop/ML_HW/car_plates/viz/" + img_name, img_orig)
