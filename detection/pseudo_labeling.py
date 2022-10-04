import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn

thr = 0.85
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to("cuda")

path = "C:/Users/pilot/Desktop/ML_HW/CarPlates/images/Cars264.png"
t = transforms.Compose([transforms.ToPILImage(),
                        transforms.ToTensor()])

img_orig = cv2.imread(path)
img = t(img_orig)
with torch.no_grad():
    output = model(img.unsqueeze(0))[0]
labels = output["labels"]
boxes = output["boxes"]
scores = output["scores"]
car_idx = np.where(labels == 3)
boxes, scores = boxes[car_idx], scores[car_idx]
score_idx = np.where(scores > thr)
boxes, scores = boxes[score_idx].cpu().numpy(), scores[score_idx].cpu().numpy()
print(list(boxes), list(scores))

