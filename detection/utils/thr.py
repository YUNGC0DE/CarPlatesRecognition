import numpy as np
import torch
from torchvision.ops import nms


def thr_output(output):
    boxes = output["boxes"]
    scores = output["scores"]
    classes = output["labels"]
    cls_one_idx = torch.where(classes == 1)
    cls_two_idx = torch.where(classes == 2)
    boxes_one = boxes[cls_one_idx]
    scores_one = scores[cls_one_idx]
    boxes_two = boxes[cls_two_idx]
    scores_two = scores[cls_two_idx]
    one_idx = nms(boxes_one, scores_one, 0.2)
    two_idx = nms(boxes_two, scores_two, 0.2)
    scores_one = scores_one[one_idx].cpu().numpy()
    boxes_one = boxes_one[one_idx].cpu().numpy().astype(np.int16)
    scores_two = scores_two[two_idx].cpu().numpy()
    boxes_two = boxes_two[two_idx].cpu().numpy().astype(np.int16)
    boxes_one_idx = np.where(scores_one > 0.8)
    boxes_one_ = boxes_one[boxes_one_idx]
    boxes_two_idx = np.where(scores_two > 0.8)
    boxes_two_ = boxes_two[boxes_two_idx]
    return boxes_one_, boxes_two_
