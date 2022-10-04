import json

import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as torch_transforms

T = torch_transforms.Compose([torch_transforms.ToPILImage(),
                              torch_transforms.ToTensor()])


class PlatesDataset(Dataset):
    def __init__(self, images, annotation, transforms=T):
        self.transforms: torch_transforms = transforms
        self.images = images
        self.annotations = annotation

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_path = self.images[idx]
        # read the image

        image_orig = cv2.imread(image_path)
        image_width = image_orig.shape[1]
        image_height = image_orig.shape[0]
        image = self.transforms(cv2.resize(image_orig, (512, 512)))
        boxes = []
        labels = []

        with open(self.annotations[idx]) as ann_f:
            ann = json.load(ann_f)

        for an in ann:
            box = an["box"]
            x1 = (box[0]/image_width) * 512
            x2 = (box[2]/image_width) * 512
            y1 = (box[1]/image_height) * 512
            y2 = (box[3]/image_height) * 512
            boxes.append([x1, y1, x2, y2])
            labels.append(an["class"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        #if self.transforms is not None:
        #    sample = self.transforms(image=image,
        #                             bboxes=target['boxes'],
        #                             labels=labels)
        #    image = sample['image']
        #    target['boxes'] = torch.Tensor(sample['bboxes'])
        return image, target

    def __len__(self):
        return len(self.images)


def reformat_coords(box, width, height):
    x1 = int((box[0] / 512) * width)
    x2 = int((box[2] / 512) * width)
    y1 = int((box[1] / 512) * height)
    y2 = int((box[3] / 512) * height)
    return [x1, y1, x2, y2]
