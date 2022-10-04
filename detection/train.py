import os.path
from glob import glob

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import RAdam

from data_utils.data import PlatesDataset
from sklearn.model_selection import train_test_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def init_model():
    model = fasterrcnn_resnet50_fpn(num_classes=3)
    print(model)
    return model


def aggregate_annotation(ann_path: str = "/home/evgenii/Desktop/ML_HW/car_plates/ann_json_full/",
                         images_path: str = "/home/evgenii/Desktop/ML_HW/car_plates/images/"):

    anns = sorted(glob(os.path.join(ann_path, "*")), key=lambda x: int(x.split('/')[-1].split(".json")[0][4:]))
    images = sorted(glob(os.path.join(images_path, "*")), key=lambda x: int(x.split('/')[-1].split(".png")[0][4:]))
    assert len(anns) == len(images)

    images_train, images_val, targets_train, targets_val = train_test_split(images, anns, test_size=0.2, random_state=1)
    return images_train, targets_train, images_val, targets_val


def collate(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def make_loaders(images_train, targets_train, images_val, targets_val, batch_size=4):
    train_dataset = PlatesDataset(images_train, targets_train)
    val_dataset = PlatesDataset(images_val, targets_val)
    train_loader = DataLoader(train_dataset, batch_size, False, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size, False, collate_fn=collate)

    return train_loader, val_loader


DEVICE = "cuda"
model = init_model()
model.train()
model.to(DEVICE)
optimizer = RAdam(model.parameters())

agregated = aggregate_annotation()
train_loader, val_loader = make_loaders(*agregated)
print("Kek")
for _ in range(10):
    for i, data in enumerate(train_loader):
        print("kek")
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        print(loss_value)
        losses.backward()
        optimizer.step()

