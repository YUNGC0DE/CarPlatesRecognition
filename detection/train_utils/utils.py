import os
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from detection.data_utils.data import PlatesDataset


def make_loaders(images_train, targets_train, images_val=None, targets_val=None, batch_size=4):
    train_dataset = PlatesDataset(images_train, targets_train)
    train_loader = DataLoader(train_dataset, batch_size, False, collate_fn=collate)

    if images_val is not None:
        val_dataset = PlatesDataset(images_val, targets_val)
        val_loader = DataLoader(val_dataset, batch_size, False, collate_fn=collate)

        return train_loader, val_loader

    return train_loader


def init_model(weigths=None):
    model = fasterrcnn_resnet50_fpn(num_classes=3)
    if weigths is not None:
        w = torch.load(weigths)
        print(w["loss"])
        model.load_state_dict(w["model"])

    return model


def aggregate_annotation(ann_path: str = "/home/evgenii/Desktop/ml_hw/CarPlates/ann_json_full/",
                         images_path: str = "/home/evgenii/Desktop/ml_hw/CarPlates/images/"):
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
