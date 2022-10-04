import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import RAdam

from detection.train_utils.utils import init_model, aggregate_annotation, make_loaders

DEVICE = "cuda"
NUM_EPOCHS = 15
best_loss = 100
model = init_model()
model.train()
model.to(DEVICE)
optimizer = RAdam(model.parameters(), lr=0.001)

agregated = aggregate_annotation()
train_loader, val_loader = make_loaders(*agregated)
writer = SummaryWriter()


def train(step):
    loss = []
    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss.append(loss_value)
        losses.backward()
        optimizer.step()
    writer.add_scalar("Train_Loss", np.mean(loss), step)


def val(step):
    global best_loss
    loss = []
    model.train()
    for i, data in enumerate(val_loader):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss.append(loss_value)
    writer.add_scalar("Val_Loss", np.mean(loss), step)
    loss = np.mean(loss)
    if loss < best_loss:
        torch.save({"model": model.state_dict(), "loss": loss},
                   "/home/evgenii/Desktop/ml_hw/CarPlates/CarPlatesRecognition/models/best.pth")
        best_loss = loss


for i in range(NUM_EPOCHS):
    train(i)
    val(i)

