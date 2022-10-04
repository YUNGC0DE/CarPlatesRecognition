import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from detection.data_utils.data import reformat_coords
from detection.utils.thr import thr_output

from detection.train_utils.utils import init_model, aggregate_annotation, make_loaders

DEVICE = "cuda"
model = init_model(weigths="/home/evgenii/Desktop/ml_hw/CarPlates/CarPlatesRecognition/models/best.pth")
model.eval()
model.to(DEVICE)
writer = SummaryWriter()
T = transforms.Compose([transforms.ToPILImage(),
                        transforms.ToTensor()])


def one_image():
    image_orig = cv2.imread("1.jpg")
    image = T(cv2.resize(image_orig, (512, 512))).to(DEVICE)
    with torch.no_grad():
        output = model(image.unsqueeze(0))[0]
    boxes_one_all, boxes_two_all = thr_output(output)

    for box_one in boxes_one_all:
        print(box_one)
        box = reformat_coords(box_one, image_orig.shape[1], image_orig.shape[0])
        print(box)
        cv2.rectangle(image_orig, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

    for box_two in boxes_two_all:
        box = reformat_coords(box_two, image_orig.shape[1], image_orig.shape[0])
        cv2.rectangle(image_orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    cv2.imwrite(f"kek.png", image_orig)


def video():
    cap = cv2.VideoCapture("/home/evgenii/Desktop/ml_hw/CarPlates/video.mp4")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            image = T(cv2.resize(frame, (512, 512))).to(DEVICE)
            with torch.no_grad():
                output = model(image.unsqueeze(0))[0]
            boxes_one_all, boxes_two_all = thr_output(output)

            for box_one in boxes_one_all:
                print(box_one)
                box = reformat_coords(box_one, frame.shape[1], frame.shape[0])
                print(box)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

            for box_two in boxes_two_all:
                box = reformat_coords(box_two, frame.shape[1], frame.shape[0])
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
            cv2.imshow('Frame', frame)
            #cv2.imwrite(f"kek.png", image_orig)



def val_set():
    agregated = aggregate_annotation()
    val_loader = make_loaders(*agregated[2:], batch_size=1)
    for idx, (images, targets) in enumerate(val_loader):
        image = images[0]
        images = list(image.to(DEVICE) for image in images)
        with torch.no_grad():
            output = model(images)[0]
        boxes_one_all, boxes_two_all = thr_output(output)

        image = np.array(image.permute(1, 2, 0).numpy().copy()) * 255

        for boxes_one in boxes_one_all:
            cv2.rectangle(image, (boxes_one[0], boxes_one[1]), (boxes_one[2], boxes_one[3]), (255, 0, 0))

        for boxes_two in boxes_two_all:
            cv2.rectangle(image, (boxes_two[0], boxes_two[1]), (boxes_two[2], boxes_two[3]), (0, 255, 0))

        cv2.imwrite(f"/home/evgenii/Desktop/ml_hw/CarPlates/CarPlatesRecognition/test_images/{idx}.png", image)

video()