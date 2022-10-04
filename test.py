import copy
import warnings
from collections import defaultdict

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from ocr.detector import Detector
from ocr.recognizer import Recognizer
from detection.data_utils.data import reformat_coords, plate_in_car
from detection.utils.thr import thr_output
from detection.sort import *
from detection.train_utils.utils import init_model, aggregate_annotation, make_loaders

warnings.filterwarnings("ignore")
DEVICE = "cuda"
detector = init_model(weigths="/home/evgenii/Desktop/ml_hw/CarPlates/CarPlatesRecognition/models/best.pth")
detector.eval()
detector.to(DEVICE)
writer = SummaryWriter()
T = transforms.Compose([transforms.ToPILImage(),
                        transforms.ToTensor()])


def video():
    vid_writer = cv2.VideoWriter("1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1920, 1080))
    cap = cv2.VideoCapture("/home/evgenii/Desktop/ml_hw/CarPlates/video.mp4")
    mot_tracker = Sort()
    counter = 0
    craft = Detector()
    craft.load()

    recognizer = Recognizer()
    recognizer.load()
    bd = defaultdict(int)
    while (cap.isOpened()):
        counter += 1
        print(counter)
        ret, frame = cap.read()
        if ret == True:
            frame[:, 0:520] = 255
            image = T(cv2.resize(frame, (512, 512))).to(DEVICE)
            with torch.no_grad():
                output = detector(image.unsqueeze(0))[0]
            car_boxes, car_scores, plate_boxes = thr_output(output)
            detections = []
            for car_box, car_score in zip(car_boxes, car_scores):
                box = reformat_coords(car_box, frame.shape[1], frame.shape[0])
                box.append(car_score)
                detections.append(np.array(box))
            if len(detections) == 0:
                continue
            track_bbs_ids = mot_tracker.update(np.array(detections))
            tracked_cars = []
            for i in range(len(track_bbs_ids)):
                box = list(map(int, track_bbs_ids.tolist()[i]))
                tracked_cars.append(copy.deepcopy(box))
                idx = box.pop()
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
                cv2.putText(frame, f"car_ID: {idx}", (box[0] + 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 0, 0), thickness=3)

            for plate_box in plate_boxes:
                box = reformat_coords(plate_box, frame.shape[1], frame.shape[0])
                idx = plate_in_car(box, tracked_cars, frame.shape[1], frame.shape[0])
                if not idx:
                    continue

                crop = frame[box[1]: box[3], box[0]:box[2]]
                roi, _, _, _ = craft.process(crop)
                full_text = ""
                for i, img in enumerate(roi):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    text, _, _ = recognizer.process(gray)
                    full_text += f"{text} "
                if len(full_text.replace(" ", "")) == 0:
                    continue

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
                cv2.putText(frame, f"plate_car_ID: {idx}", (box[0] + 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0, 255, 0), thickness=3)

                full_text = f"ID:{idx}: {full_text}"
                bd[full_text] += 1

                for i, (k, v) in enumerate(bd.items()):
                    if v > 15:
                        cv2.putText(frame, k, (30, 110 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1.8,
                                    color=(0, 0, 0), thickness=4)

            vid_writer.write(frame)
        else:
            break


def one_image():
    image_orig = cv2.imread("images/1.jpg")
    image = T(cv2.resize(image_orig, (512, 512))).to(DEVICE)
    with torch.no_grad():
        output = detector(image.unsqueeze(0))[0]
    boxes_one_all, _,  boxes_two_all = thr_output(output)

    for box_one in boxes_one_all:
        print(box_one)
        box = reformat_coords(box_one, image_orig.shape[1], image_orig.shape[0])
        print(box)
        cv2.rectangle(image_orig, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

    for box_two in boxes_two_all:
        box = reformat_coords(box_two, image_orig.shape[1], image_orig.shape[0])
        cv2.rectangle(image_orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))

    cv2.imwrite(f"car.png", image_orig)


def val_set():
    agregated = aggregate_annotation()
    val_loader = make_loaders(*agregated[2:], batch_size=1)
    for idx, (images, targets) in enumerate(val_loader):
        image = images[0]
        images = list(image.to(DEVICE) for image in images)
        with torch.no_grad():
            output = detector(images)[0]
        boxes_one_all, boxes_two_all = thr_output(output)

        image = np.array(image.permute(1, 2, 0).numpy().copy()) * 255

        for boxes_one in boxes_one_all:
            cv2.rectangle(image, (boxes_one[0], boxes_one[1]), (boxes_one[2], boxes_one[3]), (255, 0, 0))

        for boxes_two in boxes_two_all:
            cv2.rectangle(image, (boxes_two[0], boxes_two[1]), (boxes_two[2], boxes_two[3]), (0, 255, 0))

        cv2.imwrite(f"/home/evgenii/Desktop/ml_hw/CarPlates/CarPlatesRecognition/test_images/{idx}.png", image)


if __name__ == "__main__":
    one_image()
