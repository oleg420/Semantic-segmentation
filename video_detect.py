import cv2
import argparse
import time

import torch
from PIL import Image

from Segmentation import Segmentation
from utils import pad_to_square


def arg2source(x):
    try:
        return int(x)
    except:
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deeplabv3 Resnet50 Segmentation ')
    parser.add_argument('--classes', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)
    parser.add_argument('-pt', '--pytorch_model', type=str, required=True)

    parser.add_argument('--size', type=int, default=300)
    parser.add_argument('-t', '--threshold', type=float, default=0.3)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    segm = Segmentation(model_path=args.pytorch_model, classes_path=args.classes, size=args.size, device=device)

    cap = cv2.VideoCapture(args.source)
    while True:
        _, image = cap.read()
        h, w, _ = image.shape

        image, pad = pad_to_square(image)
        image = cv2.resize(image, (args.size, args.size))
        pad = int(pad / (w / image.shape[1]))

        # segmentation
        soft_segm, hard_segm = segm.detect(image, threshold=args.threshold)

        image = image[pad:image.shape[0]-pad, :, :]
        soft_segm = soft_segm[pad:soft_segm.shape[0]-pad, :, :]
        hard_segm = hard_segm[pad:hard_segm.shape[0]-pad, :, :]

        cv2.imshow('Image', image)
        cv2.imshow('soft_segm', soft_segm)
        cv2.imshow('hard_segm', hard_segm)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
