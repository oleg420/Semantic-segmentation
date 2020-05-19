import argparse

import cv2
import numpy as np

import torch

from Segmentation import Segmentation


def arg2source(x):
    try:
        return int(x)
    except:
        return x


def arg2type(x):
    if x in ['deeplab', 'unet']:
        return x
    else:
        raise ValueError(f'Unsupported network')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deeplabv3 Resnet50 Segmentation ')
    parser.add_argument('--classes', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)
    parser.add_argument('--type', type=arg2type, required=True)
    parser.add_argument('-pt', '--pytorch_model', type=str, required=True)

    parser.add_argument('--size', type=int, default=320)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    segm = Segmentation(model_path=args.pytorch_model, classes_path=args.classes, size=args.size, device=device, type=args.type)

    cap = cv2.VideoCapture(args.source)
    while True:
        _, image = cap.read()

        soft_segm, hard_segm = segm.detect(image, threshold=args.threshold)

        image = cv2.resize(image, (args.size, args.size))
        soft_segm_add = cv2.addWeighted(image, 1, soft_segm, 0.5, 0)
        hard_segm_add = cv2.addWeighted(image, 1, hard_segm, 0.5, 0)

        cv2.imshow('Image', image)
        # cv2.imshow('Soft_segm', soft_segm)
        # cv2.imshow('Hard_segm', hard_segm)
        cv2.imshow('Soft_segm_add', soft_segm_add)
        # cv2.imshow('Hard_segm_add', hard_segm_add)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
