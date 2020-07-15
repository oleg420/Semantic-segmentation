import argparse

import cv2
import torch

from SegmentationDeeplabTRT import SegmentationDeeplabTRT


def arg2source(x):
    try:
        return int(x)
    except:
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pytorch_model', type=str, required=True)

    parser.add_argument('-cls', '--classes', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)

    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('-fp16', '--fp16', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SegmentationDeeplabTRT(model_path=args.pytorch_model, classes_path=args.classes, size=args.size, fp16=args.fp16)

    cap = cv2.VideoCapture(args.source)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        _, image = cap.read()

        segms = model(image, threshold=args.threshold)

        cv2.imshow('Image', image)
        for i, s in enumerate(segms):
            cv2.imshow(f'{model.get_class(i)}', s)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
