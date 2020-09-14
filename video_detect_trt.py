import argparse

import cv2
import numpy as np
import torch

import pycuda.driver as cuda
from trt.trt_wrapper import Deeplabv3TRT

def arg2source(x):
    try:
        return int(x)
    except:
        return x

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)
    parser.add_argument('-e', '--engine-path', type=str, required=True)

    parser.add_argument('-size', '--size', type=int, default=300)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('-t', '--threshold', type=float, default=0.3)
    args = parser.parse_args()
    print(args)

    assert torch.cuda.is_available(), 'TRT support only on CUDA-compatible GPU'

    cuda.init()
    device = cuda.Device(0)

    ctx = device.make_context()
    stream = cuda.Stream()

    classes = open(args.classes, 'r').read().splitlines()

    model = Deeplabv3TRT(args.engine_path, len(classes), args.size, ctx, stream)

    cap = cv2.VideoCapture(args.source)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame, (args.size, args.size))
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)
            img = np.divide(img, 255)

            segmentations = model(img.copy())[0]

            cv2.imshow('Image', frame)
            for i, segmentation in enumerate(segmentations):
                tmp = cv2.resize(segmentation, (w, h))
                tmp[tmp >= args.threshold] = 255
                tmp[tmp < args.threshold] = 0
                tmp = np.array(tmp, dtype=np.uint8)

                cv2.imshow(f'{classes[i]}', tmp)
                if i > 7:
                    continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    ctx.pop()