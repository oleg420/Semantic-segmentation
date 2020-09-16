import argparse

import cv2
import numpy as np
import torch
import torchvision

from PIL import Image

from models import Deeplabv3Resnet50, Deeplabv3Resnet101

def arg2source(x):
    try:
        return int(x)
    except:
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=str, required=True)
    parser.add_argument('-s', '--source', type=arg2source, required=True)
    parser.add_argument('-pt', '--pt', type=str, required=True)

    parser.add_argument('-size', '--size', type=int, default=300)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = open(args.classes, 'r').read().splitlines()
    for i in range(len(classes)):
        classes[i] = int(classes[i])

    if args.backbone == 'resnet50':
        model = Deeplabv3Resnet50(len(classes)).to(device)
    else:
        model = Deeplabv3Resnet101(len(classes)).to(device)

    model.load_state_dict(torch.load(args.pt))
    model = model.eval()

    cap = cv2.VideoCapture(args.source)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        h, w, _ = frame.shape
        if ret:
            img = Image.fromarray(frame)
            img = torchvision.transforms.Resize((args.size, args.size))(img)
            img = torchvision.transforms.ToTensor()(img)
            img = img.to(device)

            segmentations = model(img.unsqueeze(0)).squeeze(0)

            cv2.imshow('Image', frame)
            for i, segmentation in enumerate(segmentations):
                tmp = torchvision.transforms.ToPILImage()(segmentation.detach().cpu())
                tmp = torchvision.transforms.Resize((h, w))(tmp)
                tmp[tmp >= args.threshold] = 255
                tmp[tmp < args.threshold] = 0
                tmp = np.array(tmp, dtype=np.uint8)

                cv2.imshow(f'{classes[i]}', tmp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
