import os
import argparse

import torch
from models import Deeplabv3Resnet50, Deeplabv3Resnet101

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cls', '--classes', type=str, required=True)
    parser.add_argument('-pt', '--pt', type=str, required=True)
    parser.add_argument('-sp', '--save-path', type=str, default='./')

    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-s', '--size', type=int, default=300)

    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()
    print(args)

    device = 'cuda' if (torch.cuda.is_available() and not (args.no_cuda)) else 'cpu'
    print(f'Device : {device}')

    classes = open(args.classes, 'r').read().splitlines()

    if args.backbone == 'resnet50':
        model = Deeplabv3Resnet50(len(classes)).to(device)
    else:
        model = Deeplabv3Resnet101(len(classes)).to(device)

    model.load_state_dict(torch.load(args.pt))
    model.eval()

    head, tail = os.path.split(args.pt)
    name = tail.split('.')[0]

    # save_path = f'{os.path.normpath(args.save_path)}/deeplabv3_{args.backbone}_{args.size}'
    save_path = os.path.normpath(args.save_path) + '/' + name + '.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size).to(device), save_path, opset_version=11)
    print(f'ONNX model saved')
