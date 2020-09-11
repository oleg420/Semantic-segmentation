import os
import glob
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models import Deeplabv3Resnet50, Deeplabv3Resnet101
from utils.dataset import SegmentationDataset

from utils.losses import DiceLoss
from utils.metrics import IoU, Accuracy, Precision, Recall, Fscore


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pt', type=str, required=True)
    parser.add_argument('-vip', '--val_image_path', type=str, required=True)
    parser.add_argument('-vlp', '--val_label_path', type=str, required=True)
    parser.add_argument('-cls', '--classes', type=str, required=True)

    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-s', '--size', type=int, default=300)

    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()
    print(args)

    device = 'cuda' if (torch.cuda.is_available() and not (args.no_cuda)) else 'cpu'
    print(f'Device : {device}')

    classes = open(args.classes, 'r').read().splitlines()

    val_images = glob.glob(os.path.normpath(args.val_image_path) + '/*.jpg')
    val_masks = glob.glob(os.path.normpath(args.val_label_path) + '/*.png')
    val_images.sort()
    val_masks.sort()

    if args.backbone == 'resnet50':
        model = Deeplabv3Resnet50(len(classes)).to(device)
    else:
        model = Deeplabv3Resnet101(len(classes)).to(device)

    model.load_state_dict(torch.load(args.pt))
    model = model.eval()

    dice_loss = DiceLoss()

    iou_metric = IoU()
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    f_score_metric = Fscore()

    val_dataset = SegmentationDataset(val_images, val_masks, classes, args.size, False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    sum_losses = 0
    sum_iou_metric = 0
    sum_accuracy_metric = 0
    sum_precision_metric = 0
    sum_recall_metric = 0
    sum_f_score_metric = 0

    desc = f'Eval'
    for images, masks in tqdm(val_dataloader, desc=desc):
        with torch.no_grad():
            images = images.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast():
                results = model(images)

            sum_losses += dice_loss(results, masks)
            sum_iou_metric += iou_metric(results, masks).item()
            sum_accuracy_metric += accuracy_metric(results, masks).item()
            sum_precision_metric += precision_metric(results, masks).item()
            sum_recall_metric += recall_metric(results, masks).item()
            sum_f_score_metric += f_score_metric(results, masks).item()

    sum_losses /= len(val_dataloader)
    sum_iou_metric /= len(val_dataloader)
    sum_accuracy_metric /= len(val_dataloader)
    sum_precision_metric /= len(val_dataloader)
    sum_recall_metric /= len(val_dataloader)
    sum_f_score_metric /= len(val_dataloader)

    print(f'Final results')
    print(f'Loss: {round(sum_losses, 5)}')
    print(f'IoU: {round(sum_iou_metric, 3)}')
    print(f'Accuracy: {round(sum_accuracy_metric, 3)}')
    print(f'Precision: {round(sum_precision_metric, 3)}')
    print(f'Recall: {round(sum_recall_metric, 3)}')
    print(f'F score: {round(sum_f_score_metric, 3)}')
