import os
import glob
import time
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
    parser.add_argument('-tip', '--train_image_path', type=str, required=True)
    parser.add_argument('-tlp', '--train_label_path', type=str, required=True)
    parser.add_argument('-vip', '--val_image_path', type=str, required=True)
    parser.add_argument('-vlp', '--val_label_path', type=str, required=True)
    parser.add_argument('-cls', '--classes', type=str, required=True)

    parser.add_argument('-e', '--epochs', type=int, default=40)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('-s', '--size', type=int, default=500)

    parser.add_argument('-sp', '--save-path', type=str, default='./')
    parser.add_argument('-se', '--save-every', type=int, default=5)

    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.00146)

    args = parser.parse_args()
    print(args)

    device = 'cuda' if (torch.cuda.is_available() and not (args.no_cuda)) else 'cpu'
    print(f'Device : {device}')

    classes = open(args.classes, 'r').read().splitlines()

    # 0 - road
    # classes = [0]
    # epochs = 10
    # size = 300
    # batch_size = 8
    # save_every = 2

    t = torch.nn.Threshold(0.6, 0)
    bce_loss = torch.nn.BCELoss()

    train_images = glob.glob(os.path.normpath(args.train_image_path) + '/*.jpg')
    train_masks = glob.glob(os.path.normpath(args.train_label_path) + '/*.png')
    train_images.sort()
    train_masks.sort()

    val_images = glob.glob(os.path.normpath(args.val_image_path) + '/*.jpg')
    val_masks = glob.glob(os.path.normpath(args.val_label_path) + '/*.png')
    val_images.sort()
    val_masks.sort()

    train_dataset = SegmentationDataset(train_images, train_masks, classes, args.size, True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    val_dataset = SegmentationDataset(val_images, val_masks, classes, args.size, False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if args.backbone == 'resnet50':
        model = Deeplabv3Resnet50(len(classes)).to(device).train()
    else:
        model = Deeplabv3Resnet101(len(classes)).to(device).train()

    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)

    iou_metric = IoU()
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    f_score_metric = Fscore()

    # train loop
    for epoch in range(args.epochs):
        desc = f'Epoch: {epoch + 1}/{args.epochs}'

        sum_losses = 0
        sum_iou_metric = 0
        sum_accuracy_metric = 0
        sum_precision_metric = 0
        sum_recall_metric = 0
        sum_f_score_metric = 0

        # train
        model.train()
        for images, masks in tqdm(train_dataloader, desc=desc):
            optimizer.zero_grad()

            images = images.to(device)
            masks = masks.to(device)

            results = model(images)
            loss = dice_loss(results, masks)
            loss += (bce_loss(t(results), masks) / len(classes))

            sum_losses += loss.item()

            loss.backward()
            optimizer.step()

        # eval
        desc = f'Eval: {epoch + 1}/{args.epochs}'
        model.eval()
        for images, masks in tqdm(val_dataloader, desc=desc):
            with torch.no_grad():
                images = images.to(device)
                masks = masks.to(device)

                results = model(images)
                results = t(results)

                sum_iou_metric += iou_metric(results, masks).item()
                sum_accuracy_metric += accuracy_metric(results, masks).item()
                sum_precision_metric += precision_metric(results, masks).item()
                sum_recall_metric += recall_metric(results, masks).item()
                sum_f_score_metric += f_score_metric(results, masks).item()

        sum_losses /= len(train_dataloader)

        sum_iou_metric /= len(val_dataloader)
        sum_accuracy_metric /= len(val_dataloader)
        sum_precision_metric /= len(val_dataloader)
        sum_recall_metric /= len(val_dataloader)
        sum_f_score_metric /= len(val_dataloader)

        # show losses and metrics
        print(f'Loss: {round(sum_losses, 5)}')

        print(f'IoU: {round(sum_iou_metric, 3)}')
        print(f'Accuracy: {round(sum_accuracy_metric, 3)}')
        print(f'Precision: {round(sum_precision_metric, 3)}')
        print(f'Recall: {round(sum_recall_metric, 3)}')
        print(f'F score: {round(sum_f_score_metric, 3)}')

        time.sleep(1)

        if (epoch+1) % args.save_every == 0:
            save_path = f'{os.path.normpath(args.save_path)}/fcn_{args.backbone}_{args.size}_ckpt_{epoch+1}'
            # save_path = f'{os.path.normpath(args.save_path)}/deeplabv3_{args.backbone}_{args.size}_ckpt_{epoch+1}'
            torch.save(model.state_dict(), save_path + '.pt')

    # training done, eval
    val_dataset = SegmentationDataset(val_images, val_masks, classes, args.size, False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    sum_losses = 0
    sum_iou_metric = 0
    sum_accuracy_metric = 0
    sum_precision_metric = 0
    sum_recall_metric = 0
    sum_f_score_metric = 0

    desc = f'Final eval'
    model.eval()
    for images, masks in tqdm(val_dataloader, desc=desc):
        with torch.no_grad():
            images = images.to(device)
            masks = masks.to(device)

            results = model(images)

            sum_losses += dice_loss(results, masks)
            sum_losses += (bce_loss(t(results), masks) / len(classes))
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

    save_path = f'{os.path.normpath(args.save_path)}/fcn_{args.backbone}_{args.size}'
    torch.save(model.state_dict(), save_path + '_final.pt')
    torch.onnx.export(model, torch.zeros(1, 3, args.size, args.size).to(device), save_path + '.onnx', opset_version=11)
    print(f'Model saved {args.save_every}')
