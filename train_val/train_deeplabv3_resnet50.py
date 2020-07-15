import glob
import time

from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader

from utils.dataset import SegmentationDataset

from utils.losses import DiceLoss
from utils.metrics import IoU, Accuracy, Precision, Recall, Fscore


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = [0, 13]
    epochs = 5
    size = 400
    batch_size = 8
    save_every = 1

    train_images = glob.glob('/media/oleg/WD/datasets/bdd100k_seg/seg/images/train/*.jpg')
    train_masks = glob.glob('/media/oleg/WD/datasets/bdd100k_seg/seg/labels/train/*.png')
    train_images.sort()
    train_masks.sort()

    val_images = glob.glob('/media/oleg/WD/datasets/bdd100k_seg/seg/images/val/*.jpg')
    val_masks = glob.glob('/media/oleg/WD/datasets/bdd100k_seg/seg/labels/val/*.png')
    val_images.sort()
    val_masks.sort()

    train_dataset = SegmentationDataset(train_images, train_masks, classes, size, True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = SegmentationDataset(val_images, val_masks, classes, size, False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=len(classes)).to(device)

    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

    iou_metric = IoU()
    accuracy_metric = Accuracy()
    precision_metric = Precision()
    recall_metric = Recall()
    f_score_metric = Fscore()

    for epoch in range(epochs):
        desc = f'Epoch: {epoch + 1}/{epochs}'
        sum_losses = 0
        sum_iou_metric = 0
        sum_accuracy_metric = 0
        sum_precision_metric = 0
        sum_recall_metric = 0
        sum_f_score_metric = 0

        model.train()
        for images, masks in tqdm(train_dataloader, desc=desc):
            optimizer.zero_grad()

            images = images.to(device)
            masks = masks.to(device)

            results = torch.sigmoid(model(images)['out'])

            loss = dice_loss(results, masks)
            sum_losses += loss.item()

            loss.backward()
            optimizer.step()

        desc = f'Eval: {epoch + 1}/{epochs}'
        model.eval()
        for images, masks in tqdm(val_dataloader, desc=desc):
            with torch.no_grad():
                images = images.to(device)
                masks = masks.to(device)

                results = torch.sigmoid(model(images)['out'])

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

        print(f'Loss: {round(sum_losses, 5)}')

        print(f'IoU: {round(sum_iou_metric, 3)}')
        print(f'Accuracy: {round(sum_accuracy_metric, 3)}')
        print(f'Precision: {round(sum_precision_metric, 3)}')
        print(f'Recall: {round(sum_recall_metric, 3)}')
        print(f'F score: {round(sum_f_score_metric, 3)}')

        time.sleep(1)

        if (epoch+1) % save_every == 0:
            save_path = f'deeplabv3_resnet50_{size}_ckpt_{epoch+1}.pt'
            torch.save(model.state_dict(), save_path)

    # done training, eval
    val_dataset = SegmentationDataset(val_images, val_masks, classes, size, False)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

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

            results = torch.sigmoid(model(images)['out'])

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

    save_path = f'deeplabv3_resnet50_{size}_final.pt'
    torch.save(model.state_dict(), save_path)
    print(f'Model saved {save_every}')
