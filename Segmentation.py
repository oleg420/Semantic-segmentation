import os

import torch
import torchvision
import cv2
import numpy as np

from UNet.unet_model import UNet


class Segmentation():
    def __init__(self, model_path, classes_path, size=300, device='cpu', type='deeplab'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Invalid directory {model_path}')

        if type not in ['deeplab', 'unet']:
            raise ValueError(f'Unsupported network')

        self.type = type

        classes = np.loadtxt(classes_path, dtype=np.str).reshape(-1, 4)
        self.labels = classes[:, 0]
        self.label_color = classes[:, 1:].astype(np.uint8)

        self.num_classes = self.labels.shape[0]
        self.size = size
        self.device = device

        if self.type == 'deeplab':
            self.net = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes).to(self.device)
        elif self.type == 'unet':
            self.net = UNet(3, self.num_classes).to(self.device)
        else:
            raise ValueError(f'Unsupported network')

        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def detect(self, cv_image, threshold=0.8):
        # prepare image
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))

        image = torch.tensor(image, dtype=torch.float32, device=self.device)
        image = image.permute(2, 0, 1)
        image = torch.div(image, 255)

        # detect
        if self.type == 'deeplab':
            result = torch.sigmoid(self.net(image.unsqueeze(0))['out']).squeeze(0)
        elif self.type == 'unet':
            result = self.net(image.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f'Unsupported network')

        # results
        soft_segm = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)
        hard_segm = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)

        for i, cls in enumerate(self.label_color):
            segm_image_cls = result[i].unsqueeze(2).repeat(1, 1, 3).detach()

            segm_image_cls *= torch.tensor(cls, dtype=torch.float32).to(self.device)
            tmp = torch.clamp(segm_image_cls, 0, 255).type(torch.uint8)
            soft_segm = cv2.addWeighted(soft_segm, 1, tmp.cpu().numpy(), 1, 0)

            segm_image_cls[segm_image_cls < int(255 * threshold)] = 0
            segm_image_cls[segm_image_cls > int(255 * threshold)] = 255
            tmp = torch.clamp(segm_image_cls, 0, 255).type(torch.uint8)
            hard_segm = cv2.addWeighted(hard_segm, 1, tmp.cpu().numpy(), 1, 0)

        return soft_segm, hard_segm

    def detect_prepared(self, image, threshold=0.8):
        image = torch.tensor(image, device=self.device)

        # detect
        if self.type == 'deeplab':
            result = torch.sigmoid(self.net(image.unsqueeze(0))['out']).squeeze(0)
        elif self.type == 'unet':
            result = self.net(image.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f'Unsupported network')

        # results
        soft_segm = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)
        hard_segm = np.zeros((result.shape[1], result.shape[2], 3), dtype=np.uint8)

        for i, cls in enumerate(self.label_color):
            segm_image_cls = result[i].unsqueeze(2).repeat(1, 1, 3).detach()

            segm_image_cls *= torch.tensor(cls, dtype=torch.float32).to(self.device)
            tmp = torch.clamp(segm_image_cls, 0, 255).type(torch.uint8)
            soft_segm = cv2.addWeighted(soft_segm, 1, tmp.cpu().numpy(), 1, 0)

            segm_image_cls[segm_image_cls < int(255 * threshold)] = 0
            segm_image_cls[segm_image_cls > int(255 * threshold)] = 255
            tmp = torch.clamp(segm_image_cls, 0, 255).type(torch.uint8)
            hard_segm = cv2.addWeighted(hard_segm, 1, tmp.cpu().numpy(), 1, 0)

        return soft_segm, hard_segm