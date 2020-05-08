import os

import torch
import torchvision

from PIL import Image

import cv2
import numpy as np

class Segmentation():
    def __init__(self, model_path, classes_path, size=300, device='cpu'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Invalid directory {model_path}')

        classes = np.loadtxt(classes_path, dtype=np.str).reshape(-1, 4)
        self.labels = classes[:, 0]
        self.label_color = classes[:, 1:].astype(np.uint8)

        self.num_classes = self.labels.shape[0]
        self.size = size
        self.device = device

        self.net = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=self.num_classes+1).to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def detect(self, image, threshold=0.5):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torchvision.transforms.ToTensor()(image)
        image = image.to(self.device)

        result = self.net(image.unsqueeze(0))['out'].squeeze(0)

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


