import cv2
import numpy as np
from PIL import Image

import torch
import torchvision

from torch.utils.data import Dataset


class DLDataset(Dataset):
    def __init__(self, images, seg_images, classes):
        self.images = images
        self.seg_images = seg_images
        self.classes = classes
        self.num_classes = len(classes)

    def __getitem__(self, index):
        img = torchvision.transforms.ToTensor()(
            Image.open(self.images[index]).convert('RGB').resize((300, 300), Image.ANTIALIAS))

        seg_img = cv2.imread(self.seg_images[index], cv2.IMREAD_GRAYSCALE)
        seg_img = cv2.resize(seg_img, (300, 300))

        squeeze_img = np.zeros(shape=(self.num_classes + 1, seg_img.shape[0], seg_img.shape[1]), dtype=seg_img.dtype)

        for i, cls in enumerate(self.classes):
            squeeze_img[i, np.where(seg_img == cls)[0], np.where(seg_img == cls)[1]] = 1
        squeeze_img = torch.tensor(squeeze_img, dtype=torch.float32)

        return img, squeeze_img

    def __len__(self):
        return len(self.images)