import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

import albumentations as albu


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes, size=224, train=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.size = size
        self.classes = classes

        self.train = train

        __preprocess = [
            albu.Lambda(image=self.__div),
            albu.Lambda(image=self.__transpose, mask=self.__transpose),
        ]

        self.preprocess = albu.Compose(__preprocess)

    def __getitem__(self, index):
        # load image
        image = cv2.imread(self.images_dir[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load mask
        mask_image = cv2.imread(self.masks_dir[index], cv2.IMREAD_GRAYSCALE)

        # resize if image and mask are not for training
        if not self.train:
            image = cv2.resize(image, (self.size, self.size))
            mask_image = cv2.resize(mask_image, (self.size, self.size))

        # generate output data
        mask = np.zeros((len(self.classes), mask_image.shape[0], mask_image.shape[1]), dtype=np.float32)
        for cls in self.classes:
            mask[self.classes.index(cls)][mask_image == cls] = 1
        mask = self.__transpose_mask(mask)

        if self.train:
            sample = self.__get_transform()(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        sample = self.preprocess(image=image, mask=mask)
        image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_dir)

    def __transpose_mask(self, x, **kwargs):
        return x.transpose(1, 2, 0)

    def __div(self, x, **kwargs):
        x = np.true_divide(x, 255)
        return x

    def __transpose(self, x, **kwargs):
        x = x.transpose(2, 0, 1)
        return torch.tensor(x, dtype=torch.float32)

    def __get_transform(self):
        transform = [
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            albu.PadIfNeeded(min_height=self.size, min_width=self.size, always_apply=True, border_mode=0),
            albu.RandomCrop(height=self.size, width=self.size, always_apply=True),
            albu.IAAAdditiveGaussianNoise(p=0.2),
            albu.IAAPerspective(p=0.5)
        ]

        return albu.Compose(transform)
