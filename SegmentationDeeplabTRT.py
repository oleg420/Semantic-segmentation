import os

import torch
import torchvision
import torch2trt
import numpy as np

from PIL import Image


class SegmentationDeeplabTRT:
    def __init__(self, model_path, classes_path, size=224, fp16=False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Invalid directory {model_path}')

        self.classes = open(classes_path, 'r').read().splitlines()

        assert len(self.classes) > 0, 'Classes load error'

        self.fp16 = fp16
        self.size = size
        self.device = 'cuda'

        self.net = torch2trt.TRTModule()
        self.net.load_state_dict(torch.load(model_path))
        self.net = self.net.eval()
        if self.fp16:
            self.net = self.net.half()

    def __call__(self, image, threshold=0.5):
        h, w, _ = image.shape
        img = Image.fromarray(image)
        img = torchvision.transforms.Resize((self.size, self.size))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = img.to(self.device)
        if self.fp16:
            img = img.half()

        out = self.net(img.unsqueeze(0)).squeeze(0)

        result = []
        for i in range(out.shape[0]):
            tmp = torchvision.transforms.ToPILImage()(out[i].float().detach().cpu())
            tmp = torchvision.transforms.Resize((h, w))(tmp)
            tmp = np.array(tmp, dtype=np.uint8)
            tmp[tmp >= threshold] = 255
            tmp[tmp < threshold] = 0

            result.append(tmp)

        return result

    def get_class(self, index):
        return self.classes[index]