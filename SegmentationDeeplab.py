import os

import torch
import torchvision
import numpy as np

from PIL import Image


class SegmentationDeeplab:
    def __init__(self, model_path, classes_path, size=300, device='cpu'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Invalid directory {model_path}')

        self.classes = open(classes_path, 'r').read().splitlines()

        assert len(self.classes) > 0, 'Classes load error'

        self.size = size
        self.device = device

        self.net = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=len(self.classes)).to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()

    def __call__(self, image, threshold=0.5):
        h, w, _ = image.shape
        img = Image.fromarray(image)
        img = torchvision.transforms.Resize((self.size, self.size))(img)
        img = torchvision.transforms.ToTensor()(img)
        img = img.to(self.device)

        out = torch.sigmoid(self.net(img.unsqueeze(0))['out']).squeeze(0)

        result = []
        for i in range(out.shape[0]):
            tmp = torchvision.transforms.ToPILImage()(out[i].detach().cpu())
            tmp = torchvision.transforms.Resize((h, w))(tmp)
            tmp = np.array(tmp, dtype=np.uint8)
            tmp[tmp >= threshold] = 255
            tmp[tmp < threshold] = 0

            result.append(tmp)

        return result

    def get_class(self, index):
        return self.classes[index]