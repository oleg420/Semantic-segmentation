import torch
import torchvision


class Deeplabv3Resnet50(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])


class Deeplabv3Resnet101(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])
