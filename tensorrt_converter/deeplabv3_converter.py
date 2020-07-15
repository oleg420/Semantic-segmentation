import argparse

import torch
import torchvision
import torch2trt


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x)['out'])
        # return self.model(x)['out']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pytorch_model', type=str, required=True)
    parser.add_argument('-cls', '--classes', type=str, required=True)
    parser.add_argument('-s', '--size', type=int, default=224)
    parser.add_argument('-fp16', '--fp16', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    num_classes = len(open(args.classes, 'r').read().splitlines())
    model = torchvision.models.segmentation.deeplabv3_resnet50(num_classes=num_classes)
    model.load_state_dict(torch.load(args.pytorch_model))
    model = model.to(device).eval()

    model_w = ModelWrapper(model)
    model_w = model_w.to(device)

    x = torch.zeros((1, 3, args.size, args.size)).to(device)

    fp = 'fp32'
    if args.fp16:
        model = model.half()
        model_w = model_w.half()
        x = x.half()
        fp = 'fp16'

    print('Converting...')
    trt_model = torch2trt.torch2trt(model_w, [x], fp16_mode=True if fp == 'fp16' else False)
    torch.save(trt_model.state_dict(), f'deeplabv3_{args.size}_{fp}_trt.pth')
    print('Done')
