import os
import torch

import torch.onnx
from models.egeunet import EGEUNet

from engine import *

import sys

from utils import *
from configs.config_test import setting_config

import warnings
warnings.filterwarnings("ignore")

config = setting_config

class inference_unet(nn.Module):
    def __init__(self, config):
        super(inference_unet, self).__init__()
        model_cfg = config.model_config
        self.model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
        best_weight = torch.load(os.path.join(config.work_dir, 'checkpoints', 'best.pth'), map_location=torch.device('cpu'))
        self.model.load_state_dict(best_weight)

    def forward(self, x):
        _, output = self.model(x)
        return output

model = inference_unet(config)
x = torch.randn(1, config.input_channels, config.input_size_w, config.input_size_h)

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        "ege_unet.onnx",
        opset_version=11,
        input_names=['input'],
        output_names=['output'])
