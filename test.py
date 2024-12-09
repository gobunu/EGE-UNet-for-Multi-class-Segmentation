import os
import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import test_datasets
from models.egeunet import EGEUNet
import shutil

from engine import *

import sys

from utils import *
from configs.config_test import setting_config

import warnings
warnings.filterwarnings("ignore")



def main(config):

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()


    print('#----------Preparing dataset----------#')
    test_dataset = test_datasets(config.test_path, config, return_path=True)
    test_loader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=config.num_workers,
                                drop_last=True)


    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'egeunet':
        model = EGEUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        bridge=model_cfg['bridge'],
                        gt_ds=model_cfg['gt_ds'],
                        )
    else: raise Exception('network in not right!')
    model = model.cuda()


    print('#----------Prepareing loss, opt, sch and amp----------#')

    best_weight_dir = os.path.join(config.work_dir, 'checkpoints/best.pth')
    if os.path.exists(best_weight_dir):
        print('#----------Testing----------#')
        best_weight = torch.load(best_weight_dir, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch_3d(
                test_loader,
                model,
                None,
                None,
                config,
            )
    else:
        print("Warning: no checkpoint found at '{}'".format(best_weight_dir))


if __name__ == '__main__':
    config = setting_config
    main(config)