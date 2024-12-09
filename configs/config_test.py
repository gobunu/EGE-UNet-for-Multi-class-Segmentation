from torchvision import transforms
from utils import *

from datetime import datetime


class setting_config:
    """
    the config of training setting.
    """

    network = 'egeunet'
    model_config = {
        'num_classes': 3,
        'input_channels': 3,
        'c_list': [8, 16, 24, 32, 48, 64],
        'bridge': True,
        'gt_ds': True,
    }

    test_path = "/data1/user/gj/ege_unet/data/output_pngs/val/images"
    datasets = 'KiTS19'

    criterion = GT_BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 3
    input_size_h = 512
    input_size_w = 512
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    work_dir = "./results/egeunet_isic17_Sunday_08_December_2024_17h_19m_52s/"

    print_interval = 20
    val_interval = 30
    save_interval = 50
    threshold = 0.5

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])
