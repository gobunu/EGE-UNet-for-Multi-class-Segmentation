import time

from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self).__init__()
        self.config = config
        if train:
            images_list = os.listdir(os.path.join(path_Data, 'train', 'images'))
            masks_list = os.listdir(os.path.join(path_Data, 'train', 'masks'))
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'train', 'images', images_list[i])
                mask_path = os.path.join(path_Data, 'train', 'masks', masks_list[i])
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer
        else:
            images_list = os.listdir(os.path.join(path_Data, 'val', 'images'))
            masks_list = os.listdir(os.path.join(path_Data, 'val', 'masks'))
            images_list = sorted(images_list)
            masks_list = sorted(masks_list)
            self.data = []
            for i in range(len(images_list)):
                img_path = os.path.join(path_Data, 'val', 'images', images_list[i])
                mask_path = os.path.join(path_Data, 'val', 'masks', masks_list[i])
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def __getitem__(self, indx):
        num_classes = self.config.num_classes
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2)
        msk = np.eye(num_classes)[msk.astype(int).squeeze()]
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)


class test_datasets(Dataset):
    def __init__(self, test_path, config, return_path=False):
        super(test_datasets, self)
        self.config = config
        self.return_path = return_path

        images_list = os.listdir(test_path)
        images_list = sorted(images_list)
        self.data = []
        for i in range(len(images_list)):
            img_path = os.path.join(test_path, images_list[i])
            self.data.append(img_path)
        self.transformer = config.test_transformer

    def __getitem__(self, indx):
        img_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB'))
        img, _ = self.transformer((img, None))
        if self.return_path:
            return img, img_path
        else:
            return img

    def __len__(self):
        return len(self.data)
    
