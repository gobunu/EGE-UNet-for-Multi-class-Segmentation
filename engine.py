import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import matplotlib.pyplot as plt
from collections import defaultdict

index_to_color = {
    -1: [0, 0, 0],
    0: [0, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255],
    3: [255, 255, 0],
    4: [0, 255, 255],
}

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    step,
                    logger, 
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
 
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        gt_pre, out = model(images)
        loss = criterion(gt_pre, out, targets)

        loss.backward()
        optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return step


def val_one_epoch(test_loader,
                    model,
                    criterion,
                    epoch,
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    num_classes = config.num_classes
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            gt_pre, out = model(img)

            loss = criterion(gt_pre, out, msk)
            loss_list.append(loss.item())

            msk = msk.cpu().detach().numpy()
            msk = np.argmax(msk, axis=1)

            gts.append(msk)

            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()

            softmax_out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
            out = np.argmax(softmax_out, axis=1)[0]

            threshold = 0.5
            softmax_out = np.max(softmax_out, axis=1)[0]
            out[softmax_out < threshold] = 0

            preds.append(out)

        if epoch % config.val_interval == 0:
            preds = np.array(preds).reshape(-1)
            gts = np.array(gts).reshape(-1)

            confusion_multi = confusion_matrix(gts, preds, labels=range(num_classes))

            accuracy = 0
            sensitivity = []
            specificity = []
            f1_or_dsc = []
            miou = []

            for i in range(num_classes):

                TP = confusion_multi[i, i]
                FP = np.sum(confusion_multi[:, i]) - TP
                FN = np.sum(confusion_multi[i, :]) - TP
                TN = np.sum(confusion_multi) - (TP + FP + FN)

                accuracy += TP + TN
                sensitivity.append(TP / (TP + FN) if (TP + FN) != 0 else 0)
                specificity.append(TN / (TN + FP) if (TN + FP) != 0 else 0)
                f1_or_dsc.append((2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0)
                miou.append(TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0)

            accuracy = accuracy / np.sum(confusion_multi) if np.sum(confusion_multi) != 0 else 0
            mean_sensitivity = np.mean(sensitivity)
            mean_specificity = np.mean(specificity)
            mean_f1_or_dsc = np.mean(f1_or_dsc)
            mean_miou = np.mean(miou)

            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {mean_miou:.4f}, f1_or_dsc: {mean_f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                        specificity: {mean_specificity:.4f}, sensitivity: {mean_sensitivity:.4f}, confusion_matrix: {confusion_multi}'
            print(log_info)
            logger.info(log_info)
        else:
            log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
            print(log_info)
            logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, _ = data
            img = img.cuda(non_blocking=True).float()
            gt_pre, out = model(img)


            if type(out) is tuple:
                out = out[0]
            out = out.cpu().detach().numpy()

            softmax_out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
            out = np.argmax(softmax_out, axis=1)[0]

            threshold = 0.5
            softmax_out = np.max(softmax_out, axis=1)[0]
            out[softmax_out < threshold] = 0
            height, width = out.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)

            for index, color in index_to_color.items():
                color_image[out == index] = color
            out_colors = color_image
            preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(img, None, out_colors, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)


def test_one_epoch_3d(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    outputs_dict = defaultdict(list)
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, img_path = data
            img_name = img_path[0].split('/')[-1].split('_')[1].split('.')[0]
            img_index = img_path[0].split('/')[-1].split('_')[2].split('.')[0]
            img_name = "case_" + img_name
            img = img.cuda(non_blocking=True).float()

            gt_pre, out = model(img)

            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()

            softmax_out = np.exp(out) / np.sum(np.exp(out), axis=1, keepdims=True)
            out = np.argmax(softmax_out, axis=1)[0]

            threshold = 0.5
            softmax_out = np.max(softmax_out, axis=1)[0]
            out[softmax_out < threshold] = 0
            height, width = out.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)

            for index, color in index_to_color.items():
                color_image[out == index] = color
            out_colors = color_image
            preds.append(out)
            outputs_dict[img_name].append((img_index, out))

            if i % config.save_interval == 0:
                save_imgs(img, None, out_colors, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        os.makedirs(os.path.join(config.work_dir, 'nii'), exist_ok=True)
        for case_key, outputs in outputs_dict.items():

            outputs.sort(key=lambda x: x[0])

            case_image_3d = np.stack([output[1] for output in outputs], axis=0)
            case_image_3d = np.array(case_image_3d, dtype=np.float32)

            save_path = os.path.join(config.work_dir, "nii", f"{case_key}.nii.gz")

            nii_image = nib.Nifti1Image(case_image_3d, np.eye(4))

            nib.save(nii_image, save_path)
            print(f"Saved {save_path}")


