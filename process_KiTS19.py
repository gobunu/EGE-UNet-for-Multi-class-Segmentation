import os
import numpy as np
import nibabel as nib
from PIL import Image
import random
import shutil
from tqdm import tqdm

root_dir = ''
output_dir = ''
os.makedirs(output_dir, exist_ok=True)

case_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

random.seed(42)
selected_cases = random.sample(case_dirs, 10)

train_images_dir = os.path.join(output_dir, 'train', 'images')
train_masks_dir = os.path.join(output_dir, 'train', 'masks')
val_images_dir = os.path.join(output_dir, 'val', 'images')
val_masks_dir = os.path.join(output_dir, 'val', 'masks')

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_masks_dir, exist_ok=True)

for case_dir in tqdm(case_dirs):

    case_imaging_path = os.path.join(root_dir, case_dir, 'imaging.nii.gz')
    case_segmentation_path = os.path.join(root_dir, case_dir, 'segmentation.nii.gz')
    output_case_dir = os.path.join(output_dir, case_dir)
    os.makedirs(output_case_dir, exist_ok=True)

    if os.path.exists(case_imaging_path) and os.path.exists(case_segmentation_path):
        imaging_img = nib.load(case_imaging_path)
        segmentation_img = nib.load(case_segmentation_path)

        imaging_data = imaging_img.get_fdata()
        segmentation_data = segmentation_img.get_fdata()

        imaging_data = (imaging_data - np.min(imaging_data)) / (np.max(imaging_data) - np.min(imaging_data)) * 255
        imaging_data = np.uint8(imaging_data)

        z_dim = segmentation_data.shape[0]

        for z in range(z_dim):
            imaging_slice = imaging_data[z, :, :]
            segmentation_slice = segmentation_data[z, :, :]

            slice_name = f'{case_dir}_{z:04d}.png'
            imaging_image = Image.fromarray(imaging_slice)
            segmentation_image = Image.fromarray(np.uint8(segmentation_slice))

            if case_dir in selected_cases:
                imaging_image.save(os.path.join(val_images_dir, slice_name))
                segmentation_image.save(os.path.join(val_masks_dir, slice_name))
            else:
                imaging_image.save(os.path.join(train_images_dir, slice_name))
                segmentation_image.save(os.path.join(train_masks_dir, slice_name))

print("Finished processing and saving images for all cases.")
