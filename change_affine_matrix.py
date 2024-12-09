import nibabel as nib
import os

base_dir = ''
nii_dir = ''

file_names = [f for f in os.listdir(nii_dir) if f.endswith('.nii.gz') and f.startswith('case_')]

for file_name in file_names:
    file_path = os.path.join(nii_dir, file_name)

    image = nib.load(file_path)

    imaging_data = image.get_fdata()

    imaging_path = os.path.join(base_dir, f'{file_name.split(".")[0]}','imaging.nii.gz')

    if os.path.exists(imaging_path):

        imaging_img = nib.load(imaging_path)

        affine_matrix = imaging_img.affine

        new_imaging_img = nib.Nifti1Image(imaging_data, affine_matrix)

        nib.save(new_imaging_img, file_path)

        print(f"Updated affine matrix for {file_path}")

    else:
        print(f"Target imaging file {imaging_path} does not exist.")
