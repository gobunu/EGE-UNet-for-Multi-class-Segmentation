
# EGE-UNet for Multi class Segmentation
This version of EGE-UNet is a modification of the [original EGE-UNet model](https://github.com/JCruan519/EGE-UNet?tab=Apache-2.0-1-ov-file). It is not the official implementation, but aims to provide similar functionality with some enhancements and adjustments.

## 0. Model Structure Modifications
To accommodate multi-class segmentation tasks, I have modified the skip connection part of the model. You can find the relevant code [here](https://github.com/gobunu/EGE-UNet-for-Multi-class-Segmentation/blob/main/models/egeunet.py#L195-L204). Additionally, the forward function of the model, for example, can be found [here](https://github.com/gobunu/EGE-UNet-for-Multi-class-Segmentation/blob/main/models/egeunet.py#L275-L277), as well as in other similar parts of the code.


## 1. Main Environments
- python 3.8
- [pytorch 1.8.0](https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-win_amd64.whl)
- [torchvision 0.9.0](https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl)

## 2. Prepare the dataset.

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- The KiTS19 dataset has been processed by slicing and normalizing along the 0th dimension to generate 512x512 PNG images. Additionally, 10 random 3D images have been selected for validation. The original dataset can be accessed [here](https://github.com/neheller/kits19). To process the raw `.nii.gz` files into PNG format, you can use the provided `process_KiTS19.py` script.

- After downloading the datasets, you are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

### ./data/isic17/
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
### ./data/KiTS19/
  - train
      - images
        - case_xxxxx_yyyy.png
      - masks
        - case_xxxxx_yyyy.png
  - val
    - images
      - case_xxxxx_yyyy.png
    - masks
      - case_xxxxx_yyyy.png
      
### 3. Train the EGE-UNet.
```
cd EGE-UNet
```
```
python train.py
```

## 4. Obtain the outputs.
- After trianing, you could obtain the outputs in `./results/`
- To obtain the predicted 3D masks for the KiTS19 dataset, use the `test_one_epoch_3d` class in the `test.py` script for inference.After that, run `change_affine_matrix.py` to apply the correct affine matrix to the predicted masks.

## 5. Deployment.
- After running onnx_unet.py, you can run inference_unet.py with your ONNX model. This will open a simple GUI.
- With the package PyInstaller, you can **package the program into an executable file**.
