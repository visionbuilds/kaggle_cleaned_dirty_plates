import os
import time
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
import torch.nn as nn
from skimage.io import imread,imsave
import pandas as pd
from dataset import Dataset
from skimage.transform import resize, downscale_local_mean
import cv2
import matplotlib
from augmentations import get_training_augmentation, get_preprocessing, get_validation_augmentation
ENCODER = "resnet34"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
HIGHT=512
WIDHT=512
DEVICE='cuda'


class Mask_predictior:
    def __init__(self,model_path,DEVICE='cuda'):
        self.device=DEVICE
        self.augmentation = get_validation_augmentation()
        self.preprocessing = get_preprocessing(preprocessing_fn)
        self.model = torch.load(model_path, map_location=self.device)


    @staticmethod
    def crop_padded_mask(pred_mask,width_initial,height_initial):
        x_diff = pred_mask.shape[1] - width_initial
        y_diff = pred_mask.shape[0] - height_initial
        x = x_diff // 2
        y = y_diff // 2
        if y > 0:
            if y_diff % 2 == 0:
                pred_mask = pred_mask[y:-y, :]
            else:
                pred_mask = pred_mask[y:-y - 1, :]

        if x > 0:
            if x_diff % 2 == 0:
                pred_mask = pred_mask[:, x:-x]
            else:
                pred_mask = pred_mask[:, x:-x - 1]
        return pred_mask


    def predict_mask(self,image):


        height_initial, width_initial = image.shape[:2]
        image = self.augmentation(image=image)['image']
        image =self.preprocessing(image=image)['image']
        x_tensor = torch.from_numpy(image).to(self.device).unsqueeze(0)
        pred_mask = self.model.predict(x_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        threshold = 0.5
        pred_mask[pred_mask < threshold] = 0
        pred_mask[pred_mask >= threshold] = 1
        pred_mask=self.crop_padded_mask(pred_mask,width_initial,height_initial)
        pred_mask = pred_mask[:, :, None]
        pred_mask *= 255
        pred_mask = np.concatenate([pred_mask, pred_mask, pred_mask], axis=-1)
        pred_mask = pred_mask.astype('uint8')
        return pred_mask

    def create_masks_from_image_dir(self,img_dir,save_to,combine_img_and_mask=False):

        for file in os.scandir(img_dir):
            image = np.asarray(imread(file.path))
            pred_mask = self.predict_mask(image)
            if combine_img_and_mask:
                combined_img = np.concatenate([image, pred_mask], axis=1)
                matplotlib.image.imsave(Path(save_to,file.name),combined_img)
            else:
                matplotlib.image.imsave(Path(save_to, file.name), pred_mask)


if __name__ == '__main__':
    # img_path=r"A:\pycharm_projects\plates\data\valid\img\2ad0c6084924b80fa43016d910e05918.png"
    # model_path = r"A:\pycharm_projects\plates\data\dataset\models\_best_model_loss.pth"
    # save_to=r'A:\pycharm_projects\plates\data\result\a.png'
    # predict_mask(img_path, model_path, save_to)

    model_path = r"A:\pycharm_projects\plates\data\dataset\models\_best_model_loss.pth"
    mask_predictior = Mask_predictior(model_path=model_path,DEVICE='cuda')
    # img_dir=r'A:\pycharm_projects\plates\data\valid\img'
    img_dir=r'A:\pycharm_projects\plates\data\test'
    save_to=r'A:\pycharm_projects\plates\data\result'
    mask_predictior.create_masks_from_image_dir(img_dir, save_to, combine_img_and_mask=True)