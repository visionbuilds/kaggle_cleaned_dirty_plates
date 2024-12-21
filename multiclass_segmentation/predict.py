import os
import time
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
from copy import deepcopy
import torch.nn as nn
import pandas as pd
from skimage.io import imread,imsave
from create_dataset import colored_mask_from_multiclass_mask
from dataset import Dataset
from skimage.transform import resize, downscale_local_mean
import cv2
import matplotlib
from augmentations import get_training_augmentation, get_preprocessing, get_validation_augmentation
ENCODER = "resnet34"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

DEVICE='cuda'

def image_to_tensor(image,augmentation,preprocessing):
    image=augmentation(image=image)['image']
    image=preprocessing(image=image)['image']
    return image

def predict_image(image,best_model,classes):
    if isinstance(image,str):
        image=np.asarray(imread(image))
    image_vis=deepcopy(image)
    # best_model = torch.load(model_path, map_location=DEVICE)
    image = image_to_tensor(image,get_validation_augmentation(),get_preprocessing(preprocessing_fn))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pred_mask = best_model.predict(x_tensor)
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = np.argmax(pred_mask, axis=0)
    pred_mask = Dataset.create_multiclass_mask(pred_mask, len(classes))
    colours = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 0, 0)}
    pred_mask = np.transpose(pred_mask, (2, 0, 1))
    pred_mask = colored_mask_from_multiclass_mask(pred_mask, colours)
    # вычислим, сколько нужно отрезать по краям (отрезает то, что дополнил albu отзеркаливанием)
    x_diff = pred_mask.shape[1] - image_vis.shape[1]
    y_diff = pred_mask.shape[0] - image_vis.shape[0]
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
    # combined_img = np.concatenate([image_vis, pred_mask], axis=1)
    return pred_mask

def image_files_to_multiclass_masks(dir,model_path,classes,save_to):
    best_model = torch.load(model_path, map_location=DEVICE)
    for file in os.scandir(dir):
        image = np.asarray(imread(file.path))
        pred_mask = predict_image(image,best_model,classes)
        count_pixels_dirty=np.sum(pred_mask[...,0])
        count_pixels_clean=np.sum(pred_mask[...,1])
        proportion_of_dirty = count_pixels_dirty/(count_pixels_dirty+count_pixels_clean)
        combined_img = np.concatenate([image, pred_mask], axis=1)
        anno_text = f"dirty_ratio = {round(proportion_of_dirty, 2)}"
        cv2.putText(combined_img, anno_text, (combined_img.shape[1] // 2-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
        if proportion_of_dirty > 0.3:
            matplotlib.image.imsave(Path(save_to,'dirty', file.name), combined_img)
        else:
            matplotlib.image.imsave(Path(save_to, 'clean', file.name), combined_img)
if __name__ == '__main__':
    classes = ['background', 'clean', 'dirty']
    img_dir=r"A:\pycharm_projects\plates\data\test"
    model_path = r"A:\pycharm_projects\plates\data\dataset_multiclass\models\_best_model_loss.pth"
    save_to=r'A:\pycharm_projects\plates\data\result_test2'
    image_files_to_multiclass_masks(img_dir,model_path,classes,save_to)