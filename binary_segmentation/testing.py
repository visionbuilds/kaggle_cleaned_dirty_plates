import os
import time
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pathlib import Path
import torch.nn as nn
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
def test_data(x_test_dir, y_test_dir,model_path,save_to,threshold=0.5):
    test_dataset_vis = Dataset(x_test_dir, y_test_dir)
    # image, mask = test_dataset_vis[4] # get some sample
    test_dataset = Dataset(x_test_dir, y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        )
    list_test = os.listdir(x_test_dir)
    print(f'Всего тестируется файлов: {len(list_test)}')#\t{list_test}')
    threshold_criterion = 0.1 # If y_true_loss and y_predicted_loss lower than threshold_criterion wa consider that prediction is correct
    threshold_null_mask = 0.01 # Если в реальной маске количество ненулевых пикселей меньше, чем threshold_null_mask*height*width, то считаем маску нулевой

    os.makedirs(save_to,exist_ok=True)
    best_model = torch.load(model_path, map_location=DEVICE)
    for i in range(len(list_test)):

        pred_masks=[]
        print(f'{i + 1}:  \t{list_test[i]}\t')


        start_time = time.time()
        image_vis,true_mask = test_dataset_vis[i]
        image= test_dataset[i][0]  # get some sample
        true_mask = true_mask.squeeze()

        # if image.shape!=(1,HIGHT,WIDHT):
        #     image=cv2.resize(image.squeeze(),(WIDHT,HIGHT))[None,:,:]
        #     true_mask = cv2.resize(true_mask.squeeze(), (WIDHT, HIGHT))

        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        pred_mask = best_model.predict(x_tensor)
        pred_mask = pred_mask.squeeze().cpu().numpy()
        print('np.unique(predicted_mask) ',np.unique(pred_mask))
        pred_mask[pred_mask < threshold] = 0
        print('mean probability {} %'.format(np.mean(pred_mask[pred_mask >= threshold])))
        pred_mask[pred_mask >= threshold] =1

        # вычислим, сколько нужно отрезать по краям (отрезает то, что дополнил albu отзеркаливанием)
        x_diff=pred_mask.shape[1] - image_vis.shape[1]
        y_diff=pred_mask.shape[0] - image_vis.shape[0]
        x = x_diff//2
        y = y_diff//2
        # true_mask = true_mask[y:-y, x:-x] # обрежем лишнее, оставшееся от albu
        # pred_mask = pred_mask[y:-y, x:-x] # обрежем лишнее, оставшееся от albu
        if y>0:
            if y_diff % 2 == 0:
                pred_mask = pred_mask[y:-y, :]
            else:
                pred_mask = pred_mask[y:-y-1, :]
            true_mask = true_mask[y:-y, :]

        if x>0:
            if x_diff % 2 == 0:
                pred_mask = pred_mask[:, x:-x]
            else:
                pred_mask = pred_mask[:, x:-x-1]
            true_mask = true_mask[:, x:-x]
        pred_mask=pred_mask[:,:,None]
        pred_mask*=255
        pred_mask=np.concatenate([pred_mask,pred_mask,pred_mask],axis=-1)
        pred_mask=pred_mask.astype('uint8')
        combined_img=np.concatenate([image_vis,pred_mask],axis=1)
        matplotlib.image.imsave(Path(save_to,list_test[i]),combined_img)


if __name__ == '__main__':
    x_test_dir=r'A:\pycharm_projects\plates\data\valid\img'
    y_test_dir=r'A:\pycharm_projects\plates\data\valid\mask'
    the_best_model=r"A:\pycharm_projects\plates\data\dataset\models\_best_model_loss.pth"
    save_to=r'A:\pycharm_projects\plates\data\result'

    test_data(x_test_dir, y_test_dir, the_best_model,save_to)