import os
import shutil
from pathlib import Path
import numpy as np
from skimage.io import imread,imsave

def transfrom_roboflow_data_to_segmentation_models_pytorch_data(dir,save_to):
    os.makedirs(save_to, exist_ok=True)
    save_img=Path(save_to,'img')
    save_mask=Path(save_to,'mask')
    os.makedirs(save_img, exist_ok=True)
    os.makedirs(save_mask, exist_ok=True)
    # unique_names=[f.split('_')[0].split('.')[-1] for f in os.listdir(dir)]
    # for name in unique_names:
    #     img_path=Path(dir,[f for f in os.scandir(dir) if ])
    for f in os.scandir(dir):
        if not np.any([f.name.lower().endswith(i) for i in ['jpg','png']]):
            continue
        if 'mask' in f.name:
            new_name=f"{f.name.split('_mask')[0].split('.')[-1]}.png"
            img=imread(f.path)
            img[img>1]=1
            img *= 255
            imsave(Path(save_mask, new_name), img)
        else:
            new_name=f"{f.name.split('.')[-2]}.png"
            shutil.copy(f.path,Path(save_img,new_name))

def transfrom_roboflow_data_to_segmentation_models_pytorch_data_for_train_and_valid(dir,save_to):

    save_train = Path(save_to,'train')
    save_valid = Path(save_to,'valid')
    # os.makedirs(save_train,exist_ok=True)
    # os.makedirs(save_valid,exist_ok=True)
    transfrom_roboflow_data_to_segmentation_models_pytorch_data(Path(dir,'train'), save_train)
    transfrom_roboflow_data_to_segmentation_models_pytorch_data(Path(dir,'valid'), save_valid)

if __name__ == '__main__':
    dir = r'A:\pycharm_projects\plates\plates.v2i.png-mask-semantic'
    save_to = r'A:\pycharm_projects\plates\data'
    transfrom_roboflow_data_to_segmentation_models_pytorch_data_for_train_and_valid(dir, save_to)