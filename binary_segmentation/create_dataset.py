import pickle
import numpy as np
import segmentation_models_pytorch as smp
import os
import shutil
from skimage.io import imread,imsave
import matplotlib
from pathlib import Path
from dataset import Dataset
from augmentations import get_training_augmentation, get_preprocessing, get_validation_augmentation
ENCODER = "resnet34"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)

def save_images_and_masks(dataset,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for i in range(len(dataset)):
        image, mask = dataset[i]

        im=np.transpose(image,(1,2,0))
        im=(im-np.min(im))/(np.max(im)-np.min(im))
        m=np.transpose(mask,(1,2,0))
        m=np.concatenate([m,m,m],axis=-1)
        matplotlib.image.imsave(Path(save_dir,f"{i}.png"), im)
        matplotlib.image.imsave(Path(save_dir,f"{i}_mask.png"), m)
def create_dataset(dir,save_to,printed=False):
    x_train_dir=Path(dir,'train/img')
    y_train_dir=Path(dir,'train/mask')
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    x_valid_dir=Path(dir,'valid/img')
    y_valid_dir=Path(dir,'valid/mask')
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    if printed:
        save_images_and_masks(train_dataset, Path(save_to,'train'))
        save_images_and_masks(valid_dataset, Path(save_to,'valid'))

    with open(save_to + '/train_dataset.pkl', 'wb') as train_dataset_file:
        pickle.dump(train_dataset, train_dataset_file)
    train_dataset_file.close()
    with open(save_to + '/valid_dataset.pkl', 'wb') as valid_dataset_file:
        pickle.dump(valid_dataset, valid_dataset_file)
    valid_dataset_file.close()

if __name__ == '__main__':
    dir=r'A:\pycharm_projects\plates\data'
    save_to=r'A:\pycharm_projects\plates\data\dataset'
    create_dataset(dir,save_to,printed=True)