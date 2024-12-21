import pickle
import numpy as np
import segmentation_models_pytorch as smp
import os
import shutil
from skimage.io import imread,imsave
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
from dataset import Dataset
from augmentations import get_training_augmentation, get_preprocessing, get_validation_augmentation
ENCODER = "resnet34"
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER)
classes=['background','clean','dirty']
colours={0:(0,0,0),1:(0,255,0),2:(255,0,0)}

def colored_mask_from_multiclass_mask(mask,colours):
    colored_mask = np.zeros((mask.shape[1],mask.shape[2], 3), dtype=np.uint8)
    for i, color in colours.items():
        colored_mask[mask[i, ...] == 1] = color
    return colored_mask
def save_images_and_masks(dataset,save_dir):
    os.makedirs(save_dir,exist_ok=True)
    for i in range(len(dataset)):
        image, mask = dataset[i]

        im=np.transpose(image,(1,2,0))
        im=(im-np.min(im))/(np.max(im)-np.min(im))
        mask = colored_mask_from_multiclass_mask(mask,colours)
        matplotlib.image.imsave(Path(save_dir,f"{i}.png"), im)
        matplotlib.image.imsave(Path(save_dir,f"{i}_mask.png"), mask)
def create_dataset(dir,save_to,printed=False):
    train_dir=Path(dir,'train')
    train_dataset = Dataset(
        train_dir,
        classes=classes,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    valid_dir=Path(dir,'valid')
    valid_dataset = Dataset(
        valid_dir,
        classes=classes,
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
    dir=r'A:\pycharm_projects\plates\plates.v3i.png-mask-semantic'
    save_to=r'A:\pycharm_projects\plates\data\dataset_multiclass'
    create_dataset(dir,save_to,printed=True)