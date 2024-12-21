import albumentations as albu
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
HIGHT=512
WIDTH=512

def get_training_augmentation():
    train_transform = [
        # albu.Flip(p=0.5),
        # albu.HorizontalFlip(p=HorFlip_p),
        albu.PadIfNeeded(min_height=HIGHT, min_width=WIDTH, always_apply=False,
                         border_mode=cv2.BORDER_REFLECT_101,value=0),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        albu.RandomCrop(height=HIGHT//2, width=WIDTH//2, always_apply=False),
        # albu.RandomScale(),
        # albu.HueSaturationValue(),
        albu.GaussianBlur(blur_limit=3),
        # albu.GridDropout (ratio=0.2, random_offset=True, p=0.5), # ???
        # albu.Downscale(scale_min=1/COEF_MIN , scale_max=1/COEF_MIN , interpolation=0, always_apply=True),

        albu.RandomGridShuffle (grid=(1, 3), always_apply=False, p=0.2),
        albu.Flip(p=0.7)
    ]

    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(HIGHT, WIDTH),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """

    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]

    return albu.Compose(_transform)