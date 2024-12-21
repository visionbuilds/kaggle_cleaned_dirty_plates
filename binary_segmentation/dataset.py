import os
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from skimage.io import imread,imsave

class Dataset(BaseDataset):



    def __init__(
            self,
            images_dir,
            masks_dir=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.classes = ['plate']
        self.ids_img = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids_img]
        self.ids_mask = os.listdir(masks_dir)
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_mask]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):


        image = np.asarray(imread(self.images_fps[i])) if type(self.images_fps[i]) == str else self.images_fps[0]
        # print(f'\ndataset 48: {self.images_fps[i]}\t{image.shape}, max color: {np.max(image)}')


        # image = np.reshape(image, (image.shape[0], image.shape[1], 1)) # (1278, 7087, 1)
        # image = np.float32(image/(np.max(image)+0.001))

        mask = np.asarray(imread(self.masks_fps[i])) if type(self.masks_fps[i]) == str else self.masks_fps[0]
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        mask=mask/255
        # apply augmentations
        if self.augmentation:
            # print(f'dataset np.max(image) 77:\t{np.max(image)}, shape: {image.shape}, type: {type(image[0][0][0])}')
            # print(f'dataset np.max(mask) 78:\t{np.max(mask)}, shape: {mask.shape}, type: {type(mask[0][0][0])}')
            sample = self.augmentation(image=image, mask=mask.astype(int))
            # print('sample.keys() ',sample.keys())
            image, mask = sample['image'], sample['mask']
            # print(f'dataset augmentation 81: \timage: {image.shape}, {np.sum(image)}, \tmask: {mask.shape}, {np.sum(mask)}')

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)
        # return len(self.ids_img)