import os
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from skimage.io import imread,imsave

class Dataset(BaseDataset):



    def __init__(
            self,
            dir,
            classes,
            augmentation=None,
            preprocessing=None,
    ):
        self.dir=dir
        self.classes = classes
        self.ids_img = [name for name in os.listdir(self.dir) if '_mask' not in name and
                        np.any([name.endswith(i) for i in ('jpg','png')])]
        self.images_fps = [os.path.join(dir, image_id) for image_id in self.ids_img]
        # self.ids_mask = os.listdir(masks_dir)
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_mask]
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    @staticmethod
    def create_multiclass_mask(input_array, n_classes=None):
        """
        Creates a multi-class segmentation mask from a single-channel array with 'n' unique values.

        Args:
          input_array: A numpy array with values 0, 1, ..., n-1.

        Returns:
          A numpy array with 'n-1' channels, where each channel represents a class (0 or 1)
          for the corresponding value in the input array.
        """
        if n_classes == None:
            n_classes = len(np.unique(input_array))
        mask = np.zeros((input_array.shape[0], input_array.shape[1], n_classes), dtype=np.uint8)

        for i in range(n_classes):
            mask[:, :, i] = (input_array == i).astype(np.uint8)

        return mask

    def __getitem__(self, i):


        image = np.asarray(imread(self.images_fps[i])) if type(self.images_fps[i]) == str else self.images_fps[0]
        # print(f'\ndataset 48: {self.images_fps[i]}\t{image.shape}, max color: {np.max(image)}')

        name = os.path.split(self.images_fps[i])[-1].split('.')[-2]
        mask_path = [f.path for f in os.scandir(self.dir) if name in f.name and '_mask' in f.name][0]

        mask = np.asarray(imread(mask_path))
        mask = self.create_multiclass_mask(mask,len(self.classes))
        if not image.shape[:2]==mask.shape[:2]:
            print(self.images_fps[i],'image and mask sizes are not equal')
        if self.augmentation:
            # try:
            sample = self.augmentation(image=image, mask=mask)
            # except:
            #     pass
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)

            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)
        # return len(self.ids_img)