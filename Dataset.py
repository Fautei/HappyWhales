from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
import albumentations
import json
from PIL import Image
import os
import torch
import cv2
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, df, path, augment = True, image_size = 512):
        self.df = df
        self.path = path
        self.image_size = image_size
        if augment:
            self.transforms = albumentations.Compose([
                    albumentations.augmentations.geometric.resize.Resize(image_size, image_size, interpolation=1, always_apply=True, p=1),
                    albumentations.augmentations.transforms.HorizontalFlip(p=0.5),

                    albumentations.core.composition.OneOf([
                    albumentations.augmentations.transforms.RGBShift(),
                    albumentations.augmentations.transforms.RandomGamma(),
                    albumentations.augmentations.transforms.Sharpen(),
                    albumentations.augmentations.transforms.CLAHE(),
                    albumentations.augmentations.transforms.HueSaturationValue(),
                    albumentations.Cutout(num_holes=1, max_h_size=(image_size//9), max_w_size=(image_size//9)),
                    albumentations.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=(-0.2, 0.2)),], p = 0.5),
                
                    albumentations.core.composition.OneOf([
                    albumentations.augmentations.transforms.RandomSunFlare(),
                    albumentations.augmentations.transforms.RandomFog(),
                    albumentations.augmentations.transforms.RandomRain(),
                    albumentations.augmentations.transforms.RandomSnow(),], p = 0.5),
                
                    albumentations.Normalize(),
                    ToTensorV2(p=1.0)
                ])
        else:
            self.transforms = albumentations.Compose([
                albumentations.augmentations.transforms.HorizontalFlip(p=0.5),
                albumentations.Normalize(),
                ToTensorV2(p=1.0)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.image.values[idx]
        # read image 
        image_path = os.path.join(self.path, image_name)
        image = cv2.imread(image_path)[:, :, ::-1]
        if image.shape[0] < self.image_size or image.shape[1] < self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size), cv2.INTER_CUBIC)
        
        # fetch and encode label
        label = self.df["individual_id_integer"].iloc[idx]
        label = torch.tensor(label, dtype=torch.long)
        image = self.transforms(image = image)["image"]
        return image, label

    def plot_sample(self, idx):
        image , _ = self[idx]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.permute(1, 2, 0).numpy() * std + mean
        plt.title("{} ({})".format(
            self.df["individual_id"].iloc[idx],
            self.df["species"].iloc[idx]
        ))
        plt.imshow(image)
        plt.show()

class TestDataset(Dataset):
    def __init__(self, df, path):
        self.path = path
        self.df = df
        # Augmentations
        self.transforms = albumentations.Compose([
                albumentations.Normalize(),
                ToTensorV2(p=1.0)
            ])


        
    def __getitem__(self, idx):
        image_name = self.df.image.values[idx]
        # read image 
        image_path = os.path.join(self.path, image_name)
        image = cv2.imread(image_path)[:, :, ::-1]

        image = self.transforms(image= image)["image"]

        return image

    def __len__(self):
        return len(self.df)


