import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "true"
import albumentations as album
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(crop_size):
    return album.Compose([
        album.RandomCrop(height=crop_size, width=crop_size),
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
        album.RandomRotate90(p=0.5),
        album.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        album.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        album.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_validation_augmentation(val_size):
    return album.Compose([
        album.Resize(height=val_size, width=val_size),
        album.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])