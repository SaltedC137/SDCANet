import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch

class Datainit(torch.utils.data.Dataset):
    def __init__(self, root_list, transform=None, class_num=1):

        self.img_dir = root_list[0]
        self.label_dir = root_list[1]
        self.transform = transform
        self.class_num = class_num
        
        self.img_names = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        self._validate_files()
        
    def _validate_files(self):
        valid_pairs = []
        for img_name in self.img_names:
            base_name = os.path.splitext(img_name)[0]
            label_candidates = [
                f for f in os.listdir(self.label_dir)
                if f.startswith(base_name) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ]
            
            if label_candidates:
                valid_pairs.append((img_name, label_candidates[0]))
            else:
                print(f"Warning: {img_name} No Label")
        
        self.img_names = [pair[0] for pair in valid_pairs]
        self.label_names = [pair[1] for pair in valid_pairs]
        
        print(f"Found {len(self.img_names)} Accessible Label-Img")
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label_name = self.label_names[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)

        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('L')) 

        if self.class_num == 2:

            label = (label > 0).astype(np.uint8)
        elif self.class_num == 1:
            if label.max() > 1:
                label = (label > 0).astype(np.uint8)
        else:
            unique_vals = np.unique(label)
            if len(unique_vals) > self.class_num:
                print(f"Warning: Found {len(unique_vals)} unique values but class_num={self.class_num}")
            label = np.clip(label, 0, self.class_num - 1).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

            if isinstance(label, torch.Tensor):
                label = label.long()
            else:
                label = torch.from_numpy(label).long()
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            label = torch.from_numpy(label).long()
        
        return {"img": image, "label": label}