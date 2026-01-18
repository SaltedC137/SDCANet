import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import tifffile
import config as cfg

class MeanStdCalculator(torch.utils.data.Dataset):
    def __init__(self, img_dir, in_channels=3):
        self.img_dir = img_dir
        self.in_channels = in_channels
        
        self.img_names = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        print(f"Found {len(self.img_names)} images for calculation")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        if self.in_channels > 3:
            try:
                image = tifffile.imread(img_path)
            except Exception:
                image = np.array(Image.open(img_path))
            
            # Ensure HWC format
            if image.ndim == 2:
                image = image[:, :, None]
            elif image.ndim == 3 and image.shape[0] == self.in_channels and image.shape[0] < image.shape[1]:
                 # Assume CHW, convert to HWC
                 image = image.transpose(1, 2, 0)
        else:
            image = np.array(Image.open(img_path).convert('RGB'))

        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image / 255.0
        elif image.dtype == np.uint16:
            image = image / 65535.0
            
        return image

def calculate_mean_std(in_channels):
    # Setup dataset
    dataset = MeanStdCalculator(cfg.TRAIN_ROOT, in_channels=in_channels)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(in_channels)
    std = torch.zeros(in_channels)
    
    print("Calculating mean and std...")
    
    cnt = 0
    fst_moment = torch.empty(in_channels)
    snd_moment = torch.empty(in_channels)

    for images in tqdm(loader):

        # images shape: [batch, H, W, C] (since we didn't use ToTensor)
        # We need [C, H, W] for easier calculation or just flatten
        b, h, w, c = images.shape
        nb_pixels = b * h * w
        # Flatten: [Batch*H*W, C]
        flattened = images.view(-1, c)
        
        sum_ = torch.sum(flattened, dim=0)
        sum_of_squares = torch.sum(flattened ** 2, dim=0)
        
        if cnt == 0:
            fst_moment = sum_
            snd_moment = sum_of_squares
        else:
            fst_moment += sum_
            snd_moment += sum_of_squares
            
        cnt += nb_pixels

    mean = fst_moment / cnt
    std = torch.sqrt(snd_moment / cnt - mean ** 2)

    return mean, std

if __name__ == '__main__':
    print(f"Starting calculation for {cfg.IN_CHANNELS} channels...")
    print(f"Data root: {cfg.TRAIN_ROOT}")
    
    try:
        mean, std = calculate_mean_std(cfg.IN_CHANNELS)
        
        print("\n" + "="*50)
        print("Calculation Complete!")
        print("="*50)
        print(f"mean = {mean.tolist()}")
        print(f"std  = {std.tolist()}")
        print("="*50)
        print("\nCopy the above lines to your config.py")
        
    except Exception as e:
        print(f"\nError: {e}")
