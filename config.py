import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os



plt.rcParams['font.family']='SimHei'

seed = 707
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


TRAIN_ROOT = "./TrainData/images/train_images"
TRAIN_LABEL = "./TrainData/labels/train_labels"
VAL_ROOT = "./TrainData/images/val_images"
VAL_LABEL = "./TrainData/labels/val_labels"
TEST_ROOT = "./TrainData/images/test_images"
TEST_LABEL = "./TrainData/labels/test_labels"



class_dict_path = "./TrainData/class_dict.csv"
if not os.path.exists(class_dict_path):
    class_dict_data = {
        'name': ['background', 'foreground'],
        'r': [0, 255],
        'g': [0, 255],
        'b': [0, 255]
    }
    df = pd.DataFrame(class_dict_data)
    os.makedirs(os.path.dirname(class_dict_path), exist_ok=True)
    df.to_csv(class_dict_path, index=False)
    print(f"establish complete: {class_dict_path}")


BATCH_SIZE = 8                     
H_size = 256   
W_size= 256   
class_num=2    
LR=0.0002       
EPOCH_NUMBER=1
WEIGHT_DECAY=0.01
LR_SCHEDULER_FACTOR=0.5
LR_SCHEDULER_PATIENCE=10
IN_CHANNELS = 3
NUM_CLASSES = class_num

SAVE_PATH = "./output/"

test_result = "./output/model-result.csv"