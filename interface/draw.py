import matplotlib.pyplot as plot
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import config as cfg
from utils.model_enum import ModelType

def plot_learning_curves(model_type: ModelType):


    temp_net = model_type.get_model(in_channels=cfg.IN_CHANNELS, num_classes=cfg.class_num)
    net_name = temp_net.__class__.__name__
    del temp_net 
    
    print(f" [{net_name}] start draw...")


    log_dir = os.path.join("./output", net_name)
    train_log_path = os.path.join(log_dir, f"{net_name}_train_log.csv")
    val_log_path = os.path.join(log_dir, f"{net_name}_val_log.csv")


    if not os.path.exists(train_log_path) or not os.path.exists(val_log_path):
        print(f"fail: can't find log \npath1: {train_log_path}\npath2: {val_log_path}")
        return False


    df_train = pd.read_csv(train_log_path)
    df_val = pd.read_csv(val_log_path)

    plt.figure(figsize=(15, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False


    plt.subplot(1, 2, 1)
    plt.plot(df_train['epoch'], df_train['loss'], label='Train Loss', color='#3498DB', linewidth=2)
    plt.plot(df_val['epoch'], df_val['loss'], label='Val Loss', color='#E74C3C', linestyle='--', linewidth=2)
    plt.title(f'{net_name} Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_train['epoch'], df_train['miou'], label='Train mIoU', color='#2ECC71', linewidth=2)
    plt.plot(df_val['epoch'], df_val['miou'], label='Val mIoU', color='#9B59B6', linestyle='--', linewidth=2)
    plt.title(f'{net_name} mIoU', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mIoU ', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(log_dir, f"{net_name}_metrics_summary.png")
    plt.savefig(save_path, dpi=300)
    plt.close() 
    
    print(f"Success: picture saved -> {save_path}")
    return True

if __name__ == '__main__':

    pass