import matplotlib.pyplot as plot
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import config as cfg
from utils.model_enum import ModelType


def _parse_log_index(filename, net_name, phase):
    pattern = rf"^{re.escape(net_name)}_{phase}_log(?:_(\\d+))?\\.csv$"
    match = re.match(pattern, filename)
    if not match:
        return None
    if match.group(1) is None:
        return 0
    return int(match.group(1))


def _find_log_pairs(log_dir, net_name):
    if not os.path.exists(log_dir):
        return {}

    train_map = {}
    val_map = {}

    for filename in os.listdir(log_dir):
        train_index = _parse_log_index(filename, net_name, "train")
        if train_index is not None:
            train_map[train_index] = os.path.join(log_dir, filename)
            continue

        val_index = _parse_log_index(filename, net_name, "val")
        if val_index is not None:
            val_map[val_index] = os.path.join(log_dir, filename)

    common_indices = sorted(set(train_map.keys()) & set(val_map.keys()))
    return {idx: (train_map[idx], val_map[idx]) for idx in common_indices}


def _resolve_log_pair(log_pairs, log_index):
    if not log_pairs:
        return None, None, None

    if log_index is None:
        selected_index = max(log_pairs.keys())
    else:
        if log_index not in log_pairs:
            return None, None, sorted(log_pairs.keys())
        selected_index = log_index

    train_log_path, val_log_path = log_pairs[selected_index]
    return selected_index, (train_log_path, val_log_path), sorted(log_pairs.keys())


def plot_learning_curves(model_type: ModelType, log_index=None):

    fontweight = 16

    temp_net = model_type.get_model(in_channels=cfg.IN_CHANNELS, num_classes=cfg.class_num)
    net_name = temp_net.__class__.__name__
    del temp_net 
    
    print(f" [{net_name}] start draw...")


    log_dir = os.path.join("./output", net_name)
    log_pairs = _find_log_pairs(log_dir, net_name)
    selected_index, selected_paths, available_indices = _resolve_log_pair(log_pairs, log_index)

    if selected_paths is None:
        if available_indices is None:
            print(f"fail: can't find train/val log pairs in {log_dir}")
        else:
            print(f"fail: log index {log_index} not found for {net_name}. available: {available_indices}")
        return False

    train_log_path, val_log_path = selected_paths
    print(f" use train log: {os.path.basename(train_log_path)}")
    print(f" use val log: {os.path.basename(val_log_path)}")


    df_train = pd.read_csv(train_log_path)
    df_val = pd.read_csv(val_log_path)

    plt.figure(figsize=(15, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False


    plt.subplot(1, 2, 1)
    plt.plot(df_train['epoch'], df_train['loss'], label='训练集 Loss', color='#3498DB', linewidth=2)
    plt.plot(df_val['epoch'], df_val['loss'], label='验证集 Loss', color='#E74C3C', linestyle='--', linewidth=2)
    plt.title(f'{net_name} Loss 曲线', fontsize=fontweight)
    plt.xticks(fontsize = fontweight)
    plt.yticks(fontsize = fontweight)
    plt.xlabel('训练轮数 (Epoch)', fontsize=fontweight)
    plt.ylabel('损失值', fontsize=fontweight)
    plt.legend(fontsize = fontweight - 2)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df_train['epoch'], df_train['miou'], label='训练集 mIoU', color='#2ECC71', linewidth=2)
    plt.plot(df_val['epoch'], df_val['miou'], label='验证集 mIoU', color='#9B59B6', linestyle='--', linewidth=2)
    plt.title(f'{net_name} mIoU 曲线', fontsize=fontweight)
    plt.xticks(fontsize = fontweight)
    plt.yticks(fontsize = fontweight)
    plt.xlabel('训练轮数 (Epoch)', fontsize=fontweight)
    plt.ylabel('mIoU 得分', fontsize=fontweight)
    plt.legend(fontsize = fontweight - 2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "" if selected_index == 0 else f"_{selected_index}"
    save_path = os.path.join(log_dir, f"{net_name}_metrics_summary{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close() 
    
    print(f"Success: picture saved -> {save_path}")
    return True

if __name__ == '__main__':

    pass