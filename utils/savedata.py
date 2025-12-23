import os
import datetime
import torch
import csv
from datetime import datetime 
import config as cfg


def save_model(model, model_name, save_dir=cfg.SAVE_PATH, epoch=None,is_best = False, **kwargs):

    save_dir = save_dir + model_name  + "/"
    os.makedirs(save_dir, exist_ok=True)

    if is_best:
        filename = f"{model_name}_best.pth"
    else:
        filename = f"{model_name}_last.pth"
    save_path = os.path.join(save_dir, filename)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_dict = {
        'model_state_dict': state_dict,
        'model_name': model_name,
        'epoch': epoch, 
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    save_dict.update(kwargs)
    torch.save(save_dict, save_path)
    
    if is_best:
        print(f"★ Best model saved to: {save_path}")
    else:
        print(f"-> Latest model saved to: {save_path}")
    return save_path

def log_single_epoch(epoch, miou, pixel_accuracy, loss, model_name, save_dir=cfg.SAVE_PATH, filename=
None):
    
    save_dir = save_dir + model_name  + "/"
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_training_log_{timestamp}.csv"

    save_path = os.path.join(save_dir, filename)

    write_header = not os.path.exists(save_path)

    with open(save_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['epoch', 'miou', 'pixel_accuracy', 'loss', 'timestamp', 'phase']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({
            'epoch': epoch,
            'miou': miou,
            'pixel_accuracy': pixel_accuracy,
            'loss': loss,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'phase': 'train'
        })

    return save_path