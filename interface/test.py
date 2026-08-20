from utils.augmentation import *
from utils.Dataset import *
from utils.Metrics import *
import numpy as np
from torch.utils.data import DataLoader
import config
import config as cfg
import glob
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import time
from utils import *
from torch.autograd import Variable
import csv
import gc


def test(usemodel) ->bool:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.cuda.empty_cache()
    gc.collect()

    BATCH_SIZE = config.BATCH_SIZE

    my_test = Datainit([config.TEST_ROOT, config.TEST_LABEL],
                       get_validation_augmentation(config.H_size, mean=config.NORM_MEAN, std=config.NORM_STD),
                       in_channels=config.IN_CHANNELS,
                       class_num=config.class_num)
    test_data = DataLoader(my_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    net = usemodel()
    net_name = net.__class__.__name__
    print(f"model : {net_name}")

    net.eval()

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    else:
        net=net.to(device)
        print(" Singal GPU to predict")

    best_model_path = "./output/" +net_name +"/"+net_name + "_best.pth"

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    net.load_state_dict(state_dict)  
    net = net.to(device)

    pre_labels = []
    true_labels = []
    for sample in test_data:
        data = Variable(sample['img']).to(device)
        label = Variable(sample['label']).to(device)
        out = net(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_labels.extend(pre_label)

        true_label = label.data.cpu().numpy()
        true_labels.extend(true_label)

    eval_metrix = eval_semantic_segmentation(pre_labels, true_labels, cfg.class_num)
    precision = np.nan_to_num(eval_metrix['precision_per_class'][1], nan=0.0)
    recall = np.nan_to_num(eval_metrix['recall_per_class'][1], nan=0.0)
    f1 = np.nan_to_num(eval_metrix['f1_per_class'][1], nan=0.0)

    result_data = {
        'model_name': net_name,
        'test_miou': eval_metrix['miou'],
        'test_accuracy': eval_metrix['pixel_accuracy'],
        'test_recall': recall,
        'test_f1': f1,
        'test_precision': precision,
        'test_kappa': eval_metrix['kappa']
    }

    csv_file_path = cfg.test_result
    write_header = not os.path.exists(csv_file_path)    
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model_name', 'test_miou', 'test_accuracy', 'test_recall', 'test_f1','test_precision', 'test_kappa']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader(result_data)
        writer.writerow(result_data)

    epoch_str = 'model_name {}, test_miou: {:.5f}, test_accuracy: {:.5f}, test_recall: {:.5f}, test_f1:{:.5f}, test_precision: {:.5f}, test_kappa: {:.5f}'.format(
        net_name,
        eval_metrix['miou'],
        eval_metrix['pixel_accuracy'],
        recall,
        f1,
        precision,
        eval_metrix['kappa']
    )

    print(epoch_str)

    del net
    torch.cuda.empty_cache()