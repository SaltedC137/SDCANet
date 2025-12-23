from utils.augmentation import *
from utils.Dataset import *
from utils.Metrics import *
import numpy as np
from torch.utils.data import DataLoader
import config
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


def test(usemodel) ->bool:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    BATCH_SIZE = config.BATCH_SIZE
    miou_list = [0]

    my_test = Datainit([config.TEST_ROOT, config.TEST_LABEL], get_validation_augmentation(config.H_size))
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

    error = 0
    train_mpa = 0
    train_miou = 0
    train_class_acc = 0
    train_pa = 0
    train_recall=0
    train_f1=0
    train_precision=0
    train_kappa=0
    for i, sample in enumerate(test_data):
        data = Variable(sample['img']).to(device)
        label = Variable(sample['label']).to(device)
        out = net(data)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = label.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrix = eval_semantic_segmentation(pre_label, true_label,cfg.class_num)
        train_mpa = eval_metrix['mean_class_accuracy'] + train_mpa
        train_miou = eval_metrix['miou'] + train_miou
        train_pa = eval_metrix['pixel_accuracy'] + train_pa
        train_recall=eval_metrix["recall"]+train_recall
        train_f1=eval_metrix["f1"]+train_f1
        train_precision=eval_metrix["precision"]+train_precision
        train_kappa=eval_metrix["kappa"]+train_kappa


        if len(eval_metrix['class_accuracy']) < config.class_num:             
            eval_metrix['class_accuracy'] = 0
            train_class_acc = train_class_acc + eval_metrix['class_accuracy']
            error += 1
        else:
            train_class_acc = train_class_acc + eval_metrix['class_accuracy']

        print(eval_metrix['class_accuracy'], '================', i)


    result_data = {
        'model_name': net_name,
        'test_miou': train_miou / (len(test_data) - error),
        'test_accuracy': train_pa / (len(test_data) - error),
        'test_recall': train_recall / (len(test_data) - error),
        'test_f1': train_f1 / (len(test_data) - error),
        'test_precision': train_precision / (len(test_data) - error),
        'test_kappa': train_kappa / (len(test_data) - error)
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
        train_miou / (len(test_data) - error),
        train_pa / (len(test_data) - error),
        train_recall / (len(test_data) - error),
        train_f1 / (len(test_data) - error),
        train_precision / (len(test_data) - error),
        train_kappa / (len(test_data) - error)
    )

    if train_miou / (len(test_data) - error) > max(miou_list):
        miou_list.append(train_miou / (len(test_data) - error))
        print(epoch_str + '==========last')