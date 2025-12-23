import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import sys
from utils.savedata import save_model
from utils.savedata import log_single_epoch
from utils.Dataset import Datainit
from utils.augmentation import get_training_augmentation
from utils.augmentation import get_validation_augmentation
from utils.Metrics import eval_semantic_segmentation
import config as cfg
from module.DiceFocalLoss import DiceFocalLoss


# no warning

# os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# warnings.filterwarnings("ignore", message="A new version of Albumentations is available.*")
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def train(usemodel) -> bool:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")

    # Load Data

    try:
        train_dataset = Datainit([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], 
                                  get_training_augmentation(cfg.H_size),
                                  class_num=cfg.class_num)
        val_dataset = Datainit([cfg.VAL_ROOT, cfg.VAL_LABEL], 
                                get_validation_augmentation(cfg.W_size),
                                class_num=cfg.class_num)

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, 
                                  shuffle=True, num_workers=8, drop_last=True,pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, 
                                shuffle=False, num_workers=8,pin_memory=True)
    except Exception as e:
        print(f"data load failure : {e}")
        sys.exit(1)

    print("Sample :")
    test_sample = next(iter(train_loader))
    test_img = test_sample["img"].to(device)
    test_label = test_sample["label"].to(device)
    print(f"input image size: {test_img.shape} (B, C, H, W)")
    print(f"label size: {test_label.shape}")    

    model = usemodel()
    model_name = model.__class__.__name__

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    else:
        model = model.to(device)
        print(" Singal GPU to train")

    model = model
    print(f"model :{model_name}")
    best_miou = 0.0

    # criterion = FocalLoss(
    #     alpha=[0.2,0.8],
    #     gamma=2,
    #     num_classes=cfg.class_num,
    #     task_type='multi-class'
    # ).to(device)

    criterion = DiceFocalLoss(
        weight_dice=0.35,
        weight_focal=0.65,
        focal_alpha=0.4,
        class_num=cfg.class_num
    ).to(device)
    
    # optimiser
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=cfg.LR_SCHEDULER_FACTOR
                                  ,patience=cfg.LR_SCHEDULER_PATIENCE, min_lr=1e-7)
    
    start_time = time.time()

    for epoch in range (1 , 1 + cfg.EPOCH_NUMBER + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\nEpoch[{epoch}/{cfg.EPOCH_NUMBER}] current lr:{current_lr}')
        
        model.train()
        total_train_loss = 0.0
        train_pred_list = []
        train_true_list = []
        
        for i, sample in enumerate(train_loader):
            img_data = sample["img"].to(device)
            img_label = sample["label"].to(device)
            
            optimizer.zero_grad()
            preds = model(img_data)

            if isinstance(preds,(list,tuple)):
                out = preds[0]
                aux_out = preds[1]
            else:
                out = preds
                aux_out = None

            if out.shape[-2:] != img_label.shape[-2:]:
                out = F.interpolate(out, size=img_label.shape[-2:], mode='bilinear', align_corners=True)

            if aux_out is not None and aux_out.shape[-2:] != img_label.shape[-2:]:
                aux_out = F.interpolate(aux_out , size=img_label.shape, mode='bilinear', align_corners=True)

            if img_label.ndim == 4 and img_label.shape[1] == 1:
                target = img_label.squeeze(1).long()
            else:
                target = img_label.long()


            if aux_out is not None:
                loss_main = criterion(out,target)
                loss_aux = criterion(aux_out,target)
                loss = loss_main + 0.4 * loss_aux
            else:
                loss = criterion(out , target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

            with torch.no_grad():
                pred = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
                true = target.cpu().numpy().astype(np.uint8)
                train_pred_list.extend(pred)
                train_true_list.extend(true)

            print(f'| Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.6f} |')

        avg_train_loss = total_train_loss / len(train_loader)
        
        # train target
        if cfg.class_num == 2:
            train_metrics = eval_semantic_segmentation(train_pred_list, train_true_list, 
                                                      n_class=cfg.class_num, ignore_label=255)
            
        train_miou = train_metrics['miou']
        train_pixel_acc = train_metrics.get('pixel_accuracy', 0.0)

        log_single_epoch(
            epoch=epoch,
            miou=train_miou,
            pixel_accuracy=train_pixel_acc,
            loss=avg_train_loss,
            model_name=model_name,
            filename=f"{model_name}_train_log.csv"
        )

        # validate
        model.eval()
        total_val_loss = 0.0
        val_pred_list = []
        val_true_list = []

        with torch.no_grad():
            for sample in val_loader:
                img = sample['img'].to(device)
                label = sample['label'].to(device)

                preds = model(img)

                if isinstance(preds,(list,tuple)):
                    out = preds[0]
                else: 
                    out = preds
                
                if out.shape[-2:] != label.shape[-2:]:
                    out = F.interpolate(out, size=label.shape[-2:], mode='bilinear', align_corners=True)
                
                if label.ndim == 4 and label.shape[1] == 1:
                    target = label.squeeze(1).long()
                else:
                    target = label.long()

                loss = criterion(out, target)
                total_val_loss += loss.item()

                pred = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
                true = label.cpu().numpy().astype(np.uint8)

                val_pred_list.extend(pred)
                val_true_list.extend(true)
                
                true = label.cpu().numpy().astype(np.uint8)

        avg_val_loss = total_val_loss / len(val_loader)

        if cfg.class_num == 2:
            val_metrics = eval_semantic_segmentation(val_pred_list, val_true_list, 
                                                    n_class=cfg.class_num, ignore_label=255)

        val_miou = val_metrics['miou']
        val_pixel_acc = val_metrics.get('pixel_accuracy', 0.0)
        scheduler.step(val_miou)
        # save model
        log_single_epoch(
            epoch=epoch,
            miou=val_miou,
            pixel_accuracy=val_pixel_acc,
            loss=avg_val_loss,
            model_name=model_name,
            filename=f"{model_name}_val_log.csv"
        )

        save_model(
            model=model,
            model_name=model_name,
            save_dir=cfg.SAVE_PATH,
            epoch=epoch,
            is_best=False,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch_miou=val_miou,
            config={
                "LR": cfg.LR,
                "BATCH_SIZE": cfg.BATCH_SIZE,
                "EPOCH": cfg.EPOCH_NUMBER,
                "CLASS_NUM": cfg.class_num
            }
        )

        if val_miou > best_miou:
            best_miou = val_miou
            save_model(
                model=model,
                model_name=model_name,
                save_dir=cfg.SAVE_PATH,
                epoch=epoch,
                is_best=True,
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch_miou=val_miou,
                config={
                    "LR": cfg.LR,
                    "BATCH_SIZE": cfg.BATCH_SIZE,
                    "EPOCH": cfg.EPOCH_NUMBER,
                    "CLASS_NUM": cfg.class_num
                }
            )
            print(f"Best model saved with mIoU: {best_miou:.5f}")

        print(f"| Train Loss: {avg_train_loss:.5f} | Train mIoU: {train_miou:.5f} |")
        print(f"| Val Loss: {avg_val_loss:.5f} | Val mIoU: {val_miou:.5f} |")
        if 'pixel_accuracy' in val_metrics:
            print(f"| Pixel Acc: {val_metrics['pixel_accuracy']:.5f} |")
        print("-" * 80)
    
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("Finish Train")
    print(f"Total: {int(hours)}h{int(minutes)}m{seconds:.1f}s")
    print(f"Best val mIoU: {best_miou:.5f}")

    return True