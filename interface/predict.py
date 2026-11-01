from utils.augmentation import *
from utils.Dataset import *
from utils.Metrics import *
import numpy as np
from torch.utils.data import DataLoader
import config
import config as cfg
import time
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
from utils import *


def predict(usemodel) -> bool:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = cfg.SAVE_PATH
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    my_test = Datainit([config.TEST_ROOT, config.TEST_LABEL],
                       get_validation_augmentation(config.H_size),
                       in_channels=cfg.IN_CHANNELS,
                       class_num=cfg.class_num)
    print(f"DataSize: {len(my_test)} | Data Path: {config.TEST_ROOT}")

    if len(my_test) == 0:
        print("No image-label pairs found, abort predict.")
        return False

    test_loader = DataLoader(my_test, batch_size=1, shuffle=False, num_workers=0)
    
    net = usemodel()
    net_name = net.__class__.__name__
    print(f"model : {net_name}")

    model_save_path = os.path.join(output_dir , net_name)
    os.makedirs(model_save_path ,exist_ok=True)

    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    else:
        net = net.to(device)
        print(" Singal GPU to predict")
    net.eval()
    
    best_model_path = os.path.join(output_dir, net_name, net_name + "_best.pth")

    try:
        load_checkpoint(net, best_model_path)
    except Exception as e:
        print(f"LoadFail: {e}")
        return False

    with torch.no_grad():
        sample_img = next(iter(test_loader))['img'].to(device)
        output = net(sample_img)
        print(f"model shape: {output.shape}")
        
        if output.shape[1] == 1:
            probs = torch.sigmoid(output)
            print("Single Output (Sigmoid):")
            for th in [0.3, 0.5, 0.7]:
                pct = (probs > th).float().mean().item() * 100
                print(f"  threshold {th}: white {pct:.1f}%")
        else:
            classes = output.argmax(1).unique().cpu().numpy()
            print(f"mutiple channels output (Softmax): predict class {classes}")

    print(f"\nStart Predict {len(test_loader)} Images (background=0, gally=255)...")
    stats = {'total_time': 0, 'whites': []}

    for idx, sample in enumerate(test_loader):
        original_name = my_test.img_names[idx]
        base_name = os.path.splitext(original_name)[0]
        
        print(f"\n[{idx+1:03d}/{len(test_loader)}] manage: {original_name}")
        start_t = time.time()
        
        img = sample['img'].to(device)
        
        with torch.no_grad():
            out = net(img)
            if out.shape[1] == 1:
                pred = (torch.sigmoid(out) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)
            else:
                pred = out.argmax(dim=1).squeeze(1).cpu().numpy().astype(np.uint8)

        white_pct = (np.sum(pred == 1) / pred.size) * 100
        stats['whites'].append(white_pct)
        
        save_img = Image.fromarray((pred * 255).astype(np.uint8))
        save_path = os.path.join(model_save_path, base_name)
        
        # save_img.save(f"{save_path}_binary_pre.png")
        save_img.save(f"{save_path}_binary_pre.tif", format='TIFF')
        # np.save(f"{save_path}_pred.npy", pred)
        
        cost_t = time.time() - start_t
        stats['total_time'] += cost_t
        
        print(f"  White scale: {white_pct:.2f}% | Time: {cost_t:.3f}s")
        print(f"  Saved: {base_name}_binary_pre.tif")

    print('\n' + '='*60)
    print(' Finish!')
    count = len(stats['whites'])
    avg_time = stats['total_time'] / count if count > 0 else 0
    whites = np.array(stats['whites'])

    print(f"Total time : {stats['total_time']:.2f}s | avg time: {avg_time:.3f}s/a sheet")
    print(f"White zone: avg {whites.mean():.2f}% | min {whites.min():.2f}% | max {whites.max():.2f}%")

    print("\ndistribute statistics:")
    ranges = [(0, 5), (5, 20), (20, 50), (50, 80), (80, 100)]
    for r_min, r_max in ranges:
        c = np.sum((whites >= r_min) & (whites < r_max))
        if c > 0: 
            print(f"  {r_min}-{r_max}%: {c}a sheet ({c/count*100:.1f}%)")

    del net
    torch.cuda.empty_cache()


def load_checkpoint(model, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file does not exist: {path}")
    
    print(f"Loading parameters: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint)))
    
    state = {k.replace('module.', ''): v for k, v in state.items()}
    
    model.load_state_dict(state, strict=True)
    return model

