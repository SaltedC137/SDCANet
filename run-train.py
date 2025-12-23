from interface.train import train
from utils.model_enum import ModelType
import config as cfg

def main():
    selected_model_type = ModelType.SDCANet
    struction = train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    if struction == True:
        print(f"{selected_model_type.get_model} train was finished")
    
    selected_model_type = ModelType.UNET
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    if struction == True:
        print(f"{selected_model_type.get_model} train was finished")

if __name__ == "__main__":
    main()