from interface.train import train
from utils.model_enum import ModelType
import config as cfg

def main():

    selected_model_type = ModelType.SDCANet
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    
    selected_model_type = ModelType.UNET
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.UNET_PLUS_PLUS
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))



    selected_model_type = ModelType.ABCNET
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.LINKNET
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.SEGNET
    train(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

if __name__ == "__main__":
    main()