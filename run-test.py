from interface.test import test
from utils.model_enum import ModelType
import config as cfg

def main():
    selected_model_type = ModelType.SDCANet
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.SEGNET
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.ABCNET
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.LINKNET
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.UNET
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))


    selected_model_type = ModelType.UNET_PLUS_PLUS
    test(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

if __name__ == "__main__":
    main()