from interface.predict import predict
from utils.model_enum import ModelType
import config as cfg

def main():
    selected_model_type = ModelType.SDCANet
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))


    selected_model_type = ModelType.UNET
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.UNET_PLUS_PLUS
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.ABCNET
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))


    selected_model_type = ModelType.LINKNET
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.SEGNET
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))

    selected_model_type = ModelType.DEEPLABV3
    struction = predict(usemodel=lambda: selected_model_type.get_model(
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ))


if __name__ == "__main__":
    main()