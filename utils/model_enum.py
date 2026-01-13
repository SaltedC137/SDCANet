from enum import Enum
from models import *
import torch.nn as nn

class ModelType(Enum):
    """
    Enum for different model types available in the project
    """
    UNET = "unet"
    UNET_PLUS_PLUS = "unetpp"
    SEGNET = "segnet"
    LINKNET = "linkent"
    ABCNET = "abcnet"
    SDCANet = "sdcanet"
    DEEPLABV3 = "deeplabv3"
    EFFICIENTUNET = "efficientnet"


    def get_model(self, in_channels: int = 3, num_classes: int = 2) -> nn.Module:

            if self == ModelType.UNET:
                return Unet(inchanel=in_channels, outchanel=num_classes)
            elif self == ModelType.UNET_PLUS_PLUS:
                return UNetPlusPlus(num_classes=num_classes, input_channels=in_channels)
            elif self == ModelType.SEGNET:
                return segnet(in_channels=in_channels, num_classes=num_classes)
            elif self == ModelType.LINKNET:
                return linknet(n_classes=num_classes)
            elif self == ModelType.ABCNET:
                return ABCNet(band=in_channels, n_classes=num_classes)
            elif self == ModelType.SDCANet:
                return SDCANet(num_classes = num_classes)
            elif self == ModelType.DEEPLABV3:
                return DeepLabV3Plus(num_classes = num_classes)
            elif self == ModelType.EFFICIENTUNET:
                 return EfficientUNet(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown model type: {self.value}")


def get_model_by_type(model_type: ModelType, in_channels: int = 3, num_classes: int = 2) -> nn.Module:
        return model_type.get_model(in_channels, num_classes)