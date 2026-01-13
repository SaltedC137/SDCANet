from interface.draw import plot_learning_curves
from utils.model_enum import ModelType
import config as cfg


def main():
    selected_model_type = ModelType.UNET
    plot_learning_curves(selected_model_type)

    selected_model_type = ModelType.UNET_PLUS_PLUS
    plot_learning_curves(selected_model_type)

    selected_model_type = ModelType.SDCANet
    plot_learning_curves(selected_model_type)

    selected_model_type = ModelType.SEGNET
    plot_learning_curves(selected_model_type)

    selected_model_type = ModelType.LINKNET
    plot_learning_curves(selected_model_type)

    selected_model_type = ModelType.ABCNET
    plot_learning_curves(selected_model_type)


if __name__ == "__main__":
    main()