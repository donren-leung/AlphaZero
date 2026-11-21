import torch.nn as nn

from .CNNs import ResNet, SE_ResNet

DEFAULT_ARCHITECTURE = "resnet"
MODEL_ARCHITECTURES: dict[str, type[nn.Module]] = {
    "resnet": ResNet,
    "se_resnet": SE_ResNet
}

DEFAULT_BLOCKS = 9
DEFAULT_CHANNELS = 256
DEFAULT_HEAD_CHANNELS = 256

def create_model(*,
                 architecture: str = DEFAULT_ARCHITECTURE,
                 state_dict: dict | None = None,
                 **model_args) -> nn.Module:
    try:
        model_type = MODEL_ARCHITECTURES[architecture]
    except KeyError:
        raise ValueError(
            f"Unknown model architecture {architecture!r}. Choose from {', '.join(MODEL_ARCHITECTURES)}."
        ) from None

    model = model_type(**model_args)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model
