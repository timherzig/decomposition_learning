import importlib
import torch


def get_class(class_name, modules):
    """Get class from a string"""
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported class: {class_name}")


def get_pos_weight(x):
    """Get the positive weight for a binary cross entropy loss

    Args:
        x (torch.Tensor): The input tensor consisting of 0s and 1s. Shape: (batch_size, channels, height, width)

    Returns:
        torch.Tensor: The positive weight
    """

    # compute proportion 0s to 1s in image
    # #pixels with value 0 / #pixels with value 1
    factor = ((x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]) - x.sum()) / x.sum()
    return (torch.ones([x.shape[2], x.shape[3]]).to(x.device) * factor.item()).to(
        x.device
    )
