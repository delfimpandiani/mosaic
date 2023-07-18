from typing import Union, List, Callable
from pathlib import Path

from itertools import islice

import torch
import lightning as pl
from torchvision.models import vgg16, VGG16_Weights
import timm
from PIL import Image

class ImageEncoder:
  @property
  def output_shape(self) -> int:
    """
    Returns:
        int: The dimensionality of the vector poduced by the image model.
    """
    return self._out_shape

  def __call__(self, batch: torch.tensor) -> torch.tensor:
    """
    Call the model on the provided batch.

    Args:
        batch (torch.tensor): Input batch.

    Returns:
        torch.tensor: Features extracted from the image encoder
    """
    return self.model(batch)


class VGGImageEncoder(ImageEncoder):
  def __init__(self, weights: Path = None):
    """
    Initialise the image encoder using a specific model and weight.

    Args:
        weights (str, optional): The weight to be used for the model. Defaults to None.
    """
    self.model = vgg16(pretrained=True)
    self._out_shape = self.model.classifier[-1].in_features
    self.model.classifier[-1] = torch.nn.Dropout(0.5)

    if weights is not None:  
      # extract state_dict from lightning model
      self.model.load_state_dict({
        k.replace("model.", ""): v  
        for k, v in torch.load(weights)["state_dict"].items()
        if "model" in k
      })

    self.model.eval()


class ViTImageEncoder(ImageEncoder):
  def __init__(self, weights: Path = None):
    """
    Initialise the ViT image encoder using a specific model and weight.

    Args:
        weights (str, optional): The weight to be used for the model. Defaults to None.
    """
    self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
    self._out_shape = self.model.head.in_features
    self.model.head = torch.nn.Dropout(0.5)

    if weights is not None:  
      # extract state_dict from lightning model
      self.model.load_state_dict({
        k.replace("model.", ""): v  
        for k, v in torch.load(weights)["state_dict"].items()
        if "model" in k
      })

    self.model.eval()