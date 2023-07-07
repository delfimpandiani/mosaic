from typing import Union, List, Callable
from pathlib import Path
import torch
from torchvision.models import vgg16, VGG16_Weights
from transformers import AutoProcessor, CLIPModel
from PIL import Image


class ConvImageEncoder:
  def __init__(self, weights: Path = None, use_classifier: bool = False):
    """
    Initialise the image encoder using a specific model and weight.

    Args:
        weights (str, optional): The weight to be used for the model. Defaults to None.
    """
    if weights is None:
      self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    else:
      self.model = vgg16()
      self.model.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=8)
      self.model.load_state_dict(torch.load(str(weights)))

    self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms(antialias=True)
    
    self.model.eval()
    
    if use_classifier:
      self.model.classifier[-2] = torch.nn.Identity()
      self.model.classifier[-1] = torch.nn.Identity()
    else:
      self.model.classifier = torch.nn.Identity()
      
    # workaround to get the output dimension of the model
    tmp_image = self.preprocess(Image.new("RGB", (50, 50))).unsqueeze(0)
    self._out_shape = self(tmp_image).squeeze(0).shape[0]

  def transform(self, *args, **kwargs) -> Callable:
    """
    Returns:
        Callable: The transform (preprocessing) function defined by a specific model.
    """
    return self.preprocess(*args, **kwargs)

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

class CLIPImageEncoder:
  def __init__(self):
    """
    Initialise the CLIP image encoder.
    """
    self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
  def transform(self, *args, **kwargs) -> Callable:
    """
    Returns:
        Callable: The transform (preprocessing) function defined by a specific model.
    """
    return self.processor(images=args[0], return_tensors="pt")["pixel_values"]

  @property
  def output_shape(self) -> int:
    """
    Returns:
        int: The dimensionality of the vector poduced by the image model.
    """
    return self.model.projection_dim

  def __call__(self, batch) -> torch.tensor:
    """
    Call the model on the provided batch.

    Args:
        batch: Input batch.

    Returns:
        torch.tensor: Features extracted from the image encoder
    """
    return self.model.get_image_features(batch)