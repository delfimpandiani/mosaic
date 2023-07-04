from typing import Callable, Union, Tuple
from pathlib import Path
import torch
from torchvision.io import read_image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from mosaic.model.projection import Image2KGEProjection


FRAMESTER_PREFIX = "https://w3id.org/framester/conceptnet5/data/en/%s"

class ARTstractDataset(torch.utils.data.Dataset):
  def __init__(self, dataset: Path, transform: Union[Callable, None] = None):
    """
    Initialise the ARTstract dataset using torch Dataset interface.

    Args:
        dataset (Path): Path to ARTstract dataset. The path should point
          to a folder containing a folder for each cluster.
        transform (Union[Callable, None], optional):
          The transform routine undergone by an image.
    """
    self.path = dataset
    self.transform = transform

    self.img_path = []
    self.img_label = []
    for cluster in self.path.iterdir():
      if cluster.is_dir():
        for img in cluster.iterdir():
          self.img_path.append(img.absolute())
          self.img_label.append(FRAMESTER_PREFIX % cluster.name)

    self.img_path = np.array(self.img_path)
    self.img_label = np.array(self.img_label)
  
  def __len__(self) -> int:
    """
    Returns:
        int: Number of samples in the ontology.
    """
    return len(self.img_path)

  def __getitem__(self, idx: int) -> Tuple[np.array, str, str]:
    """
    Retrieve a sample from the dataset.

    Args:
        idx (int): The index of the sample.

    Returns:
        Tuple[np.array, str, str]: 
          Tuple containing the image matrix, the positive label string
          and the negative label string.
    """
    img_path = self.img_path[idx]
    image = self.transform(read_image(str(img_path)))

    label = self.img_label[idx]

    negative_idx = np.random.choice(np.where(self.img_label != label)[0])
    negative_label = self.img_label[negative_idx]

    return image, label, negative_label
  