from typing import Callable, Union, Tuple, List
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from itertools import chain
import json
import rdflib


FRAMESTER_PREFIX = "https://w3id.org/framester/conceptnet5/data/en/%s"
MUSCO = rdflib.Namespace("https://w3id.org/musco#")


def transform(image):
  t = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return t(image)


class ARTstractDataset(torch.utils.data.Dataset):
  def __init__(self, dataset: Path, augment: bool = True):
    """
    Initialise the ARTstract dataset using torch Dataset interface.

    Args:
        dataset (Path): Path to ARTstract dataset. The path should point
          to a folder containing a folder for each cluster.
        augment (bool, optional): Wether to augment the data or not. Defaults to True.
    """
    self.path = dataset
    self.augment = augment

    self.img_path = []
    self.img_label = []
    for cluster in self.path.iterdir():
      if cluster.is_dir():
        for img in cluster.iterdir():
          self.img_path.append(img.absolute())
          self.img_label.append(FRAMESTER_PREFIX % cluster.name)

    self.img_path = np.array(self.img_path)
    self.img_label = np.array(self.img_label)

    self.targets = np.unique(self.img_label)
    self.targets.sort()

    self.class_weight = compute_class_weight(
      class_weight="balanced", 
      classes=self.targets, 
      y=self.img_label)

    if augment:
      self.transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.RandomHorizontalFlip(p=0.5),
          transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.3),
          transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
          transforms.RandomApply([transforms.RandomCrop(224, padding=20)], p=0.3),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
    else:
      self.transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
  
  def __len__(self) -> int:
    """
    Returns:
        int: Number of samples in the ontology.
    """
    return len(self.img_path)

  def __getitem__(self, idx: int) -> Tuple[np.array, str, int]:
    """
    Retrieve a sample from the dataset.

    Args:
        idx (int): The index of the sample.

    Returns:
        Tuple[np.array, str, int]: 
          Tuple containing the image matrix, the positive label string
          and the index of that label.
    """
    img_path = self.img_path[idx]
    image = Image.open(str(img_path))
    image = self.transform(image)

    label = self.img_label[idx]
    label_idx = torch.tensor(np.argwhere(self.targets == label).reshape(-1))

    return image, label, label_idx


class ClusterARTstractDataset(ARTstractDataset):
  def __init__(self, dataset: Path, kg: Path, augment: bool = True):
    """
    ARTstract dataset where for each label the corresponding cluster concepts are
    provided.

    Args:
        dataset (Path): Path to ARTstract dataset. The path should point
          to a folder containing a folder for each cluster.
        augment (bool, optional): Wether to augment the data or not. Defaults to True.
    """
    super().__init__(dataset, augment)

    graph = rdflib.Graph().parse(str(kg))
    
    # get all the concepts for each cluster
    self.cluster_concept_map = {
      t: [
        f"{FRAMESTER_PREFIX % str(x).split('/')[-1]}" 
        for _, _, x in graph.triples((MUSCO[f"{t.split('/')[-1]}_cluster"], MUSCO["RelatedConcept"], None))
      ]
      for t in self.targets
    }
    
    self.cluster_concepts = np.unique(list(chain(*self.cluster_concept_map.values())))

  def __getitem__(self, idx: int) -> Tuple[np.array, str, int]:
    image, label, label_idx = super().__getitem__(idx)
    clusters = self.cluster_concept_map[label]
    return image, label, label_idx, clusters


class PerceptualARTstractDataset(ClusterARTstractDataset):
  def __init__(self, dataset: Path, kg: Path, perception_path: Path, augment: bool = True):
    """
    ARTstract dataset where for each label the corresponding cluster concepts are
    provided.

    Args:
        dataset (Path): Path to ARTstract dataset. The path should point
          to a folder containing a folder for each cluster.
        augment (bool, optional): Wether to augment the data or not. Defaults to True.
    """
    super().__init__(dataset, kg, augment)

    # read the JSON from perception Path
    self.perception = json.load(open(perception_path))
    self.perception = {
      k: [FRAMESTER_PREFIX % x.split("conceptnet:")[-1] for x in v]
      for k, v in self.perception.items()
    }


  def __getitem__(self, idx: int) -> Tuple[np.array, str, int]:
    image, label, label_idx, clusters = super().__getitem__(idx)
    return image, label, label_idx, clusters, self.perception[self.img_path[idx].stem]