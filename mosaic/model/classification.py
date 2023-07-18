from typing import Union, List, Callable
from pathlib import Path

from itertools import islice

import torch
import lightning as pl
from torchvision.models import vgg16
import timm
from transformers import AutoProcessor, CLIPModel
from PIL import Image

from typing import List, Dict
import os
import torch
from torch import nn
import numpy as np
import lightning.pytorch as pl
import math

from sklearn.metrics import classification_report

from mosaic.model.projection import Image2KGEProjection
from mosaic.embedding import ImageEncoder, KGE

class SimilarityLinkPrediction(pl.LightningModule):
  def __init__(self,
               kge: KGE,
               projection: Image2KGEProjection,
               target_links: List[str],
               device: str = "cuda"):
    """
    Initialise the Image to KGE projection method.

    Args:
        projection (Image2KGEProjection): Projection model from the image space to the KGE space.
        image_encoder (ImageEncoder): Encoder used to extract the features from the image.
        target_links (List[str]): Target links to predict the connection to.
        kge (KGE): Knowledge Graph Embedding method used to convert an entity/individual
          to a vectorial representation
        device (str, optional): Device used for the component models. Defaults to "cuda".
    """
    super().__init__()
    self._device_name = device
    self.kge = kge
    self.projection = projection.to(self._device_name)
    
    self.target = np.array(target_links)
    self.target_emb = torch.stack([self.kge[t] for t in self.target])
    # normalize the targets
    norm = self.target_emb.norm(dim=1)[:, None]
    self.target_emb = self.target_emb / torch.max(norm, 1e-8 * torch.ones_like(norm))
    self.target_emb = self.target_emb.to(self._device_name)
    
  def score_link(self, images: torch.tensor) -> torch.tensor:
    """
    Compute the similarity between the input image projected to the KGE
    space and the target KGE entities.

    Args:
        images (torch.tensor): Input image.

    Returns:
        torch.tensor: Similarity between each image and the target links.
    """
    with torch.autocast(device_type="cuda" if "cuda" in str(self._device_name) else "cpu", 
                        dtype=torch.float16):
      proj = self.projection.project(images)

      proj_norm = proj.norm(dim=1)[:, None]
      normalized_proj = proj / torch.max(proj_norm, 1e-8 * torch.ones_like(proj_norm))
      
      sim = torch.mm(normalized_proj, self.target_emb.T)
    
    return sim

  def predict(self, image: torch.tensor) -> str:
    """
    Perform a training step.

    Args:
        image (torch.tensor): The input batch.

    Returns:
        str: The output label
    """
    with torch.autocast(device_type=self._device_name, dtype=torch.float16):
      labels_sim = self.score_link(image.to(dtype=torch.float16, device=self._device_name))

      # get the most similar label
      pred_idx = labels_sim.argmax(dim=1).cpu().detach().numpy()
      preds = self.target[pred_idx]

    return preds


class ProjectedEncoderLinkPrediction(pl.LightningModule):
  def __init__(self,
               kge: KGE,
               image_encoder: ImageEncoder,
               num_classes: int,
               cluster_entities: List[str],
               device: str = "cuda",
               lr: float = 4e-4):
    """
    Initialise the Image to KGE projection method.

    Args:
        kge (KGE): Knowledge Graph Embedding method used to convert an entity/individual
          to a vectorial representation.
        image_encoder (ImageEncoder): Encoder used to extract the features from the image.
        projection (Image2KGEProjection): Projection model from the image space to the KGE space.
        target_links (List[str]): Target links to predict the connection to.
        device (str, optional): Device used for the component models. Defaults to "cuda".
        lr (float, optional): Learning rate.
    """
    super().__init__()
    self._device_name = device
    self.kge = kge
    self.image_encoder = image_encoder
    self.image_encoder.model.to(self._device_name)
    self.lr = lr
    
    self.num_classes = num_classes
    
    self.clusters_emb = torch.stack([self.kge[t] for t in cluster_entities if t in self.kge])
    self.clusters_emb = self.clusters_emb.to(self._device_name)
    
    self.projection = nn.Linear(self.image_encoder.output_shape, self.kge.output_shape)
    self.classifier = nn.Linear(self.kge.output_shape + self.clusters_emb.shape[0], num_classes)
    
  def score_link(self, images: torch.tensor) -> torch.tensor:
    """
    Compute the similarity between the input image projected to the KGE
    space and the target KGE entities.

    Args:
        images (torch.tensor): Input image.

    Returns:
        torch.tensor: Similarity between each image and the target links.
    """
    image_emb = self.image_encoder(images)
    
    proj = self.projection(image_emb)
    proj = torch.nn.functional.dropout(proj, 0.5)
    
    # compute Dot Product attention
    dot_prod_attn = torch.nn.functional.softmax((proj @ self.clusters_emb.T) / math.sqrt(proj.size(-1)))

    sim = self.classifier(torch.cat([proj, dot_prod_attn], dim=1))

    return sim, proj, image_emb

  def data_step(self, batch: torch.tensor, batch_idx: int, log_name: str = "loss", prog_bar: bool = True) -> torch.optim:
    """
    Perform a training step.

    Args:
        batch (torch.tensor): The input batch.
        batch_idx (int): The batch index.
        log_name (str, optional): The name to log the loss into. Defaults to "loss".
        prog_bar (bool, optional): Show the loss on the progression bar. Defaults to True.

    Returns:
        torch.optim: The optimizer result
    """
    image, label, label_idx = batch

    one_hot_labels = torch.nn.functional.one_hot(
      torch.tensor(label_idx.view(-1)), 
      num_classes=self.num_classes)

    sims, proj, _ = self.score_link(image)
    
    loss = torch.nn.functional.cross_entropy(sims, one_hot_labels.float())

    self.log(log_name, loss, prog_bar=prog_bar)
      
    return loss, sims

  def predict(self, image: torch.tensor) -> str:
    """
    Perform a training step.

    Args:
        image (torch.tensor): The input batch.

    Returns:
        str: The output label
    """
    with torch.autocast(device_type=self._device_name, dtype=torch.float16):
      labels_sim, _ = self.score_link(image.to(dtype=torch.float16, device=self._device_name))

      # get the most similar label
      pred_idx = labels_sim.argmax(dim=1).cpu().detach().numpy()
      preds = self.target[pred_idx]

    return preds

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    loss, _ = self.data_step(batch, batch_idx, "train/loss", True)
    return loss

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    _, _, label_idx = batch
    loss, pred =  self.data_step(batch, batch_idx, "valid/loss", False)
    
    preds_idxs = torch.argmax(pred, axis=-1)
    accuracy = (preds_idxs == label_idx.view(-1)).sum() / len(label_idx)
    self.log("valid/accuracy", accuracy)

    return loss

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer


class ConvImageClassifier(pl.LightningModule):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    """
    Initialise the image classifier using a specific model and weight.
    Train according to the findings of [1]
    
    [1] Kandel, I., & Castelli, M. (2020). 
      How deeply to fine-tune a convolutional neural network: a case study using a histopathology dataset. 
      Applied Sciences, 10(10), 3359.

    Args:
        lr (float, optional): Learning rate to be used. Defaults to 1e-4.
        only_head (bool, optional): Train only the classification head or the whole model. Defaults to the whole model.
    """
    super().__init__()
    self.lr = lr
    self.num_classes = num_classes

    self.model = vgg16(pretrained=True)
    
    if only_head:
      for param in self.model.parameters():
        param.requires_grad = False
    else:
      # freeze the layers from 0 to 24 as suggested in [1]
      for param in islice(self.model.features.parameters(), 0, 10):
        param.requires_grad = False
    
    # last two layers of the classifier are replaced with a Dropout layer
    out_shape = self.model.classifier[-1].in_features
    self.model.classifier[-1] = torch.nn.Dropout(0.5)
    
    self.classifier = nn.Linear(out_shape, self.num_classes)

  
  def data_step(self, batch, batch_idx: int, log_name: str = "loss", prog_bar: bool = True) -> torch.optim:
    """
    Perform a step on the input data.

    Args:
        batch: The input batch.
        batch_idx (int): The batch index.
        log_name (str, optional): The name to log the loss into. Defaults to "loss".
        prog_bar (bool, optional): Show the loss on the progression bar. Defaults to True.

    Returns:
        torch.optim: The optimizer result
    """
    image, label, label_idx = batch

    one_hot_labels = torch.nn.functional.one_hot(
      torch.tensor(label_idx.view(-1)), 
      num_classes=self.num_classes)

    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    loss = torch.nn.functional.cross_entropy(pred, one_hot_labels.float())
    
    self.log(log_name, loss, prog_bar=prog_bar)
      
    return loss, pred

  def predict(self, image: torch.tensor) -> str:
    """
    Perform a training step.

    Args:
        image (torch.tensor): The input batch.

    Returns:
        str: The output label
    """
    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    return torch.argmax(pred, dim=-1)

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    loss, _ = self.data_step(batch, batch_idx, "train/loss", True)
    return loss

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    _, _, label_idx = batch
    loss, pred =  self.data_step(batch, batch_idx, "valid/loss", False)

    preds_idxs = torch.argmax(pred, axis=-1)
    accuracy = (preds_idxs == label_idx.view(-1)).sum() / len(label_idx)

    self.log("valid/accuracy", accuracy)

    return loss

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer


class ViTImageClassifier(pl.LightningModule):
  def __init__(self, num_classes: int, lr: float = 1e-4, only_head: bool = False):
    """
    ViT Classifier

    Args:
        lr (float, optional): Learning rate to be used. Defaults to 1e-4.
        only_head (bool, optional): Train only the classification head or the whole model. Defaults to the whole model.
    """
    super().__init__()
    self.lr = lr
    self.num_classes = num_classes

    self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    if only_head:
      for param in self.model.parameters():
        param.requires_grad = False
    
    out_shape = self.model.head.in_features
    self.model.head = nn.Dropout(0.5)
    
    self.classifier = nn.Linear(out_shape, self.num_classes)

  def data_step(self, batch, batch_idx: int, log_name: str = "loss", prog_bar: bool = True) -> torch.optim:
    """
    Perform a step on the input data.

    Args:
        batch: The input batch.
        batch_idx (int): The batch index.
        log_name (str, optional): The name to log the loss into. Defaults to "loss".
        prog_bar (bool, optional): Show the loss on the progression bar. Defaults to True.

    Returns:
        torch.optim: The optimizer result
    """
    image, label, label_idx = batch

    one_hot_labels = torch.nn.functional.one_hot(
      torch.tensor(label_idx.view(-1)), 
      num_classes=self.num_classes)

    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    loss = torch.nn.functional.cross_entropy(pred, one_hot_labels.float())
    
    self.log(log_name, loss, prog_bar=prog_bar)
      
    return loss, pred

  def predict(self, image: torch.tensor) -> str:
    """
    Perform a training step.

    Args:
        image (torch.tensor): The input batch.

    Returns:
        str: The output label
    """
    image_emb = self.model(image)
    pred = self.classifier(image_emb)
    return torch.argmax(pred, dim=-1)

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    loss, _ = self.data_step(batch, batch_idx, "train/loss", True)
    return loss

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    _, _, label_idx = batch
    loss, pred =  self.data_step(batch, batch_idx, "valid/loss", False)

    preds_idxs = torch.argmax(pred, axis=-1)
    accuracy = (preds_idxs == label_idx.view(-1)).sum() / len(label_idx)

    self.log("valid/accuracy", accuracy)

    return loss

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer
