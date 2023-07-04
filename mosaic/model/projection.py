import os
import torch
from torch import nn
import lightning.pytorch as pl
from info_nce import InfoNCE

from mosaic.embedding import ImageEncoder, KGE

class Image2KGEProjection(pl.LightningModule):
  def __init__(self, 
               img_encoder: ImageEncoder, 
               kge: KGE, 
               hidden_dim: int = 200, 
               hidden_layers: int = 1,
               margin: float = 1,
               device: str = "cuda"):
    """
    Initialise the Image to KGE projection method.

    Args:
        img_encoder (ImageEncoder): Encoder used to extract the features from the image.
        kge (KGE): Knowledge Graph Embedding method used to convert an entity/individual
          to a vectorial representation
        hidden_dim (int, optional): Hidden size of the projector. Defaults to 200.
        hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
        margin (float, optional): Margin used by the loss function. Defaults to 0.7.
        device (str, optional): Device used for the component models. Defaults to "cuda".
    """
    super().__init__()
    self._device = device
    self.image_encoder = img_encoder
    self.kge = kge

    self.kge_batch_norm = nn.BatchNorm1d(kge.output_shape)
    self.img_batch_norm = nn.BatchNorm1d(img_encoder.output_shape)

    hiddens = [
      nn.Linear(img_encoder.output_shape, hidden_dim),
      nn.ReLU()
    ]
    for _ in range(hidden_layers):
      hiddens.append(nn.Linear(hidden_dim, hidden_dim))
      hiddens.append(nn.ReLU())
    hiddens.append(nn.Linear(hidden_dim, kge.output_shape))

    self.projection = nn.Sequential(*hiddens)

    self.margin = margin

    self.image_encoder.model.to(self._device)
    self.projection.to(self._device)

  def project(self, image: torch.tensor) -> torch.tensor:
    """
    Perform the projection of an image to a latent space compatible
    with the KGE.

    Args:
        image (torch.tensor): Input image.

    Returns:
        torch.tensor: Projection of the input image to the KGE space.
    """
    image_emb = self.image_encoder(image.squeeze(1))

    if image.shape[0] > 1:
      image_emb = self.img_batch_norm(image_emb)

    image_emb = self.projection(image_emb)
    return image_emb

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    """
    Perform a training step.

    Args:
        batch (torch.tensor): The input batch.
        batch_idx (int): The batch index.

    Returns:
        torch.optim: The optimizer result
    """
    image, label, negative_label = batch

    with torch.autocast(device_type="cuda" if "cuda" in str(self._device) else "cpu", 
                        dtype=torch.float16):
      image_emb = self.project(image)    
      positive_emb = self.kge_batch_norm(self.kge[label].to(self.device))
      negative_emb = self.kge_batch_norm(self.kge[negative_label].to(self.device))

      loss = nn.functional.triplet_margin_with_distance_loss(
        image_emb, positive_emb, negative_emb, 
        swap=True,
        margin=self.margin,
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y))
      
    self.log("train/loss", loss, prog_bar=True)
    
    return loss

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    """
    Perform a validation step.

    Args:
        batch (torch.tensor): The input batch.
        batch_idx (int): The batch index.

    Returns:
        torch.optim: The optimizer result
    """
    image, label, negative_label = batch

    with torch.autocast(device_type="cuda" if "cuda" in str(self._device) else "cpu", 
                        dtype=torch.float16):
      image_emb = self.project(image)    
      positive_emb = self.kge_batch_norm(self.kge[label].to(self.device))
      negative_emb = self.kge_batch_norm(self.kge[negative_label].to(self.device))

      loss = nn.functional.triplet_margin_with_distance_loss(
        image_emb, positive_emb, negative_emb, 
        swap=True,
        margin=self.margin,
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y))
      
    self.log("valid/loss", loss)
    
    return loss

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer