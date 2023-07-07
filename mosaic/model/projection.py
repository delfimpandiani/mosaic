import os
import torch
from torch import nn
import lightning.pytorch as pl
from info_nce import InfoNCE

from mosaic.embedding import ConvImageEncoder, KGE

class Image2KGEProjection(pl.LightningModule):
  def __init__(self, 
               img_encoder: ConvImageEncoder, 
               kge: KGE,  
               device: str = "cuda",
               loss: str = "cosine"):
    """
    Initialise the Image to KGE projection method.

    Args:
        img_encoder (ImageEncoder): Encoder used to extract the features from the image.
        kge (KGE): Knowledge Graph Embedding method used to convert an entity/individual
          to a vectorial representation
        device (str, optional): Device used for the component models. Defaults to "cuda".
        loss(str, optional): Loss to be used. Defaults to "cosine".
    """
    super().__init__()
    self._device = device
    self.image_encoder = img_encoder
    self.kge = kge

    self.projection = nn.Linear(img_encoder.output_shape, kge.output_shape)
    
    self.loss_name = loss
    if loss == "cosine":
      self.loss = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - nn.functional.cosine_similarity(x, y))
    elif loss == "infonce":
      self.loss = InfoNCE(negative_mode="paired")

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
    image_emb = self.projection(image_emb)
    return image_emb

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
    image, label, negative_label = batch

    with torch.autocast(device_type="cuda" if "cuda" in str(self._device) else "cpu", 
                        dtype=torch.float16):
      image_emb = self.project(image)    
      positive_emb = self.kge[label].to(self.device)
      negative_emb = self.kge[negative_label].to(self.device)

      if self.loss_name == "infonce":
        loss = self.loss(image_emb, positive_emb, negative_emb.unsqueeze(1))
      elif self.loss_name == "cosine":
        loss = self.loss(image_emb, positive_emb, negative_emb)
    
    self.log(log_name, loss, prog_bar=prog_bar)
    
    return loss

  def training_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    return self.data_step(batch, batch_idx, "train/loss", True)

  def validation_step(self, batch: torch.tensor, batch_idx: int) -> torch.optim:
    return self.data_step(batch, batch_idx, "valid/loss", False)

  def configure_optimizers(self) -> torch.optim:
    """
    Returns:
        torch.optim: Create the optimizer for the neural network
    """
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    return optimizer