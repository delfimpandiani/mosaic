from typing import Union, List, Tuple, Iterable
from pathlib import Path

import pickle

from torchkge.models import TransEModel
import torch

class KGE:
  def __init__(self, weights_path: Path, kg_path: Path):
    """
    Initialise the Knowledge Graph Embedding method with the provided weigths and
    TorchKGE object path.

    Args:
        weights_path (Path): The path to the weigths of the KG.
        kg_path (Path): The path to the KG.
    """
    # load data info
    with open(kg_path, "rb") as f:
      self.ent2id = pickle.load(f).ent2ix

    # load model
    self.model = torch.load(weights_path)

    self._out_shape = self.model['kge.ent_emb.weight'].shape[1]
    
  @property
  def output_shape(self) -> int:
    """
    Returns:
        int: Embedding dimension of the KGE.
    """
    return self._out_shape

  def __contains__(self, iri: str) -> bool:
    """
    Check if the provided IRI is within the training set of the KGE method.

    Args:
        iri (str): Input IRI

    Returns:
        bool: True if the method contains the IRI, False otherwise.
    """
    return f"<{iri}>" in self.ent2id

  def __len__(self) -> int:
    """
    Returns:
        int: The number of samples in the KGE method.
    """
    return max(self.ent2id.values())

  def __getitem__(self, iri: Union[Iterable[str], str]) -> torch.tensor:
    """
    Compute the embedding of an IRI.

    Args:
        iri (Union[Iterable[str], str]): Single IRI or iterable of IRIs
            to be embedded.
    Returns:
        torch.tensor: A tensor of dimension ~output_shape for each
          element in the ~iri parameter.
    """
    iris = list(iri) if type(iri) == tuple else [iri,]

    for i in iris: assert i in self, f"{i} is not available!"
    
    idxs = torch.tensor([self.ent2id[f"<{i}>"] for i in iris])
    emb = self.model['kge.ent_emb.weight'][idxs]

    if emb.shape[0] == 1:
      emb = emb.squeeze()
  
    return emb