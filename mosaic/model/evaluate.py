from typing import Tuple, List

from mosaic.model.dataset import ARTstractDataset
from mosaic.model.projection import Image2KGEProjection

from sklearn.metrics.pairwise import cosine_similarity

def evaluate_cos_distance_classification(dataset: ARTstractDataset, model: Image2KGEProjection) -> Tuple[List[str], List[str]]:
  """
  Evaluate the classification results on the provided ARTstract dataset using
  the provided model.
  The classification is obtained by computing the similarity with the embeddings
  of the different labels and taking the most similar element.

  Args:
      dataset (ARTstractDataset): The input dataset
      model (Image2KGEProjection): The model used to project an image onto the KGE latent space

  Returns:
      Tuple[List[str], List[str]]: Tuple containing a list of target labels and their
          corresponding predictions. 
  """
  labels = list(set(dataset.dataset.img_label))
  assert len(labels) == 8

  labels_emb = torch.stack([model.kge[l] for l in labels]).detach().numpy()
  dataloader = torch.utils.data.DataLoader(dataset)
  
  y = []
  y_pred = []
  for sample in dataloader:
    image, label, _ = sample
    y.append(label)
    proj = model.project(image).cpu().detach().numpy()
    y_pred.append(labels[cosine_similarity(proj.reshape(1, -1), labels_emb).argmax()])
  
  return y, y_pred