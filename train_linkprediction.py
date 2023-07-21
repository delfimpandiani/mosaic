"""
Script to train the model that projects an image to the KGE space.
"""
import yaml
from pathlib import Path

import torch
import lightning as pl
import os
import joblib

from mosaic.model.dataset import ClusterARTstractDataset
from mosaic.model.evaluate import evaluate_cos_distance_classification
from mosaic.model.utils import stratified_split
from mosaic.model.projection import Image2KGEProjection
from mosaic.embedding import KGE, VGGImageEncoder, ViTImageEncoder
from mosaic.model.classification import ProjectedSimPrediction

import yaml
from pathlib import Path

import torch
import lightning as pl
import os
import shutil

from mosaic.model.dataset import ARTstractDataset

from argparse import ArgumentParser

from sklearn.metrics import classification_report

from argparse import ArgumentParser

argument_parser = ArgumentParser(description="Train the projection layer from an image encoder to a Knowledge Graph Embedding.")
argument_parser.add_argument("-c", "--conf", required=True, type=Path, help="Path to the configuration file.")

def check_for_conf_field(conf, conf_path, field):
  assert field in conf, f"{field} is missing in {conf_path}"
  
if __name__ == "__main__":
  args = argument_parser.parse_args()
  with open(args.conf, "r") as f:
    try:
        conf = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError("Configuration file is not readable: ", exc)

  check_for_conf_field(conf, args.conf, "kge")
  check_for_conf_field(conf["kge"], args.conf, "weights")
  check_for_conf_field(conf["kge"], args.conf, "data")
  check_for_conf_field(conf, args.conf, "output")
  check_for_conf_field(conf, args.conf, "encoder")
  check_for_conf_field(conf, args.conf, "dataset")
  check_for_conf_field(conf["dataset"], args.conf, "train")
  check_for_conf_field(conf["dataset"], args.conf, "kg")
  check_for_conf_field(conf["training"], args.conf, "batch_size")
  check_for_conf_field(conf["dataset"], args.conf, "valid")
  check_for_conf_field(conf["training"], args.conf, "epochs")
  check_for_conf_field(conf["training"], args.conf, "lr")
  
  check_for_conf_field(conf, args.conf, "prediction")
  check_for_conf_field(conf["prediction"], args.conf, "targets")

  pl.seed_everything(conf.get("seed", 42))

  kge = KGE(conf["kge"]["weights"], conf["kge"]["data"])
  num_classes = len(conf["prediction"]["targets"])

  encoder_weigths = conf["encoder"].get("weights", None)
  if conf["encoder"]["name"] == "vgg":
    image_encoder = VGGImageEncoder()
  elif conf["encoder"]["name"] == "vgg_artstract":
    image_encoder = VGGImageEncoder(Path(conf["encoder"]["weights"]))
  elif conf["encoder"]["name"] == "vit":
    image_encoder = ViTImageEncoder()
  elif conf["encoder"]["name"] == "vit_artstract":
    image_encoder = ViTImageEncoder(Path(conf["encoder"]["weights"]))

  def collate_fn(batch):
    image, label, label_idx, clusters = zip(*batch)
    return torch.stack(image), label, torch.stack(label_idx), clusters
  
  train_data = ClusterARTstractDataset(Path(conf["dataset"]["train"]), Path(conf["dataset"]["kg"]), augment=True)
  train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=conf["training"]["batch_size"], 
                                             shuffle=True, 
                                             num_workers=os.cpu_count(),
                                             collate_fn=collate_fn)

  valid_data = ClusterARTstractDataset(Path(conf["dataset"]["valid"]), Path(conf["dataset"]["kg"]), augment=False)
  valid_loader = torch.utils.data.DataLoader(valid_data, 
                                             batch_size=conf["training"]["batch_size"],
                                             num_workers=os.cpu_count(),
                                             collate_fn=collate_fn)

  model = ProjectedSimPrediction(
    kge, image_encoder, len(train_data.targets), train_data.cluster_concept_map,
    lr=float(conf["training"]["lr"]),
    weights=train_data.class_weight)

  wandb_logger = pl.pytorch.loggers.WandbLogger(project="mosaic", name=conf["output"].split("/")[-1])
  wandb_logger.experiment.config.update(conf)

  if Path(conf["output"]).exists():
    shutil.rmtree(conf["output"])
  Path(conf["output"]).mkdir()
  
  callbacks = [
    pl.pytorch.callbacks.ModelCheckpoint(
      dirpath=conf["output"],
      save_top_k=1,
      monitor="valid/accuracy",
      mode="max",
      filename="model"),
    pl.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
  ]

  trainer = pl.Trainer(
    max_epochs=conf["training"]["epochs"],
    devices=1, 
    accelerator="gpu",
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=callbacks)

  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
