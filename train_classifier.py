"""
Script to train the model that projects an image to the KGE space.
"""
import yaml
from pathlib import Path

import torch
import lightning as pl
import os
import shutil

from mosaic.model.dataset import ARTstractDataset
from mosaic.model.classification import ConvImageClassifier, ViTImageClassifier

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

  check_for_conf_field(conf, args.conf, "output")
  check_for_conf_field(conf, args.conf, "model")
  check_for_conf_field(conf["model"], args.conf, "name")
  check_for_conf_field(conf["model"], args.conf, "num_classes")
  check_for_conf_field(conf, args.conf, "dataset")
  check_for_conf_field(conf["dataset"], args.conf, "train")
  check_for_conf_field(conf["dataset"], args.conf, "valid")
  check_for_conf_field(conf["training"], args.conf, "batch_size")
  check_for_conf_field(conf["training"], args.conf, "epochs")
  check_for_conf_field(conf["training"], args.conf, "lr")
  
  pl.seed_everything(conf.get("seed", 42))

  train_data = ARTstractDataset(Path(conf["dataset"]["train"]), 
                                augment=True)
  train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=conf["training"]["batch_size"], 
                                             shuffle=True,
                                             num_workers=os.cpu_count())

  valid_data = ARTstractDataset(Path(conf["dataset"]["valid"]), 
                                augment=False)
  valid_loader = torch.utils.data.DataLoader(valid_data, 
                                             batch_size=conf["training"]["batch_size"],
                                             num_workers=os.cpu_count())

  if conf["model"]["name"] == "vit":
    model = ViTImageClassifier(
      int(conf["model"]["num_classes"]),
      only_head=True,
      lr=float(conf["training"]["lr"]),
      weights=train_data.class_weight)
  elif conf["model"]["name"] == "vit_artstract":
    model = ViTImageClassifier(
      int(conf["model"]["num_classes"]),
      only_head=False,
      lr=float(conf["training"]["lr"]),
      weights=train_data.class_weight)
  elif conf["model"]["name"] == "vgg":
    # Train using image net pretrained features
    model = ConvImageClassifier(
      int(conf["model"]["num_classes"]),
      only_head=True,
      lr=float(conf["training"]["lr"]),
      weights=train_data.class_weight)
  elif conf["model"]["name"] == "vgg_artstract":
    # Train the whole convnet
    model = ConvImageClassifier(
      int(conf["model"]["num_classes"]),
      only_head=False,
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
