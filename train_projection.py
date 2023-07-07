"""
Script to train the model that projects an image to the KGE space.
"""
import yaml
from pathlib import Path

import torch
import lightning as pl
import os
import joblib

from mosaic.model.dataset import ARTstractDataset
from mosaic.model.evaluate import evaluate_cos_distance_classification
from mosaic.model.utils import stratified_split
from mosaic.model.projection import Image2KGEProjection
from mosaic.embedding import ConvImageEncoder, CLIPImageEncoder, KGE

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
  check_for_conf_field(conf["kge"], args.conf, "kge_weights")
  check_for_conf_field(conf["kge"], args.conf, "kge_data")
  check_for_conf_field(conf, args.conf, "output")
  check_for_conf_field(conf, args.conf, "encoding")
  check_for_conf_field(conf, args.conf, "dataset")
  check_for_conf_field(conf["dataset"], args.conf, "train")
  check_for_conf_field(conf["training"], args.conf, "batch_size")
  check_for_conf_field(conf["dataset"], args.conf, "valid")
  check_for_conf_field(conf["dataset"], args.conf, "test")
  check_for_conf_field(conf["training"], args.conf, "loss")
  check_for_conf_field(conf["training"], args.conf, "epochs")

  pl.seed_everything(conf.get("seed", 42))

  kge = KGE(conf["kge"]["kge_weights"], conf["kge"]["kge_data"])

  if conf["encoding"]["name"] == "clip":
    image_encoder = CLIPImageEncoder()
  elif conf["encoding"]["name"] == "vgg":
    encoder_weigths = conf["encoding"].get("image_weights", None)
    use_classifier = conf["encoding"].get("use_classifier", False)
    image_encoder = ConvImageEncoder(weights=encoder_weigths, use_classifier=use_classifier)

  train_data = ARTstractDataset(conf["dataset"]["train"], 
                                transform=image_encoder.transform)
  train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=conf["training"]["batch_size"], 
                                             shuffle=True, 
                                             num_workers=os.cpu_count())

  valid_data = ARTstractDataset(conf["dataset"]["valid"], 
                                transform=image_encoder.transform)
  valid_loader = torch.utils.data.DataLoader(valid_data, 
                                             batch_size=conf["training"]["batch_size"],
                                             num_workers=os.cpu_count())

  test_data = ARTstractDataset(conf["dataset"]["test"], 
                               transform=image_encoder.transform)
  
  model = Image2KGEProjection(image_encoder, kge, loss=cong["training"]["loss"])

  wandb_logger = pl.pytorch.loggers.WandbLogger(project="mosaic")
  wandb_logger.experiment.config.update(conf)

  trainer = pl.Trainer(
    max_epochs=conf["epochs"], 
    devices=1, 
    accelerator="gpu",
    log_every_n_steps=10,
    logger=wandb_logger,
    enable_checkpointing=False,
    callbacks=
    [
      pl.pytorch.callbacks.early_stopping.EarlyStopping(monitor="valid/loss", mode="min", patience=5)
    ] if conf["training"].get("early_stop", False) else [])
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

  y, y_pred = evaluate_cos_distance_classification(test_data, model)

  report = classification_report(y, y_pred)
  print(report)
  report_dict = classification_report(y, y_pred, output_dict=True)
  wandb_logger.log_metrics(report_dict)
  
  # export the model
  torch.save(model.state_dict(), conf["output"])
