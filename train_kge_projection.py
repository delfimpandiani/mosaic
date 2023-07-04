"""
Script to train the model that projects an image to the KGE space.
"""
from pathlib import Path

import torch
import lightning as pl
import os
import joblib

from mosaic.model.dataset import ARTstractDataset, evaluate_classification
from mosaic.model.utils import stratified_split
from mosaic.model.projection import Image2KGEProjection
from mosaic.embedding import ImageEncoder, KGE

from sklearn.metrics import classification_report

from argparse import ArgumentParser

argument_parser = ArgumentParser(description="Train the projection layer from an image encoder to a Knowledge Graph Embedding.")
argument_parser.add_argument("--kge-weights", required=True, type=Path, help="Path to the KGE weights file.")
argument_parser.add_argument("--kge-data", required=True, type=Path, help="Path to the KGE joblib file.")
argument_parser.add_argument("--image-weights", required=False, type=Path, help="Path to custom vgg16 trained models.", default=None)
argument_parser.add_argument("-o", "--output", required=True, type=str, help="Output file where training results are saved.")
argument_parser.add_argument("--data", required=True, type=Path, help="Path to the ARTstract dataset.")
argument_parser.add_argument("--hidden-layers", required=False, default=1, type=int, help="Hidden layers in the projection model.")
argument_parser.add_argument("--epochs", required=False, default=8, type=int, help="Hidden layers in the projection model.")
argument_parser.add_argument("--seed", required=False, default=42, type=int, help="Random seed to use.")
argument_parser.add_argument("--test-fraction", required=False, default=0.8, type=float, help="Fraction of samples used as testing samples.")
argument_parser.add_argument("--batch-size", required=False, default=128, type=int, help="Batch size for training.")
argument_parser.add_argument("--classifier-layers", required=False, default=False, action="store_true", help="Use the layers of the classifier.")

if __name__ == "__main__":
  args = argument_parser.parse_args()

  pl.seed_everything(args.seed)

  kge = KGE(args.kge_weights, args.kge_data)
  image_encoder = ImageEncoder(weights=args.image_weights, 
                               use_classifier=args.classifier_layers)
  dataset = ARTstractDataset(args.data, transform=image_encoder.transform)
  train_data, train_labels, test_data, _ = stratified_split(dataset, dataset.img_label, 
                                                            fraction=args.test_fraction, 
                                                            random_state=args.seed)
  train_data, train_labels, valid_data, _ = stratified_split(dataset, dataset.img_label, 
                                                             fraction=0.9, 
                                                             random_state=args.seed)

  train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             num_workers=os.cpu_count())

  valid_loader = torch.utils.data.DataLoader(valid_data, 
                                             batch_size=args.batch_size, 
                                             num_workers=os.cpu_count())

  model = Image2KGEProjection(image_encoder, kge, hidden_layers=args.hidden_layers)

  wandb_logger = pl.pytorch.loggers.WandbLogger(project="mosaic")
  wandb_logger.experiment.config.update(vars(args))

  trainer = pl.Trainer(
    max_epochs=args.epochs, 
    devices=1, 
    accelerator="gpu",
    log_every_n_steps=10,
    logger=wandb_logger,
    callbacks=[
      pl.pytorch.callbacks.early_stopping.EarlyStopping(monitor="valid/loss", mode="min")
    ])
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

  y, y_pred = evaluate_classification(test_data, model)

  report = classification_report(y, y_pred)
  print(report)
  report_dict = classification_report(y, y_pred, output_dict=True)
  wandb_logger.log_metrics(report_dict)
  
  # export the model
  joblib.dump({"model": model, "results": {"y": y, "y_pred": y_pred}}, args.output)
