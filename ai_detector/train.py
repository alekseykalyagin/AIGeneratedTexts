import os

import dvc.api
import dvc.repo
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from classifiers import EssayClassifier
from dataset import get_dataloader
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# Hydra config path
@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # Using OmegaConf to load the data paths and model configurations
    data_cfg = cfg["data"]  # Loads data paths configuration
    model_cfg = cfg["model"]  # Loads model configuration

    # DVC data path
    data_path = data_cfg.data_path
    pull_dvc_data(data_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    # Split train/validation data
    split_train_val(
        data_cfg.data_path,
        data_cfg.train_output,
        data_cfg.val_output,
        val_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
    )

    # Data loaders
    train_loader = get_dataloader(
        data_cfg.train_output, tokenizer, batch_size=data_cfg.batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        data_cfg.val_output, tokenizer, batch_size=data_cfg.batch_size, shuffle=False
    )

    # Model and training setup
    model = EssayClassifier()
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_cfg.checkpoint_dir,
        filename=model_cfg.checkpoint_filename,
        save_top_k=1,
        monitor=model_cfg.monitor_metric,
        mode=model_cfg.monitor_mode,
    )
    logger = CSVLogger(save_dir=model_cfg.log_dir)

    trainer = pl.Trainer(
        max_epochs=model_cfg.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
    )

    # Training the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Save final model
    final_model_path = model_cfg.final_model_path
    os.makedirs(model_cfg.model_dir, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    # Push model to DVC
    add_and_push_dvc_model(final_model_path)


def split_train_val(csv_path, train_output, val_output, val_size=0.2, random_state=42):
    data = pd.read_csv(csv_path)
    train_data, val_data = train_test_split(
        data, test_size=val_size, random_state=random_state
    )
    train_data.to_csv(train_output, index=False)
    val_data.to_csv(val_output, index=False)


def pull_dvc_data(data_path):
    repo = dvc.repo.Repo()
    repo.pull(data_path)
    print(f"Pulled data from DVC: {data_path}")


def add_and_push_dvc_model(model_path):
    repo = dvc.repo.Repo()
    repo.add(model_path)
    os.system(f"git add {model_path}.dvc .gitignore")
    os.system(f'git commit -m "Added {model_path} to DVC"')
    repo.push(model_path)
    print(f"Pushed model to DVC: {model_path}")


if __name__ == "__main__":
    main()
