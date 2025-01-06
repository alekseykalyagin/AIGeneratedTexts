import logging
import os

import hydra
import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from classifiers import EssayClassifier
from dataset import get_dataloader
from dvc.repo import Repo
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_train_val(csv_path, train_output, val_output, val_size=0.2, random_state=42):
    try:
        data = pd.read_csv(csv_path)
        train_data, val_data = train_test_split(
            data, test_size=val_size, random_state=random_state
        )
        train_data.to_csv(train_output, index=False)
        val_data.to_csv(val_output, index=False)

        repo = Repo()

        repo.add(train_output)
        repo.scm.add([f"{train_output}.dvc", ".gitignore"])
        repo.scm.commit(f"Added {train_output} to DVC")
        repo.push()

        repo.add(val_output)
        repo.scm.add([f"{val_output}.dvc", ".gitignore"])
        repo.scm.commit(f"Added {val_output} to DVC")
        repo.push()

        logger.info("Successfully split and tracked train/val datasets in DVC.")
    except Exception as e:
        logger.error(f"Error splitting datasets: {e}")
        raise


def pull_dvc_data():
    try:
        repo = Repo()
        repo.pull()
        logger.info("Successfully pulled data from DVC.")
    except Exception as e:
        logger.error(f"Error pulling data from DVC: {e}")
        raise


def add_and_push_dvc_model(model_path):
    try:
        repo = Repo()
        repo.add(model_path)
        repo.scm.add([f"{model_path}.dvc", ".gitignore"])
        repo.scm.commit(f"Added {model_path} to DVC")
        repo.push()
        logger.info(f"Successfully pushed model to DVC: {model_path}")
    except Exception as e:
        logger.error(f"Error pushing model to DVC: {e}")
        raise


@hydra.main(config_path="../configs", config_name="config")
def main(cfg):
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    mlflow_cfg = cfg["mlflow"]

    pull_dvc_data()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)

    split_train_val(
        data_cfg.data_path,
        data_cfg.train_output,
        data_cfg.val_output,
        val_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
    )

    train_loader = get_dataloader(
        data_cfg.train_output, tokenizer, batch_size=data_cfg.batch_size, shuffle=True
    )
    val_loader = get_dataloader(
        data_cfg.val_output, tokenizer, batch_size=data_cfg.batch_size, shuffle=False
    )

    model = EssayClassifier()

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_cfg.checkpoint_dir,
        filename=model_cfg.checkpoint_filename,
        save_top_k=1,
        monitor=model_cfg.monitor_metric,
        mode=model_cfg.monitor_mode,
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=mlflow_cfg.experiment_name,
        tracking_uri=mlflow_cfg.tracking_uri,
        log_model=True,
    )

    trainer = pl.Trainer(
        max_epochs=model_cfg.max_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
    )

    mlflow_logger.log_hyperparams(model_cfg)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_model_path = model_cfg.final_model_path
    os.makedirs(model_cfg.model_dir, exist_ok=True)
    torch.save(model.state_dict(), final_model_path)

    mlflow.pytorch.log_model(model, artifact_path="model")

    add_and_push_dvc_model(final_model_path)


if __name__ == "__main__":
    main()
