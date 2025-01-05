import pandas as pd
import pytorch_lightning as pl
import torch
from dataset import get_dataloader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForSequenceClassification


class EssayClassifier(pl.LightningModule):
    def __init__(
        self, model_name="bert-base-uncased", num_classes=1, learning_rate=1e-5
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].float()
        logits = self(input_ids, attention_mask).squeeze(-1)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"].float()
        logits = self(input_ids, attention_mask).squeeze(-1)
        loss = self.loss_fn(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)


def split_train_val(csv_path, train_output, val_output, val_size=0.2, random_state=42):
    data = pd.read_csv(csv_path)
    train_data, val_data = train_test_split(
        data, test_size=val_size, random_state=random_state
    )
    train_data.to_csv(train_output, index=False)
    val_data.to_csv(val_output, index=False)

    print(f"Training data saved to {train_output}")
    print(f"Validation data saved to {val_output}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    split_train_val(
        "../data/train_essays.csv",
        "..data/train_split.csv",
        "..data/val_split.csv",
        val_size=0.2,
    )

    train_loader = get_dataloader(
        "..data/train_split.csv", tokenizer, batch_size=16, shuffle=True
    )
    val_loader = get_dataloader(
        "..data/val_split.csv", tokenizer, batch_size=16, shuffle=False
    )
    model = EssayClassifier()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best_model-{epoch}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    logger = CSVLogger(save_dir="logs")

    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(model.state_dict(), "final_model.pth")
    print("Training complete. Final model saved at final_model.pth")


if __name__ == "__main__":
    main()
