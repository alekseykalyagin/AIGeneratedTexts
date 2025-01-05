import pytorch_lightning as pl
from torch.optim import AdamW
import torch
from transformers import BertForSequenceClassification


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
