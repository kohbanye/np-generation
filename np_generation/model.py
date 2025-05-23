import pytorch_lightning as pl
import torch
from transformers import GPT2LMHeadModel


class NpGptModel(pl.LightningModule):
    def __init__(self, config):
        super(NpGptModel, self).__init__()
        self.config = config
        self.transformer = GPT2LMHeadModel(config)

    def forward(self, input_ids, labels, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask
        )
        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        outputs = self(input_ids, labels)
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        outputs = self(input_ids, labels)
        self.log("val_loss", outputs.loss, sync_dist=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)

    def save(self, save_dir):
        self.transformer.save_pretrained(save_dir)

    def load(self, model_dir):
        self.transformer = GPT2LMHeadModel.from_pretrained(model_dir)
