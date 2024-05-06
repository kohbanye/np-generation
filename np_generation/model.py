import torch
from transformers import GPT2Model
import pytorch_lightning as pl
import torch.nn as nn


class NpGptModel(pl.LightningModule):
    def __init__(self, config):
        super(NpGptModel, self).__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        outputs = self(input_ids)
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, self.config.vocab_size), labels.view(-1)
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        outputs = self(input_ids)
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, self.config.vocab_size), labels.view(-1)
        )
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)
