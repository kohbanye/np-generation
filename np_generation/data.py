import os
import requests
from typing import Literal
import pytorch_lightning as pl
import torch
from transformers import DataCollatorForLanguageModeling

from .tokenizer import SmilesTokenizer

COCONUT_CANONICAL_SMILES = "https://coconut.naturalproducts.net/download/smiles"
COCONUT_ABSOLUTE_SMILES = "https://coconut.naturalproducts.net/download/absolutesmiles"


def _download_data(
    data_dir="data",
    filename="coconut.smi",
    smiles_type: Literal["canonical", "absolute"] = "canonical",
):
    if os.path.exists(data_dir) and os.path.exists(os.path.join(data_dir, filename)):
        return
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url = (
        COCONUT_CANONICAL_SMILES
        if smiles_type == "canonical"
        else COCONUT_ABSOLUTE_SMILES
    )
    response = requests.get(url)
    response.raise_for_status()

    with open(os.path.join(data_dir, filename), "wb") as f:
        f.write(response.content)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: SmilesTokenizer,
        split_ratio=0.9,
        batch_size=32,
        num_workers=4,
        data_dir="model",
        filename="coconut.smi",
        smiles_type="canonical",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.filename = filename
        self.smiles_type = smiles_type
        self.collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        self.smiles_list = []
        self.input_ids = []

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, self.filename)):
            _download_data(
                data_dir=self.data_dir,
                filename=self.filename,
                smiles_type=self.smiles_type,
            )
        with open(os.path.join(self.data_dir, self.filename)) as f:
            self.smiles_list = f.read().splitlines()
        self.input_ids = self.tokenizer.encode_batch(self.smiles_list)

    def setup(self, stage: str) -> None:
        train_size = int(self.split_ratio * len(self.input_ids))
        val_size = len(self.input_ids) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.input_ids, [train_size, val_size]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
