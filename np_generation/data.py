import os
from typing import Literal

import pytorch_lightning as pl
import requests
import torch
from rdkit import Chem
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

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


class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        split_ratio=0.9,
        batch_size=16,
        num_workers=4,
        max_length=512,
        data_dir="data",
        filename="coconut.smi",
        smiles_type="canonical",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
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

        if os.path.exists(os.path.join(self.data_dir, "input_ids.pt")):
            self.input_ids = torch.load(os.path.join(self.data_dir, "input_ids.pt"))
        else:
            self.input_ids = self.tokenizer(
                self.smiles_list,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )["input_ids"]
            torch.save(self.input_ids, os.path.join(self.data_dir, "input_ids.pt"))

    def setup(self, stage: str) -> None:
        if not os.path.exists(os.path.join(self.data_dir, "input_ids.pt")):
            raise ValueError("Tokenized data do not exist.")
        self.input_ids = torch.load(os.path.join(self.data_dir, "input_ids.pt"))

        full_dataset = SmilesDataset(self.input_ids)
        num_train = int(len(full_dataset) * self.split_ratio)
        num_val = len(full_dataset) - num_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [num_train, num_val]
        )

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty.")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty.")
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


class CustomSmilesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        split_ratio=0.9,
        batch_size=16,
        num_workers=4,
        max_length=512,
        data_dir="data",
        filename="coconut.smi",
        smiles_type="canonical",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.data_dir = data_dir
        self.filename = filename
        self.smiles_type = smiles_type
        self.collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        self.smiles_list = []
        self.input_ids = []

    def _generate_custom_smiles(self, smiles: str, max_attempts: int = 2**16) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        custom_smiles = Chem.MolToSmiles(mol, doRandom=True)
        cnt = 0
        while "@@" in custom_smiles:
            if cnt >= max_attempts:
                raise ValueError("Failed to generate custom SMILES.")
            custom_smiles = Chem.MolToSmiles(mol, doRandom=True)
            cnt += 1
        return custom_smiles

    def prepare_data(self):
        if not os.path.exists(os.path.join(self.data_dir, self.filename)):
            _download_data(
                data_dir=self.data_dir,
                filename=self.filename,
                smiles_type=self.smiles_type,
            )
        with open(os.path.join(self.data_dir, self.filename)) as f:
            smiles_data = f.read().splitlines()

        input_ids_filename = "input_ids_custom_smiles.pt"
        if os.path.exists(os.path.join(self.data_dir, input_ids_filename)):
            self.input_ids = torch.load(os.path.join(self.data_dir, input_ids_filename))
            return

        for smiles in tqdm(smiles_data, desc="Generating custom SMILES"):
            try:
                custom_smiles = self._generate_custom_smiles(smiles)
            except ValueError:
                continue
            self.smiles_list.append(custom_smiles)

        custom_smiles_filename = "coconut_custom_smiles.smi"
        with open(os.path.join(self.data_dir, custom_smiles_filename), "w") as f:
            f.write("\n".join(self.smiles_list))

        self.input_ids = self.tokenizer(
            self.smiles_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"]
        torch.save(self.input_ids, os.path.join(self.data_dir, input_ids_filename))

    def setup(self, stage: str) -> None:
        if not os.path.exists(
            os.path.join(self.data_dir, "input_ids_custom_smiles.pt")
        ):
            raise ValueError("Tokenized data do not exist.")
        self.input_ids = torch.load(
            os.path.join(self.data_dir, "input_ids_custom_smiles.pt")
        )

        full_dataset = SmilesDataset(self.input_ids)
        num_train = int(len(full_dataset) * self.split_ratio)
        num_val = len(full_dataset) - num_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [num_train, num_val]
        )

    def train_dataloader(self):
        if len(self.train_dataset) == 0:
            raise ValueError("Training dataset is empty.")
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        if len(self.val_dataset) == 0:
            raise ValueError("Validation dataset is empty.")
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


if __name__ == "__main__":
    from np_generation.tokenizer import SmilesTokenizer

    root_dir = "/gs/bs/tga-ohuelab/sakano/np-generation"
    chiral_training = True

    data_dir = os.path.join(root_dir, "data")
    if chiral_training:
        model_dir = os.path.join(root_dir, "model", "chiral")
        smiles_filename = "coconut_chiral.smi"
    else:
        model_dir = os.path.join(root_dir, "model", "no_chiral")
        smiles_filename = "coconut.smi"
    checkpoint_path = os.path.join(model_dir, "checkpoint.ckpt")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")

    if os.path.exists(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    else:
        tokenizer = SmilesTokenizer()
        tokenizer.train([os.path.join(data_dir, smiles_filename)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.save_pretrained(model_dir)

    max_length = 512
    datamodule = CustomSmilesDataModule(
        tokenizer,
        max_length=max_length,
        num_workers=192,
        batch_size=128,
        data_dir=data_dir,
        filename=smiles_filename,
        smiles_type="absolute",
    )

    datamodule.prepare_data()
