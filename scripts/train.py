import argparse
import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from transformers import GPT2Config, PreTrainedTokenizerFast

from np_generation.data import DataModule
from np_generation.model import NpGptModel
from np_generation.tokenizer import SmilesTokenizer


def train(
    root_dir: str, use_pretrained: bool, version: str, notes: str, dataset_name: str
):
    data_dir = os.path.join(root_dir, "data")
    model_dir = os.path.join(root_dir, "model", dataset_name)
    smiles_filename = f"{dataset_name}.smi"
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
    datamodule = DataModule(
        tokenizer,
        max_length=max_length,
        num_workers=192,
        batch_size=128,
        data_dir=data_dir,
        filename=smiles_filename,
        smiles_type="absolute",
    )

    torch.backends.cudnn.benchmark = True
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=max_length,
        n_ctx=max_length,
        n_embd=12 * 64,
        n_layer=6,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    )

    model = NpGptModel(config)
    if use_pretrained:
        model.load(model_dir)

    logger = WandbLogger(
        project="np-generation",
        name=f"{dataset_name}_{version}",
        version=version,
        tags=[
            "np-generation",
            dataset_name,
        ],
        notes=notes,
        save_dir=model_dir,
        log_model=True,
    )
    trainer = pl.Trainer(
        devices="auto",
        strategy="ddp",
        accelerator="gpu",
        max_epochs=30,
        logger=logger,
        default_root_dir=model_dir,
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule)
    trainer.save_checkpoint(checkpoint_path)
    model.save(model_dir)

    artifact = wandb.Artifact(
        name=dataset_name,
        type="model",
    )
    artifact.add_dir(model_dir)
    logger.experiment.log_artifact(artifact)


def main():
    parser = argparse.ArgumentParser(
        description="Training script for chemical language model"
    )

    parser.add_argument("root", help="Root directory for data and models")
    parser.add_argument(
        "--use-pretrained",
        action=argparse.BooleanOptionalAction,
        help="Whether to train pretrained model",
    )
    parser.add_argument(
        "--version",
        default="v0",
        help="Version of the model to train",
        required=False,
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Description of the experiment",
        required=False,
    )
    parser.add_argument(
        "--dataset",
        default="coconut",
        help="Name of the dataset to use (default: coconut)",
        required=False,
    )

    args = parser.parse_args()
    train(
        args.root,
        args.use_pretrained,
        args.version,
        args.notes,
        args.dataset,
    )


if __name__ == "__main__":
    main()
