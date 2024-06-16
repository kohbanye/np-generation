import os
from transformers import GPT2Config, PreTrainedTokenizerFast
import pytorch_lightning as pl
import torch

from np_generation.data import DataModule
from np_generation.model import NpGptModel
from np_generation.tokenizer import SmilesTokenizer

ROOT_DIR = os.environ["ROOT_DIR"]


def main():
    data_dir = os.path.join(ROOT_DIR, "data")
    model_dir = os.path.join(ROOT_DIR, "model")
    smiles_filename = "coconut_chiral.smi"
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

    logger = pl.loggers.WandbLogger(
        project="np-generation", save_dir=model_dir, log_model=True
    )
    trainer = pl.Trainer(
        devices=4,
        strategy="ddp",
        accelerator="gpu",
        max_epochs=30,
        logger=logger,
        default_root_dir=os.path.join(ROOT_DIR, "model"),
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule)
    trainer.save_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()
