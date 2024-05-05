from transformers import GPT2Config
import pytorch_lightning as pl

from np_generation.data import DataModule
from np_generation.model import NpGptModel
from np_generation.tokenizer import SmilesTokenizer


def main():
    data_path = "model/coconut.smi"
    checkpoint_path = "model/checkpoint.ckpt"
    tokenizer_path = "model"

    tokenizer = SmilesTokenizer()
    tokenizer.train([data_path])
    tokenizer.save(tokenizer_path)

    datamodule = DataModule(tokenizer, filename="coconut_chiral.smi")

    config = GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=512,
        n_ctx=512,
        n_embd=12 * 64,
        n_layer=6,
        n_head=12,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
    )
    model = NpGptModel(config)

    logger = pl.loggers.WandbLogger(project="np-generation")
    trainer = pl.Trainer(
        devices=4,
        strategy="ddp",
        accelerator="gpu",
        max_epochs=30,
        logger=logger,
    )

    trainer.fit(model, datamodule)
    model.transformer.save_pretrained(checkpoint_path)


if __name__ == "__main__":
    main()
