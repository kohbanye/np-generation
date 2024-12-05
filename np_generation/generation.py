from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GenerationConfig
import torch
from typing import Optional


class Generator:
    def __init__(self, model_dir: str, config: Optional[GenerationConfig] = None):
        self.model_dir = model_dir
        self.config = config
        if config is None:
            self.config = GenerationConfig(
                max_length=512,
                temperature=1.2,
                do_sample=True,
                top_p=0.84,
                top_k=100,
            )
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(
            model_dir, pad_token_id=3
        )

    def generate(self, inputs: Optional[torch.Tensor]) -> list[str]:
        outputs = self.model.generate(inputs=inputs, generation_config=self.config)
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [output.replace(" ", "") for output in decoded_outputs]

    def batch_generate(self, num: int) -> list[str]:
        cls_token_id = 1
        inputs = torch.full((num, 1), cls_token_id, dtype=int)
        return self.generate(inputs)
