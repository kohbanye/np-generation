from tokenizers import Tokenizer, processors, trainers
from tokenizers.models import BPE
from tokenizers.implementations import BaseTokenizer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFKC, Sequence

UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"


class SmilesTokenizer(BaseTokenizer):
    def __init__(self):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{CLS_TOKEN} $0 {SEP_TOKEN}",
            special_tokens=[(CLS_TOKEN, 1), (SEP_TOKEN, 0)],
        )
        super().__init__(tokenizer)

    def train(self, files):
        trainer = trainers.BpeTrainer(special_tokens=[UNK_TOKEN, CLS_TOKEN, SEP_TOKEN])
        self._tokenizer.train(files, trainer=trainer)

    def save(self, path):
        self._tokenizer.model.save(path)
