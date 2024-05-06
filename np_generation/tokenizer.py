from collections.abc import Collection
from dataclasses import dataclass
from itertools import chain
from typing import Any, FrozenSet, Set
from tokenizers import Tokenizer, processors, trainers
from tokenizers.models import BPE
from tokenizers.implementations import BaseTokenizer
from tokenizers.pre_tokenizers import Whitespace

UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"


class SmilesTokenizer(BaseTokenizer):
    def __init__(self):
        self.alphabet = list(SmilesAlphabet().get_alphabet())

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{CLS_TOKEN} $0 {SEP_TOKEN}",
            special_tokens=[(CLS_TOKEN, 1), (SEP_TOKEN, 0), (PAD_TOKEN, 2)],
        )
        super().__init__(tokenizer)

    def train(self, files):
        trainer = trainers.BpeTrainer(
            special_tokens=[UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, PAD_TOKEN],
            initial_alphabet=self.alphabet,
        )
        self._tokenizer.train(files, trainer=trainer)


@dataclass(init=True, eq=False, repr=True, frozen=True)
class SmilesAlphabet(Collection):
    atoms: FrozenSet[str] = frozenset(
        [
            "Ac",
            "Ag",
            "Al",
            "Am",
            "Ar",
            "As",
            "At",
            "Au",
            "B",
            "Ba",
            "Be",
            "Bh",
            "Bi",
            "Bk",
            "Br",
            "C",
            "Ca",
            "Cd",
            "Ce",
            "Cf",
            "Cl",
            "Cm",
            "Co",
            "Cr",
            "Cs",
            "Cu",
            "Db",
            "Dy",
            "Er",
            "Es",
            "Eu",
            "F",
            "Fe",
            "Fm",
            "Fr",
            "Ga",
            "Gd",
            "Ge",
            "H",
            "He",
            "Hf",
            "Hg",
            "Ho",
            "Hs",
            "I",
            "In",
            "Ir",
            "K",
            "Kr",
            "La",
            "Li",
            "Lr",
            "Lu",
            "Md",
            "Mg",
            "Mn",
            "Mo",
            "Mt",
            "N",
            "Na",
            "Nb",
            "Nd",
            "Ne",
            "Ni",
            "No",
            "Np",
            "O",
            "Os",
            "P",
            "Pa",
            "Pb",
            "Pd",
            "Pm",
            "Po",
            "Pr",
            "Pt",
            "Pu",
            "Ra",
            "Rb",
            "Re",
            "Rf",
            "Rh",
            "Rn",
            "Ru",
            "S",
            "Sb",
            "Sc",
            "Se",
            "Sg",
            "Si",
            "Sm",
            "Sn",
            "Sr",
            "Ta",
            "Tb",
            "Tc",
            "Te",
            "Th",
            "Ti",
            "Tl",
            "Tm",
            "U",
            "V",
            "W",
            "Xe",
            "Y",
            "Yb",
            "Zn",
            "Zr",
        ]
    )

    # Bonds, charges, etc.
    non_atoms: FrozenSet[str] = frozenset(
        [
            "-",
            "=",
            "#",
            ":",
            "(",
            ")",
            ".",
            "[",
            "]",
            "+",
            "-",
            "\\",
            "/",
            "*",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "0",
            "@",
            "AL",
            "TH",
            "SP",
            "TB",
            "OH",
        ]
    )

    additional: FrozenSet[str] = frozenset()

    def __contains__(self, item: Any) -> bool:
        return item in self.atoms or item in self.non_atoms

    def __iter__(self):
        return (token for token in chain(self.atoms, self.non_atoms))

    def __len__(self) -> int:
        return len(self.atoms) + len(self.non_atoms) + len(self.additional)

    def get_alphabet(self) -> Set[str]:
        alphabet = set()
        for token in self.atoms:
            if len(token) > 1:
                alphabet.update(list(token))
                alphabet.add(token[0].lower())
            else:
                alphabet.add(token)
                alphabet.add(token.lower())
        for token in chain(self.non_atoms, self.additional):
            if len(token) > 1:
                alphabet.update(list(token))
            else:
                alphabet.add(token)
        return alphabet
