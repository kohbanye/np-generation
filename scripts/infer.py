import os
import random
from pprint import pprint

from rdkit import Chem, RDLogger

from np_generation.generation import Generator

ROOT_DIR = os.environ["ROOT_DIR"]
RDLogger.DisableLog("rdApp.error")


def main():
    is_chiral = False
    model_dir = os.path.join(ROOT_DIR, "model", "chiral" if is_chiral else "no_chiral")

    generator = Generator(model_dir)

    smiles_list = generator.batch_generate(1000)
    pprint(smiles_list[:10])
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    mols = [m for m in mols if m is not None]
    print(len(mols) / len(smiles_list))

    filename = "generated_chiral.smi" if is_chiral else "generated_no_chiral.smi"
    with open(filename, "w") as f:
        canonical_smiles_list = [Chem.MolToSmiles(m) for m in mols]
        f.write("\n".join(canonical_smiles_list))


if __name__ == "__main__":
    main()
