import random

from rdkit import Chem


def assign_random_chirality(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    for center in chiral_centers:
        idx = center[0]
        atom = mol.GetAtomWithIdx(idx)
        chiral_tag = random.choice(
            [
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            ]
        )
        atom.SetChiralTag(chiral_tag)

    return Chem.MolToSmiles(mol, isomericSmiles=True)
