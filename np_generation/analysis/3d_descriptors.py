from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D


def calc_descriptors(smiles_list: list[str]):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [Chem.AddHs(mol) for mol in mols if mol is not None]

    for mol in mols:
        AllChem.EmbedMolecule(mol)
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except ValueError:
            pass

    # Calculate 3D descriptors
    desc_list = []
    for mol in mols:
        desc = Descriptors3D.CalcMolDescriptors3D(mol)
        print(len(desc))
        desc_list.append(desc)
    return desc_list


if __name__ == "__main__":
    smiles_list = [
        "COc1cc([C@H]2Oc3ccc(-c4cc(=O)c5c(O)cc(O)cc5o4)cc3O[C@@H]2CO)ccc1O",
        "C/C=C/C1=C(CC[C@H](C)O)COC1=O",
        "C=C[C@H]1[C@H](O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2OC(=O)c2cccc(O)c2O)OC=C2C(=O)OCC[C@H]21",
        "CC1=C[C@@H](CNC(=O)NC(C)C)[C@H](C(C)C)C[C@H]1Cc1nnc(-c2cncn2C)o1",
        "O=C(O[C@@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O)c1c(O)cc(O)cc1/C=C/c1ccccc1",
        "COc1ccc2c(c1O)COC2=O",
        "C[C@]12CC[C@H](OC(=O)CBr)C[C@@H]1CC[C@@H]1[C@H]2[C@H](O)C[C@]2(C)[C@@H](C3=CC(=O)OC3)CC[C@]12O",
    ]
    desc_list = calc_descriptors(smiles_list)
    print(desc_list)
