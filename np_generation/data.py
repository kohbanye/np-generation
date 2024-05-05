import os
from typing import Literal

import requests

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
