from vina import Vina

class Docking:
    def __init__(self, ligand, receptor, outdir):
        self.vina = Vina()
        self.ligand = ligand
        self.receptor = receptor
        self.outdir = outdir
        self.docking = None
