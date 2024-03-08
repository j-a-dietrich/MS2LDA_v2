from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFMCS
import numpy as np

from CDK_pywrapper import CDK, FPType
import warnings

# Only important to calculate descriptors and we are not doing that
warnings.filterwarnings("ignore", message="Molecule lacks hydrogen atoms.*") 



def smiles2mols(smiles_per_motif):
    """convert smiles to rdkit mol object"""
    mols = []
    for smiles in smiles_per_motif:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    return mols


def mols2fps(mols_per_motif, fp_type):
    """calculates the selected fingerprint for a given list of rdkit mol objects"""
    fps_generator = CDK(fingerprint=fp_type)
    fps = fps_generator.calculate(mols_per_motif, show_banner=False)

    return fps


def scale_fps(fps_per_motif):
    """defines the number of aligning SMARTS patterns"""
    cumulative_fps = fps_per_motif.sum().to_numpy()
    scaled_fps = cumulative_fps/len(fps_per_motif.index)

    return scaled_fps


def fps2motif(scaled_fps, threshold):
    """overlaps fingerprints of compounds allocated to the same topic/motif"""
    # above_threshold_indices = np.where(scaled_fps > threshold)[0] # useful for retrieval, but maybe you can do it in another function
    # maybe you can use the masking also for the retrieveal of SMARTS patterns

    lower_as_threshold = scaled_fps < threshold
    higher_as_threshold = scaled_fps >= threshold

    scaled_fps[lower_as_threshold] = 0
    scaled_fps[higher_as_threshold] = 1

    return scaled_fps


def annotate_motifs(smiles_per_motif, fp_type=FPType.MACCSFP, threshold=0.8):
    """runs all the scripts to generate a selected fingerprint for a motif"""
    mols_per_motif = smiles2mols(smiles_per_motif)
    fps_per_motif = mols2fps(mols_per_motif, fp_type)
    scaled_fps = scale_fps(fps_per_motif)
    fps_motif = fps2motif(scaled_fps, threshold)
   
    return fps_motif



if __name__ == "__main__":

    smiles_per_motif = ["O=C(C)Oc1ccccc1C(=O)O", "COC(=O)C1CCC(C1)C(=O)O"]
    fps_motif = annotate_motifs(smiles_per_motif, fp_type=FPType.SubFP)
