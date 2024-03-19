from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFMCS

import numpy as np
from itertools import combinations

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
    # error with Nan if cluster is empty
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


def fps2smarts(fps_motif, fp_type):
    """returns substructures SMARTS pattern for overlaping bit in fingerprint for given fp"""
    one_indices = np.where(fps_motif == 1)[0]
    smarts = []

    if fp_type == FPType.MACCSFP:
        from fingerprints.MACCS import return_SMARTS
        
        for one_index in one_indices:
            maccs_smarts = return_SMARTS(one_index)
            smarts.append(maccs_smarts)

    elif fp_type == FPType.KRFP:
        from fingerprints.klekota_roth import return_SMARTS
        
        for one_index in one_indices:
            maccs_smarts = return_SMARTS(one_index)
            smarts.append(maccs_smarts)
    
    return smarts


def motifs2tanimotoScore(fps_motifs):
    """tanimoto similarity for two given motifs"""
    motifs_similarities = []

    motifs_index_combinations = combinations(range(len(fps_motifs)),2)
    for motif_A_index, motif_B_index in motifs_index_combinations:
        #print(motif_A_index, motif_B_index)
        intersection = 0
        union = 0
        for motif_A_bit, motif_B_bit in zip(fps_motifs[motif_A_index], fps_motifs[motif_B_index]):
            if motif_A_bit == 1 and motif_B_bit == 1:
                intersection += 1
            if motif_A_bit == 1 or motif_B_bit == 1:
                union += 1

        motifs_similarity = intersection / union
        motifs_similarities.append(motifs_similarity)

    return motifs_similarities



def annotate_motifs(smiles_per_motifs, fp_type=FPType.MACCSFP, threshold=0.8):
    """runs all the scripts to generate a selected fingerprint for a motif"""
    fps_motifs = []
    smarts_per_motifs = []
    for smiles_per_motif in smiles_per_motifs:
        mols_per_motif = smiles2mols(smiles_per_motif)
        fps_per_motif = mols2fps(mols_per_motif, fp_type)
        scaled_fps = scale_fps(fps_per_motif)
        fps_motif = fps2motif(scaled_fps, threshold)
        smarts_per_motif = fps2smarts(fps_motif, fp_type)

        fps_motifs.append(fps_motif)
        smarts_per_motifs.append(smarts_per_motif)

    motifs_similarities = motifs2tanimotoScore(fps_motifs)

    return fps_motifs, smarts_per_motifs, motifs_similarities



if __name__ == "__main__":

    smiles_per_motifs = [["O=C(C)Oc1ccccc1C(=O)O", "COC(=O)C1CCC(C1)C(=O)O"]]
    fps_motif = annotate_motifs(smiles_per_motifs, fp_type=FPType.SubFP)
