from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFMCS

import numpy as np
from itertools import combinations

import multiprocessing

from CDK_pywrapper import CDK, FPType # Everyone uses PADDEL, maybe I should use it too?
import warnings

# Only important to calculate descriptors and we are not doing that
warnings.filterwarnings("ignore", message="Molecule lacks hydrogen atoms.*") 



def smiles2mols(smiles_per_motif):
    """convert smiles to rdkit mol object
    
    ARGS:
        smiles_per_motif (list(str)): list of smiles that are associated with one motif

    RETURNS:
        mols (list(rdkit.mol.objects)): list of rdkit.mol.objects from the smiles
    
    !!! Currently only valid smiles are allowed; program could break if invalid smiles are given
    """
    mols = []
    for smiles in smiles_per_motif:
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)

    return mols


def mols2fps(mols_per_motif, fp_type):
    """calculates the selected fingerprint for a given list of rdkit mol objects
    
    ARGS:
        mols_per_motif (list(rdkit.mol.objects)): list of rdkit.mol.objects associated with one motif
        fp_type (CDK_pywrapper.fp_type.object): a object that represents a type of fingerprint that will be calculated

    RETURNS:
        fps (pandas.dataframe): a dataframe (rows are molecules and columns are fingerprint bit) for all molecular fingerprints
    """
    fps_generator = CDK(fingerprint=fp_type)
    fps = fps_generator.calculate(mols_per_motif, show_banner=False)

    return fps


def scale_fps(fps_per_motif):
    """calculates the percentage of the presents of every fingerprint bit in a motif
    
    ARGS:
        fps_per_motif (pandas.dataframe): a dataframe (rows are molecules and columns are fingerprint bit) for all molecular fingerprints

    RETURNS:
        scaled_fps (np.array): a fingerprint array with values between 0 and 1 showing the presents of substructures within a motif
    
    """
    cumulative_fps = fps_per_motif.sum().to_numpy()
    scaled_fps = cumulative_fps/len(fps_per_motif.index)
    # error with Nan if cluster is empty
    return scaled_fps


def fps2motif(scaled_fps, threshold):
    """overlaps fingerprints of compounds allocated to the same topic/motif
    
    ARGS:
        scaled_fps (np.array): a fingerprint array with values between 0 and 1 showing the presents of substructures within a motif
        threshold (float; 0 > x <= 1): number that defines if a bit in the fingerprint with be set to zero (below threshold) or to one (above threshold)

    RETURNS:
        scaled_fps (np.array): could also be called motif_fps, because it represents the most common fingerprint bits in a motif (bits above the threshold)
    """
    # above_threshold_indices = np.where(scaled_fps > threshold)[0] # useful for retrieval, but maybe you can do it in another function
    # maybe you can use the masking also for the retrieveal of SMARTS patterns

    lower_as_threshold = scaled_fps < threshold
    higher_as_threshold = scaled_fps >= threshold

    scaled_fps[lower_as_threshold] = 0
    scaled_fps[higher_as_threshold] = 1

    return scaled_fps


def fps2smarts(fps_motif, fp_type):
    """returns substructures SMARTS pattern for overlaping bit in fingerprint for given fp
    
    ARGS:
        fps_motif (np.array): represents the most common fingerprint bits in a motif (bits above the set threshold)
        fp_type (CDK_pywrapper.fp_type.object): a object that represents a type of fingerprint, which was calculated

    RETURNS:
        smarts (rdkit.mol.object): retrieved smarts pattern from the selected fingerprint and the most present fingerprint bits in a motif

    !!! currently only MACCSFP and KRFP supported! no idea if we can implement PubChemFP and more without losing too much time...
    """
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
    """tanimoto similarity for two given motifs
    
    ARGS:
        fps_motifS (list(np.array)): a list of motif fingerprints

    RETURNS:
        motifs_similarities (list): tanimoto score for every motif combination

    !!! which combination it is coming from is not returned at the moment!
    """
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
    """runs all the scripts to generate a selected fingerprint for a motif

    - smiles2mol: convert smiles to mol objects
    - mols2fps: convert mol objects to selected fingerprint
    - scale_fps: check present of fingerprints bits across motif
    - fps2motif: make the motif fingerprint binary based on given threshold
    - fps2smarts: retrieve SMARTS for found motif fingerprint bits

    - motifs2tanimotoScore: calculated motif similarity based on motif fingerprints using tanimoto similarity


    ARGS:
        smiles_per_motifs: list(list(str)): SMILES for every motif in a different list
        fp_type (CDK_pywrapper.fp_type.object): a object that represents a type of fingerprint that will be calculated
        threshold (float; 0 > x <= 1): number that defines if a bit in the fingerprint with be set to zero (below threshold) or to one (above threshold)

    RETURNS:
        fps_motifs (list(list(np.array))): binary fingerprint for motifs, based on given threshold for including/excluding bits on their presents in a motif
        smarts_per_motifs (list(list(rdkit.mol.object))): mol object for the present bits in fps_motifs (SMARTS pattern)
        motifs_similarities (list): tanimoto score for every motif combination
    """
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

def annotate_motifs_parallel(smiles_per_motif, fp_type=FPType.MACCSFP, threshold=0.8):
    """function suitable for parallelization
    !!! Should currently not be used; I will work on that when we need speed improvements!!!

    ARGS:
        smiles_per_motifs: list(list(str)): SMILES for every motif in a different list
        fp_type (CDK_pywrapper.fp_type.object): a object that represents a type of fingerprint that will be calculated
        threshold (float; 0 > x <= 1): number that defines if a bit in the fingerprint with be set to zero (below threshold) or to one (above threshold)
    
    RETURNS:


    """

    mols_per_motif = smiles2mols(smiles_per_motif)
    fps_per_motif = mols2fps(mols_per_motif, fp_type)
    scaled_fps = scale_fps(fps_per_motif)
    fps_motif = fps2motif(scaled_fps, threshold)
    smarts_per_motif = fps2smarts(fps_motif, fp_type)

    return list(zip(fps_motif, smarts_per_motif))


def run_parallel_annotation(smiles_per_motifs):
    """run parallization
    !!! Should currently not be used; I will work on that when we need speed improvements!!!

    """
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        fps_and_smarts_per_motifs = pool.map(annotate_motifs_parallel, smiles_per_motifs)

    return fps_and_smarts_per_motifs


if __name__ == "__main__":

    smiles_per_motifs = [["O=C(C)Oc1ccccc1C(=O)O", "COC(=O)C1CCC(C1)C(=O)O"]]
    fps_motif = annotate_motifs(smiles_per_motifs, fp_type=FPType.SubFP)
