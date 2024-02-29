from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary

import numpy as np
from itertools import chain

from matchms.importing import load_from_mgf
import matchms.filtering as msfilters

def load_mgf(spectra_path):
    """loads spectra from a mgf file

    ARGS:
        spectra_path (str): path to the spectra.mgf file

    RETURNS:
        spectra (generator): matchms generator object with the loaded spectra
    """

    spectra = load_from_mgf(spectra_path)

    return spectra


def clean_spectra(spectra):
    """uses matchms to normalize intensities, add information and add losses to the spectra
    
    ARGS:
        spectra (generator): generator object of matchms.Spectrum.objects loaded via matchms in python
    
    RETURNS:
        cleaned_spectra (list): list of matchms.Spectrum.objects; spectra that do not fit will be removed
    """
    cleaned_spectra = []

    for spectrum in spectra:
        # metadata filters
        spectrum = msfilters.default_filters(spectrum)
        spectrum = msfilters.add_retention_index(spectrum)
        spectrum = msfilters.add_retention_time(spectrum)
        spectrum = msfilters.require_precursor_mz(spectrum)

        # normalize and filter peaks
        spectrum = msfilters.normalize_intensities(spectrum)
        spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
        spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
        spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
        spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
        spectrum = msfilters.add_losses(spectrum)

        if spectrum:
            cleaned_spectra.append(spectrum)

    return cleaned_spectra



def frag_and_loss2word(spectra): #You should write some unittests for this function; seems to be error prone
    """generates a list of lists for fragments and losses for a dataset

    ARGS:
        spectra (list): list of matchms.Spectrum.objects; they should be cleaned beforehand e.g. intensity normalization, add losses

    RETURNS:
        dataset_frag (list): is a list of lists where each list represents fragements from one spectrum
        dataset_loss (list): is a list of lists where each list represents the losses from one spectrum
    """
    dataset_frag = []
    dataset_loss = []

    for spectrum in spectra:
        intensities_from_0_to_100 = (spectrum.peaks.intensities * 100).round()

        frag_with_2_digits = [ [str(round(mz, 2))+"+"] for mz in spectrum.peaks.mz] # every fragment is in a list
        frag_multiplied_intensities = [frag * int(intensity) for frag, intensity in zip(frag_with_2_digits, intensities_from_0_to_100)]
        frag_flattend = list(chain(*frag_multiplied_intensities))
        dataset_frag.append(frag_flattend)

        loss_with_2_digits = [ [str(round(mz, 2))] for mz in spectrum.losses.mz] # every fragment is in a list
        loss_multiplied_intensities = [loss * int(intensity) for loss, intensity in zip(loss_with_2_digits, intensities_from_0_to_100)]
        loss_flattend = list(chain(*loss_multiplied_intensities))
        loss_without_zeros = list(filter(lambda loss: float(loss) > 0.01, loss_flattend)) # removes 0 or negative loss values
        dataset_loss.append(loss_without_zeros)

    return dataset_frag, dataset_loss



def combine_frag_loss(dataset_frag, dataset_loss):
    """combines fragments and losses for a list of spectra

    ARGS:
        dataset_frag(list): is a list of lists where each list represents fragements from one spectrum
        dataset_loss (list): is a list of lists where each list represents the losses from one spectrum

    RETURNS:
        frag_and_loss (list): is a list of list where each list represents the fragments and losses from one spectrum
    """

    dataset_frag_and_loss = []
    for spectrum_frag, spectrum_loss in zip(dataset_frag, dataset_loss):
        dataset_frag_and_loss.append(spectrum_frag + spectrum_loss)

    return dataset_frag_and_loss


def generate_corpus(dataset_frag_and_loss):
    """generates a corpus (dictionary) for the lda model

    ARGS:
        frag_and_loss (list): is a list of list where each list represents the fragments and losses from one spectrum

    RETURNS:
        corpus4frag_and_loss (list): list of tuple with the count and id of frag or loss
        id2dataset_frag_and_loss (dict): Dictionary with id for fragments and losses
    """

    id2dataset_frag_and_loss = Dictionary(dataset_frag_and_loss)
    
    corpus4dataset_frag_and_loss = []
    for spectrum_frag_and_loss in dataset_frag_and_loss:
        id_count_per_spectrum = id2dataset_frag_and_loss.doc2bow(spectrum_frag_and_loss)
        corpus4dataset_frag_and_loss.append(id_count_per_spectrum)

    return corpus4dataset_frag_and_loss, id2dataset_frag_and_loss


def run_lda(spectra_path, num_topics, iterations=300, update_every=1):

    spectra = load_mgf(spectra_path)
    cleaned_spectra = clean_spectra(spectra)
    dataset_frag, dataset_loss = frag_and_loss2word(cleaned_spectra)
    dataset_frag_and_loss = combine_frag_loss(dataset_frag, dataset_loss)
    corpus4dataset_frag_and_loss, id2dataset_frag_and_loss = generate_corpus(dataset_frag_and_loss)

    lda_model = LdaModel(corpus=corpus4dataset_frag_and_loss,
                     id2word=id2dataset_frag_and_loss,
                     num_topics=num_topics, 
                     random_state=73,
                     update_every=update_every,
                     iterations=iterations) # there are more here!!!
    
    return lda_model, corpus4dataset_frag_and_loss, id2dataset_frag_and_loss

if __name__ == "__main__":
    spectra_path = r"C:\Users\dietr004\Documents\PhD\computational mass spectrometry\Spec2Struc\Project_SubformulaAnnotation\raw_data\_RAWdata1\GNPS-SCIEX-LIBRARY.mgf"
    run_lda(spectra_path, 12)