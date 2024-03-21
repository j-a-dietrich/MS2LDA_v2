import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def extract_motifs(lda_model):
    """extract motifs and associated words from an LDA model.

    Parameters:
    lda_model (gensim.models.LdaModel): Trained LDA model.

    Returns:
    list: List of motifs where each motif is represented as a list of associated words.
    """
    motifs = []
    for motif_idx in range(lda_model.num_topics):
        top_words = lda_model.show_topic(motif_idx)
        words = [word for word, _ in top_words]
        motifs.append(words)
    return motifs



def sep_frags_losses(ext_motifs):
    """Separates fragment / losses in the motifs obtained

    ARGS:
    motifs_list (list): List of motifs where each motif is represented as a list of associated words.

    RETURNS:
    list: List of fragments in the motifs
    list: List of losses in the motifs
    """
    frags = []
    losses = []
    for frags_losses in ext_motifs:
        for word in frags_losses:
            if '+' in word:
                frags.append(word)
            else:
                losses.append(word)
    return frags, losses


def frags_losses_2num(frags_losses_list):
    """Converts a list of frag or losses to a list of floats

    ARGS:
    frag (list): List of fragment or losses (str)

    RETURNS:
    list: List of frag or losses (float)
    """
    frags_losses = []
    for val in frags_losses_list:
        if '+' in val:
            val = val.replace('+', '')  # Remove the '+' character
        frags_losses.append(float(val))
    return frags_losses


def ext_spec(spectra_list):
    """Extracts the peaks and intensities from a list of spectra
    
    ARGS:
    spectra_list: from predict_lda, the spectra associated a motif_select

    RETURNS:
    peaks_extract (list): the peaks of all spectra
    int_extract (list): the intensities of all spectra
    """
    peaks_extract = [np.round(spectra.peaks.mz, 2) for spectra in spectra_list]
    int_extract = [np.round(spectra.peaks.intensities, 2) for spectra in spectra_list]

    return peaks_extract,int_extract


def check_peak_intensity(peaks_extract, int_extract, independent_array):
    """Function to check if elements in an independent array (frag or losses) match peaks and return corresponding intensities.

    ARGS:
    peaks_extract (list): List of peak values.
    int_extract (list): List of corresponding intensities.
    independent_array (list): Array to check for matching elements: fragments or losses.

    RETURNS:
    matched_peaks (list): List of peaks matching elements in the independent array.
    matched_intensities (list): List of intensities corresponding to matched peaks.
    """
    independent_array=list(independent_array)
    matched_peaks = []
    matched_intensities = []
    
    for element in independent_array:
        # Use np.any() to check if any element in peaks_extract matches the current element
        if np.any(peaks_extract == element):
            # Get indices of matching elements
            indices = np.where(peaks_extract == element)[0]
            for idx in indices:
                matched_peaks.append(peaks_extract[idx])
                matched_intensities.append(int_extract[idx])

    return matched_peaks, matched_intensities   


def matching_motifs_spectra(select_motif,ext_peaks, ext_int, frags_losses):
    """Function that exact the matched frag or losses to the spectras predicted from motif selected
    
    ARGS:
    select_motif (list): motif_select from previous use, for example spectra_per_motifs[0]
    ext_peaks (list): list of peaks (floats)
    ext_int (list): list of intensities (floats)
    frags_losses (list): list of fragments or losses (floats)
    
    RETURNS:
    list: list of matched peaks with fragments or losses

    """
    output_frags_losses_spectra=[]
    for i in range(0,len(select_motif)):
        match_peaks,match_int=check_peak_intensity(ext_peaks[i], ext_int[i], frags_losses)
        output_frags_losses_spectra.append((match_peaks, match_int))
    return output_frags_losses_spectra


def plot_spectra_with_motifs(ext_peaks, ext_int, matches_found_frags, matches_found_losses):
    """Plots the spectra per motif along with output motif spectra.

    Parameters:
    spectra_per_motifs: List of lists of Spectrum objects
    output_motif_spectra: List of lists of (x, y) pairs for output motif spectra

    Returns:
    Plots of spectra per motif with corresponding output motif spectra
    """
    for i in range(0,len(matches_found_losses)):
        plt.figure(figsize=(10, 5))
        plt.bar(ext_peaks[i], ext_int[i], width=2, color='gray')  # Plot spectrum peaks
        plt.bar(matches_found_frags[i][0], matches_found_frags[i][1], width=2, color='red')
        plt.bar(matches_found_losses[i][0], matches_found_losses[i][1], width=2, color='blue')
        # Create legend
        spectrum_legend = mpatches.Patch(color='gray', label='Spectrum Peaks')
        motif_frags_legend = mpatches.Patch(color='red', label='Motif_Frags')
        motif_losses_legend = mpatches.Patch(color='blue', label='Motif_Losses')
        plt.legend(handles=[spectrum_legend, motif_frags_legend, motif_losses_legend])
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.xlim(0, 500)  # Change me
        plt.ylim(0, 1) 
        plt.show()
    return None