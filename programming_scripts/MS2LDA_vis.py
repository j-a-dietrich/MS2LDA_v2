import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def extract_topics(lda_model):
    """extract topics and associated words from an LDA model.

    Parameters:
    lda_model (gensim.models.LdaModel): Trained LDA model.

    Returns:
    list: List of topics where each topic is represented as a list of associated words.
    """
    topics = []
    for topic_idx in range(lda_model.num_topics):
        top_words = lda_model.show_topic(topic_idx)
        words = [word for word, _ in top_words]
        topics.append(words)
    return topics


def convert_str_topic(topics):
    """Convert a list of strings to a list of floats, considering only the first two decimal places.

    Parameters:
    str_array (list): List of strings representing floating-point numbers.

    Returns:
    list: List of floats.
    """
    return [float(val[:val.index('.') + 3]) for val in topics]



def extract_peaks(spectra_list):
    """Extracts the intensities from a list of spectra
    
    Parameters:
    spectra_list from predict_lda   

    Returns:
    the intensities of all spectra
    """
    peaks_extract = [np.round(spectra.peaks.mz, 2) for spectra in spectra_list]
    int_extract = [np.round(spectra.peaks.intensities, 2) for spectra in spectra_list]

    return peaks_extract,int_extract


def check_peak_intensity(peaks_extract, int_extract, independent_array):
    """
    Function to check if elements in an independent array match peaks and return corresponding intensities.

    Parameters:
        peaks_extract (list): List of peak values.
        int_extract (list): List of corresponding intensities.
        independent_array (list): Array to check for matching elements.

    Returns:
        matched_peaks (list): List of peaks matching elements in the independent array.
        matched_intensities (list): List of intensities corresponding to matched peaks.
    """
    peak_intensity_tuples = list(zip(peaks_extract, int_extract))  # Tuple peaks and intensities
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



def plot_spectra_with_motif(ext_peaks,ext_int, output_motif_spectra):
    """Plots the spectra per motif along with output motif spectra.

    Parameters:
    spectra_per_motifs: List of lists of Spectrum objects
    output_motif_spectra: List of lists of (x, y) pairs for output motif spectra

    Returns:
    Plots of spectra per motif with corresponding output motif spectra
    """
    for i in range(0,len(output_motif_spectra)):
        plt.figure(figsize=(10, 5))
        plt.bar(ext_peaks[i], ext_int[i], width=2, color='gray')  # Plot spectrum peaks
        plt.bar(output_motif_spectra[i][0], output_motif_spectra[i][1], width=2, color='red')
         # Create legend
        spectrum_legend = mpatches.Patch(color='gray', label='Spectrum Peaks')
        motif_legend = mpatches.Patch(color='red', label='Motif')
        plt.legend(handles=[spectrum_legend, motif_legend])
        plt.xlabel('m/z')
        plt.ylabel('Intensity')
        plt.xlim(0, 500)  # Example x limits
        plt.ylim(0, 1) 
        plt.show()
    return None