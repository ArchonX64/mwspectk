from __future__ import annotations

import subprocess as sub

from typing import Union
from time import time

import numpy as np
import matplotlib as mpl
import pandas as pd

import scipy.signal as sp_sig
import scipy.interpolate as sp_int

import matplotlib.pyplot as plt
from numpy import dtype

# Remove comment to enable PySide backend
mpl.use("QtAgg")

fig = plt.figure()
ax = fig.add_subplot()


class Spectrum:
    """
        The abstract object of a spectrum. Only guarantees a name, a method that returns an array of frequencies, and
        a method that returns an array of intensities. The two methods should return concurrent arrays.
        :ivar name: A string containing a simple name for the spectrum
    """

    def __init__(self, name: str):
        self.name = name

    def get_frequencies(self) -> np.ndarray:
        """
        An abstract method for obtaining frequencies of a spectrum. Must be implemented manually
        """
        raise TypeError("The method get_frequencies needs to be overridden in class" + self.__name__)

    def get_intensities(self) -> np.ndarray:
        """
                An abstract method for obtaining intensities of a spectrum. Must be implemented manually
                """
        raise TypeError("The method get_intensities needs to be overridden in class" + self.__name__)


class Ratio:
    """
        An object containing ratios of a spectrum that correspond to an index in a peak list. The internal DataFrame
        contains three columns:
            p_inds: The indexes of the corresponding parent spectrum\n
            a_inds: The indexes of the corresponding spectrum parent was divided against\n
            ratios: The value of the calculated ratio
        :ivar data: A DataFrame containing the quantitative data
        :ivar parent: The spectrum that a ratio was generated for (i.e. divide_by was called on it)
        :ivar against: The spectrum that parent was divided against
        :ivar freq_var: The frequency variability that was used in creating the ratios


    """

    def __init__(self, parent: ExperimentalSpectrum, against: ExperimentalSpectrum, freq_var: float,
                 ratios: np.ndarray = None, parent_inds: np.ndarray = None, against_inds: np.ndarray = None,
                 df: pd.DataFrame = None):
        self.data = pd.DataFrame({
            "p_inds": parent_inds,
            "a_inds": against_inds,
            "ratios": ratios
        }) if df is None else df
        self.parent: ExperimentalSpectrum = parent
        self.against: ExperimentalSpectrum = against
        self.freq_var: float = freq_var

    def copy(self, new_owner: ExperimentalSpectrum):
        """
        Generates a new ratio object with the data copied. Usually called when a spectrum is being copied, so
        the parent may change.
        :param new_owner: The parent spectrum that this object will have
        :return: A copy of this ratio object
        """
        return Ratio(parent=new_owner, against=self.against, df=self.data.copy(True), freq_var=self.freq_var)


class ExperimentalSpectrum(Spectrum):
    """
    Represents a spectrum that was recorded experimentally. Will have a large amounts of values that may or may not be
    of value, and must be sorted.
    :ivar dataframe: A DataFrame containing the frequencies ("freq") and intensities (the name of the spectrum) of the
    spectrum
    :ivar freqs: Easy access of "freq" column of dataframe
    :ivar inten: Easy access of "inten" column of dataframe
    :ivar baseline: The minimum value that should be considered when looking at this spectrum. Anything below this value
    is considered noise
    :ivar peaks: An object containing the calculated peaks of this spectrum
    :ivar ratios: DEPRECATED
    """

    # Do not call without filling in peaks!
    def __init__(self, name: str, freq: np.ndarray, inten: np.ndarray, baseline: float = 0, ratios: list[Ratio] = None):
        super().__init__(name)

        self.dataframe = pd.DataFrame({"freq": freq, name: inten})
        self.freqs: pd.Series = self.dataframe["freq"]
        self.inten: pd.Series = self.dataframe[name]
        self.baseline: float = baseline
        self.peaks: Union[DeterminedPeaks, None] = None  # Will ONLY be None until get_spectrum produces peaks
        self.ratios: list[Ratio] = [] if ratios is None else ratios

    def __add__(self, other) -> SpectrumCollection:
        if isinstance(other, ExperimentalSpectrum):
            return SpectrumCollection([self, other])

    def plot_baseline(self) -> None:
        """
        Plot a horizontal line to v
        """
        plt.axhline(y=self.baseline)

    # Peaks is the INDEX in peak_inds
    def _remove_peaks(self, peaks: np.ndarray):
        # Set the width of each peak equal to the baseline, creating horizontal lines where the peak was.
        for peak in peaks:
            #self.inten[self.peaks.left_bases[peak]:self.peaks.right_bases[peak]] = self.baseline
            self.inten[self.peaks.left_bases[peak]:self.peaks.right_bases[peak]] = 0

    def divide_by(self, other: ExperimentalSpectrum, freq_variability) -> Ratio:
        """
            Generate ratios of the intensities of one spectrum divided by another\n
            BOTH SPECTRA MUST HAVE THE SAME FREQUENCY AT EACH INDEX
            :param other: The spectrum to be divided against
            :param freq_variability: The maximum frequency (in MHz) that peaks can be separated by for matching
            :return: An object containing the calculated ratios based on peak indexes, along with other information
            """
        ratio_object = self._divide_by(other, freq_variability)
        self.peaks.ratios.append(ratio_object)
        return ratio_object

    def _divide_by(self, other: ExperimentalSpectrum, freq_variability) -> Ratio:
        # Check if a ratio calculation already exists
        #for ratio in self.peaks.ratios:
        #if ratio.against == other:
        #return ratio

        # If ratio has not yet been calculated,
        # Figure out which peaks are matching in each spectrum
        self_inds, other_inds = self.peaks.same_peaks_as(other, freq_variability)

        # Divide each matching point by each other to generate a ratio for each
        ratios = self.peaks.get_peak_inten()[self_inds] / other.peaks.get_peak_inten()[other_inds]

        ratio_object = Ratio(parent=self, against=other, ratios=ratios, parent_inds=self_inds,
                             against_inds=other_inds, freq_var=freq_variability)
        return ratio_object

    def keep_ratio_of(self, ratio: Ratio, lbound: float = 0, ubound: float = np.inf):
        ratio_axis = "p_inds" if self == ratio.parent else "a_inds"

        # Remove all values that do not have a ratio
        non_ratio = np.arange(len(self.peaks.peak_inds))
        non_ratio = np.delete(non_ratio, ratio.data[ratio_axis])
        #self.remove_peaks(non_ratio)
        #self.peaks.remove_indexes(self.peaks.peak_inds[non_ratio])

        ratio = self.divide_by(ratio.against, ratio.freq_var)
        inds = ratio.data[(ratio.data["ratios"] > ubound) | (ratio.data["ratios"] < lbound)][ratio_axis].to_numpy()
        #self.remove_peaks(np.asarray(inds, dtype="int"))
        #self.peaks.remove_indexes(np.asarray(inds, dtype="int"))

        full = np.append(non_ratio, inds)
        self._remove_peaks(full)
        self.peaks.remove_indexes(full)

    def plot(self, name: str = None) -> None:
        plt.plot(self.freqs, self.inten, label=self.name if name is None else name)
        plt.axhline(self.baseline, ls="--")

    def export(self, out_type: str, name: str = None) -> None:
        name = name if name is not None else self.name + "_data"

        match out_type:
            case "csv":
                self.dataframe.to_csv(name + ".csv", header=False, index=False)
            case "txt":
                self.dataframe.to_csv(name + ".txt", sep=" ", header=False, index=False)
            case "ft":
                self.dataframe.to_csv(name + ".ft", sep=" ", header=False, index=False)
            case _:
                raise ValueError("Unsupported export type for spectrum: " + self.name)

    def get_frequencies(self) -> np.ndarray:
        return self.dataframe["freq"]

    def get_intensities(self) -> np.ndarray:
        return self.dataframe[self.name]

    def get_cdp(self, freq_variability: float, max_doub_sep: float, max_cdp_sep: float, inten_rat_var: float):
        # Save peak information to files
        np.savetxt("freq", self.peaks.get_peak_freq(), delimiter="\n")
        np.savetxt("inten", self.peaks.get_peak_inten(), delimiter="\n")

        # Run external CDP finder
        try:
            sub.check_call(["./CDPfinder.exe", "freq", "inten", f"{len(self.peaks.get_peak_freq())}",
                            f"{freq_variability}", f"{max_doub_sep}", f"{max_cdp_sep}", f"{inten_rat_var}"], stdout=sub.PIPE)
        except sub.CalledProcessError as e:
            e.add_note("There was an error running the CDP executable!")
            raise e

        # Create a inverse mask of CDP peaks
        cdps = np.unique(np.loadtxt("cdps", dtype=int))
        not_cdps = np.delete(np.arange(len(self.peaks.peak_inds)), cdps)

        # Remove non-CDP lines from spectrum
        self._remove_peaks(not_cdps)
        self.peaks.remove_indexes(not_cdps)

    def copy(self) -> ExperimentalSpectrum:
        exp = ExperimentalSpectrum(
            name=self.name,
            freq=self.get_frequencies().copy(),
            inten=self.get_intensities().copy(),
            baseline=self.baseline,
        )
        exp.peaks = self.peaks.copy(self)
        return exp


class SpectrumPeaks:
    peak_color_index = 1

    def __init__(self, name):
        self.name = name

    def plot(self, indexes: np.ndarray = None, name: str = None, scatter=False):
        SpectrumPeaks.peak_color_index += 1
        name = self.name if name is None else name

        freq_to_plot = self.get_peak_freq() if indexes is None else self.get_peak_freq()[indexes]
        ints_to_plot = self.get_peak_inten() if indexes is None else self.get_peak_inten()[indexes]
        if scatter:
            plt.scatter(freq_to_plot, ints_to_plot, label=name,
                        c="C" + str(SpectrumPeaks.peak_color_index))
        else:
            plt.stem(freq_to_plot, ints_to_plot, label=name, markerfmt=" ",
                     linefmt="C" + str(SpectrumPeaks.peak_color_index))
        SpectrumPeaks.peak_color_index += 1

    def get_peak_freq(self) -> np.ndarray:
        raise TypeError("The method get_peak_freq needs to be overridden in class" + self.__name__)

    def get_peak_inten(self) -> np.ndarray:
        raise TypeError("The method get_peak_inten needs to be overridden in class" + self.__name__)

    def remove_indexes(self, inds: np.ndarray) -> None:
        raise TypeError("The method remvove_indexes needs to be overridden in class" + self.__name__)

    def same_peaks_as(self, other: Union[SpectrumPeaks, ExperimentalSpectrum, set[SpectrumPeaks]],
                      freq_variability: float, inten_variance: float = None, unique: bool = False) -> (
            np.ndarray, np.ndarray):
        if isinstance(other, SpectrumPeaks):
            self_inds, other_inds = self._same_peaks_as(other, freq_variability, unique)
            if inten_variance is not None:
                self_inds, other_inds = self._same_peak_inten_as(self, other, self_inds, other_inds, inten_variance)
            return self_inds, other_inds
        elif isinstance(other, ExperimentalSpectrum):
            self_inds, other_inds = self._same_peaks_as(other.peaks, freq_variability, unique)
            if inten_variance is not None:
                self_inds, other_inds = self._same_peak_inten_as(self, other.peaks, self_inds, other_inds,
                                                                 inten_variance)
            return self_inds, other_inds
        elif isinstance(other, set):
            inds = []
            for each in other:
                self_inds, other_inds = self._same_peaks_as(each, freq_variability, unique)
                if inten_variance is not None:
                    self_inds, other_inds = self._same_peak_inten_as(self, other, self_inds, other_inds)
                inds.append((self_inds, other_inds))
            return inds

    # Returns: Indexes of peak_inds
    def _same_peaks_as(self, other: SpectrumPeaks, freq_variability: float, unique: bool) -> (np.ndarray, np.ndarray):
        self_freqs = self.get_peak_freq()
        other_freqs = other.get_peak_freq()

        # Indexes now contains locations closest to other peaks
        # Each index i represents an index of other_freqs, and indexes[i] represents index of self_freqs
        indexes = np.searchsorted(self_freqs, other_freqs)

        # Iteratively check each peak of other against the closest frequencies in self to determine whether
        #   they are the "same" (are within freq_variability), and add them to a list
        # np.searchsorted give the indexes of where each peak in other would land within self's peaks, so the peaks
        #   that are before and after need be checked if they are within freq_variability. Additionally, if two peaks
        #   are within the allowed variance, the peak with the least variance is used.
        self_inds = np.ndarray(shape=len(indexes))
        other_inds = np.ndarray(shape=len(indexes))
        freq_len = len(self_freqs)
        counter = 0

        for i in range(0, len(indexes)):
            # Prevent 0 or max indexing, as values that are too low/high will be placed there
            if indexes[i] == 0 or indexes[i] >= freq_len:
                continue
            remove_next = None
            has_after = False

            after_variance = self_freqs[indexes[i]] - other_freqs[i]
            on_variance = other_freqs[i] - self_freqs[indexes[i] - 1]

            if after_variance < freq_variability:
                remove_next = indexes[i]
                has_after = True
            if on_variance < freq_variability:
                if not has_after:
                    remove_next = indexes[i] - 1
                elif on_variance < after_variance:
                    remove_next = indexes[i]

            if remove_next is not None:
                self_inds[counter] = remove_next
                other_inds[counter] = i
                counter += 1

        # Shave off unused space
        self_inds = self_inds[:counter - 1]
        other_inds = other_inds[:counter - 1]

        if unique:
            other_inds, indices = np.unique(other_inds, return_index=True)
            self_inds = self_inds[indices]

        # Convert back to numpy array
        return np.asarray(self_inds, dtype="int"), np.asarray(other_inds, dtype="int")

    def _same_peak_inten_as(self, other: SpectrumPeaks, self_inds: np.ndarray, other_inds: np.ndarray,
                            inten_var: float) -> (np.ndarray, np.ndarray):
        self_inds_out = np.ndarray(shape=len(self_inds))
        other_inds_out = np.ndarray(shape=len(other_inds))

        if inten_var <= 0:
            raise ValueError("The intensity_variance parameter in same_peaks_as must be greater than 0!")

        arr_counter = 0
        for i in range(0, len(self_inds)):
            if abs(1 - self.get_peak_inten()[self_inds[i]] / other.get_peak_inten()[other_inds[i]]) < inten_var:
                self_inds_out[arr_counter] = self_inds[i]
                other_inds_out[arr_counter] = other_inds[i]
                arr_counter += 1

        self_inds_out = self_inds_out[:arr_counter - 1]
        other_inds_out = self_inds_out[:arr_counter - 1]

        return self_inds_out, other_inds_out

    def remove_peaks_of(self, other: Union[SpectrumPeaks, ExperimentalSpectrum, set[SpectrumPeaks]],
                        freq_variability: float, inten_variability: float = None) -> (np.ndarray, np.ndarray):
        if isinstance(other, SpectrumPeaks):
            return self._remove_peaks_of(other, freq_variability, inten_variability)

        elif isinstance(other, ExperimentalSpectrum):
            return self._remove_peaks_of(other.peaks, freq_variability, inten_variability)

        elif isinstance(other, set):
            inds = []
            for each in other:
                inds.append(self._remove_peaks_of(each, freq_variability, inten_variability))
            return inds

    def _remove_peaks_of(self, other: SpectrumPeaks, freq_variability: float, inten_variability: float) -> (
            np.ndarray, np.ndarray):
        self_inds, other_inds = self.same_peaks_as(other, freq_variability, inten_variability)
        self.remove_indexes(self_inds)
        return self_inds, other_inds


class DeterminedPeaks(SpectrumPeaks):
    def __init__(self, spectrum: ExperimentalSpectrum, inten_thresh: float = None, prominence: float = None,
                 wlen: float = None, peaks: np.ndarray = None, left_bases: np.ndarray = None,
                 right_bases: np.ndarray = None):
        super().__init__(name=spectrum.name + " (peaks)")

        if prominence is not None:
            (peaks, properties) = sp_sig.find_peaks(spectrum.inten, height=inten_thresh, prominence=prominence)
            (prominences, left_bases, right_bases) = sp_sig.peak_prominences(spectrum.inten, peaks, wlen=wlen)

        self.spectrum = spectrum
        self.peak_inds = peaks
        self.left_bases = left_bases
        self.right_bases = right_bases
        self.ratios: list[Ratio] = []

    def remove_indexes(self, inds: np.ndarray):
        self.peak_inds = np.delete(self.peak_inds, inds)
        self.left_bases = np.delete(self.left_bases, inds)
        self.right_bases = np.delete(self.right_bases, inds)

        # Ratios depend on indexes of peaks, so they must be recalculated
        old_rats = self.ratios.copy()  # Save old ratios so that the spectrum they were calculated against is known
        self.ratios = []  # Clear old ratios
        for ratio in old_rats:
            self.spectrum.divide_by(ratio.against, ratio.freq_var)

    def get_peak_freq(self) -> np.ndarray:
        return self.spectrum.freqs[self.peak_inds].to_numpy()

    def get_peak_inten(self) -> np.ndarray:
        return self.spectrum.inten[self.peak_inds].to_numpy()

    def _remove_peaks_of(self, other, freq_variability, inten_variability):
        self_inds, other_inds = self.same_peaks_as(other, freq_variability, inten_variability)
        self.spectrum._remove_peaks(self_inds)
        self.remove_indexes(self_inds)
        return self_inds, other_inds

    def plot(self, scatter=False, sides=False, indexes: np.ndarray = None, name: str = None):
        super().plot(indexes, name=name, scatter=scatter)
        if sides:
            if indexes is None:
                plt.scatter(self.spectrum.get_frequencies()[self.left_bases],
                            self.spectrum.get_intensities()[self.left_bases],
                            c="C" + str(SpectrumPeaks.peak_color_index))
                plt.scatter(self.spectrum.get_frequencies()[self.right_bases],
                            self.spectrum.get_intensities()[self.right_bases],
                            c="C" + str(SpectrumPeaks.peak_color_index))
            else:
                plt.scatter(self.spectrum.get_frequencies()[self.left_bases[indexes]],
                            self.spectrum.get_intensities()[self.left_bases],
                            c="C" + str(SpectrumPeaks.peak_color_index))
                plt.scatter(self.spectrum.get_frequencies()[self.right_bases[indexes]],
                            self.spectrum.get_intensities()[self.right_bases],
                            c="C" + str(SpectrumPeaks.peak_color_index))
            SpectrumPeaks.peak_color_index += 1

    def copy(self, new_parent: ExperimentalSpectrum) -> DeterminedPeaks:
        det = DeterminedPeaks(
            spectrum=new_parent,
            peaks=self.peak_inds.copy(),
            left_bases=self.left_bases.copy(),
            right_bases=self.right_bases.copy(),
        )
        det.ratios = [ratio.copy(new_parent) for ratio in self.ratios]
        return det

    def export(self, ext: str, name: str = None):
        name = name if name is not None else self.name + "_data"

        df = pd.DataFrame({
            "Frequency (MHz)": self.get_peak_freq(),
            "Intensity (V)": self.get_peak_inten(),
        })

        for ratio in self.ratios:
            ratio_series = np.ndarray(shape=len(self.get_peak_inten()))
            ratio_series[:] = np.nan
            ratio_series[ratio.data["p_inds"]] = ratio.data["ratios"]
            df = pd.concat([df, pd.DataFrame({ratio.parent.name + " / " + ratio.against.name: ratio_series})], axis=1)

        match ext:
            case "csv":
                df.to_csv(name + ".csv", index=False)
            case "txt":
                df.to_csv(name + ".txt", sep=" ", header=False, index=False)
            case "ft":
                df.to_csv(name + ".ft", sep=" ", header=False, index=False)
            case _:
                raise ValueError("Unsupported export type for spectrum: " + self.name)


class ClusterSpectrum(SpectrumPeaks):
    min_freq = 0
    max_freq = None
    qnums = ["uJ", "uKa", "uKc", "lJ", "uKa", "uKc"]
    color_index = 0

    def __init__(self, name: str, dataframe: pd.DataFrame, freq: str, rel_inten: str):
        super().__init__(name=name)
        self.dataframe = dataframe
        self.freq = self.dataframe[freq]
        self.rel_inten = self.dataframe[rel_inten]

    def plot(self):
        plt.stem(self.freq, self.rel_inten, label=self.name, markerfmt=" ",
                 linefmt="C" + str(ClusterSpectrum.color_index))
        ClusterSpectrum.color_index += 1

    def get_peak_freq(self) -> np.ndarray:
        return self.freq

    def get_peak_inten(self) -> np.ndarray:
        return self.rel_inten

    def remove_indexes(self, inds):
        self.dataframe.drop(index=inds, inplace=True)


class SpectrumCollection:
    class Ratio:
        def __init__(self, against: ExperimentalSpectrum, held: list[ExperimentalSpectrum]):
            self.against: ExperimentalSpectrum = against
            self.held: list[ExperimentalSpectrum] = held
            self.dataframe = pd.DataFrame()

    def __init__(self, spectrum_list: list[ExperimentalSpectrum]):
        if len(spectrum_list) == 0:
            raise ValueError("SpectrumCollection list argument must contain at least one value!")

        self.spectrum_list = spectrum_list.copy()
        self.ratios: list[SpectrumCollection.Ratio] = []

    def __add__(self, other) -> Union[ExperimentalSpectrum, SpectrumCollection]:
        if isinstance(other, ExperimentalSpectrum):
            return SpectrumCollection(self.spectrum_list + [other])
        elif isinstance(other, SpectrumCollection):
            return SpectrumCollection(self.spectrum_list + other.spectrum_list)

    def remove_peaks_of(self, others: set[SpectrumPeaks], freq_variability: float):
        for spectrum in self.spectrum_list:
            spectrum.remove_peaks_of(others, freq_variability)

    def create_ratios(self, against: ExperimentalSpectrum):
        pass

    def plot(self) -> None:
        for spectrum in self.spectrum_list:
            spectrum.plot()


def on_click(event):
    if event is None:
        return  # None will be passed if an out-of-bounds area is clicked.

    for text in fig.texts:
        text.remove()

    fig.text(0.5, 1.10, "Click At: {x:.4f}, {y:.4f}".format(x=event.xdata, y=event.ydata), verticalalignment="top",
             horizontalalignment="center", transform=ax.transAxes, fontsize=10)


def show(interactive: bool = False):
    if interactive:
        plt.connect("button_press_event", on_click)

    if ax.lines or ax.collections:
        plt.legend(loc=2)
        plt.show()


def get_spectrum(path: str, name: str, inten_thresh: float, prominence: float,
                 wlen: int = None, resolution_mult: float = None) -> ExperimentalSpectrum:
    if path.split(".")[1] == "csv":
        #data = np.loadtxt(path, delimiter=",")w
        data = np.genfromtxt(path, delimiter=",")

    else:
        data = np.loadtxt(path)

    wlen = 20 if wlen is None else wlen

    data = data[data[:, 0].argsort()]

    freq = data[:, 0]
    inten = data[:, 1]

    if resolution_mult is not None:
        if resolution_mult <= 1:
            raise ValueError("Resolution multiplier cannot be 1 or below for spectrum: " + name)

        cs = sp_int.CubicSpline(freq, inten)
        freq = np.arange(freq.min(), freq.max(), (freq.max() - freq.min()) / (freq.shape[0] * resolution_mult))
        inten = cs(freq)

    spectrum = ExperimentalSpectrum(name=name, freq=freq, inten=inten)
    spectrum.peaks = DeterminedPeaks(spectrum, inten_thresh, prominence, wlen)
    return spectrum


def get_cat(path: str, name: str, min_freq: float = ClusterSpectrum.min_freq,
            max_freq: float = ClusterSpectrum.max_freq) -> ClusterSpectrum:
    if path.split(sep=".")[-1] != "cat":
        raise ValueError("Fitted spectra must be provided using cat file!")

    # Columns are not homogenous sos columns must be used
    try:
        # print(pd.read_csv(path, header=None))
        data = pd.read_csv(path, usecols=(0, 2), header=None, sep='\\s+')
    except ValueError as e:
        e.add_note("File with name " + name + " had an error while reading columns. This may be due to SPCAT\n"
                                              "overfilling the error line into the previous column. If this is\n"
                                              "the case, rerun SPCAT with the experimental error turned down.")
        raise e

    # data = np.loadtxt(path, usecols=(0, 2))
    data.columns = ["Frequency", "Intensity"]

    # Keep only needed values
    if max_freq is not None:
        data = data[(data.loc[:, "Frequency"] > min_freq) & (data.loc[:, "Frequency"] < max_freq)]
    else:
        data = data[(data.loc[:, "Frequency"] > min_freq)]

    # Intensity is stored as base 10 log
    data["Intensity"] = np.power(10, data["Intensity"])
    return ClusterSpectrum(name, data, "Frequency", "Intensity")


def get_lin(path: str, name: str, min_freq: float = ClusterSpectrum.min_freq,
            max_freq: float = ClusterSpectrum.max_freq) -> ClusterSpectrum:
    if path.split(sep=".")[-1] != "lin":
        raise ValueError("Fitted spectra must be provided using .lin file!")

    # Columns are not homogenous so usecols must be used
    data = pd.read_csv(path, usecols=[12], sep='\\s+', header=None)
    data.insert(1, "Intensity", np.zeros(len(data)))

    # data = np.loadtxt(path, usecols=(0, 2))
    data.columns = ["Frequency", "Intensity"]

    # Keep only needed values
    if max_freq is not None:
        data = data[(data.loc[:, "Frequency"] > min_freq) & (data.loc[:, "Frequency"] < max_freq)]
    else:
        data = data[(data.loc[:, "Frequency"] > min_freq)]
    return ClusterSpectrum(name, data, "Frequency", "Intensity")


def plot_RVI(spectrum: ExperimentalSpectrum, ratio: Ratio, label: str = None, color: str = None):
    if color is None:
        plt.scatter(spectrum.peaks.get_peak_inten()[ratio.data["a_inds"]],
                    ratio.data["ratios"], label=label)
    else:
        plt.scatter(spectrum.peaks.get_peak_inten()[ratio.data["a_inds"]],
                    ratio.data["ratios"], label=label, color=color)


def construct_RVI(parent_spec: ExperimentalSpectrum, div_spec: ExperimentalSpectrum, cut: list[SpectrumPeaks],
                  freq_variability: float, plot_all: bool = True, x_label: str = None, y_label: str = None):
    ratio = div_spec.divide_by(other=parent_spec, freq_variability=freq_variability)

    if plot_all:
        plot_RVI(parent_spec, ratio)

    # ADDITIONAL
    for cluster in cut:
        parent_spec.peaks.remove_peaks_of(other={cluster}, freq_variability=freq_variability)

        ratio = div_spec.divide_by(other=parent_spec, freq_variability=freq_variability)

        if plot_all:
            plot_RVI(parent_spec, ratio)

    plot_RVI(parent_spec, ratio, color="black")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if not plot_all:
        plot_RVI(parent_spec, ratio)
