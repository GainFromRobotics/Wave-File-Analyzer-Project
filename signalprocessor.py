from enum import Enum
from typing import Any

import numpy as np
import scipy as sp
import librosa
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from collections.abc import Callable

# A very special thanks to the following sources
# https://ismir2005.ismir.net/proceedings/1019.pdf
# https://pythonguides.com/scipy-find-peaks/
# https://www.qeios.com/read/UFBEFX

class SignalProcessor:
    """ PURE COMPUTATION!!!"""

    def __init__(self, sample_rate: int, TET: int = 12):
        self.sample_rate = sample_rate
        self.tet = TET

    @staticmethod
    def normalize_pcm(samples: np.ndarray, sample_width: int = 2) -> np.ndarray:
        """Convert raw PCM bytes to normalized float [-1, 1]"""
        max_val = 2 ** (8 * sample_width - 1)
        return samples.astype(np.float32) / max_val
    
    @staticmethod
    def to_decibels(x: np.ndarray, ref: float = 1.0, floor: float = -120) -> np.ndarray:
        """Convert magnitude to dB"""
        db = 20 * np.log10(np.maximum(x, 1e-10) / ref)
        return np.maximum(db, floor)

    # Helper function: compute the cosine similarity of two signals
    @staticmethod
    def compute_cosine_similarity(x: np.ndarray, y: np.ndarray):
        
        x = np.asarray(x, dtype=np.float32).ravel()
        y = np.asarray(y, dtype=np.float32).ravel()

        if x.size != y.size:
            raise ValueError("x and y must be of same length")

        denom = np.linalg.norm(x)*np.linalg.norm(y)
        if denom == 0:
            return 0.0

        return float(np.dot(x,y)/denom)

    @staticmethod
    def parabolic_interp(mag, idx):
        if idx <= 0 or idx >= len(mag)-1:
            return float(idx)

        alpha = mag[idx-1]
        beta = mag[idx]
        gamma = mag[idx+1]
        denom = (alpha - 2*beta + gamma)
        if denom == 0:
            return float(idx)
        p = 0.5 * (alpha - gamma) / denom
        return float(idx)+p

    @staticmethod
    def get_note_and_difference(freq: float, A4: float = 440):
        if freq <= 0:
            raise ValueError("Frequency must be positive")

        BASE_OCTAVE = 4
        TET = 12

        TET_12_note_names = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']
        
        semitone_offset = TET*np.log2(freq/A4)+9
        nearest_semitone = int(np.round(semitone_offset))

        note_idx = nearest_semitone % TET
        note_octave = BASE_OCTAVE + (nearest_semitone // TET)

        note_name = f"{TET_12_note_names[note_idx]}{note_octave}"
        cents_off = (semitone_offset - nearest_semitone)

        if cents_off <= 0.001:
            cents_off = 0

        return note_name, int(cents_off*100)

    @staticmethod
    def freq_to_semitone(f: float):
        A4 = 440
        TET = 12
        return TET*np.log2(f/A4)

    @staticmethod
    def semitone_to_freq(n: int | np.int32):
        A4 = 440
        TET = 12
        return A4*np.power(2,n/TET)

    """
    compute_snr:
    > computes the signal to noise ratio of a given signal
    """
    def snr(self,signal: np.ndarray, window_size: int = 1024) -> float:

        if len(signal) < window_size:
            signal = np.pad(signal,(0,window_size-len(signal)))

        m = signal[:window_size].mean()
        sd = signal[:window_size].std()
        if sd == 0:
            return 0
        else:
            return m/sd
    
    def full_snr(self, full_signal:np.ndarray,window_size: int = 1024, hop_size: int = 512) -> np.ndarray:
        if len(full_signal) < window_size:
            full_signal = np.pad(full_signal,(0,window_size - len(full_signal)))

        compute = []
        for i in range(0,len(full_signal) - window_size,hop_size):
            frame = full_signal[i:i+window_size]
            compute_frame = self.snr(frame,window_size)
            compute.append(compute_frame)

        return np.array(compute)

    """
    compute_spectrum:
    > get the spectrum of a signal
    > computes an fft on the provided signal
    > returns spectrum magnitude and frequency bins as a tuple
    """

    class FFT_RETURN_TYPE(Enum):
        COMPLEX = 1,
        LINEAR = 2,
        DECIBELS = 3,

    def spectrum(
            self, 
            signal: np.ndarray, 
            window_size: int = 4096,
            zero_pad_factor: int = 2, # Zero padding multiplier for higher FFT bin density
            smoothing_sigma: float = 0.0,
            return_type: FFT_RETURN_TYPE = FFT_RETURN_TYPE.LINEAR
        ):
        if len(signal) < window_size:
            signal = np.pad(signal,(0,window_size-len(signal)))

        # perform windowing and a larger FFT (zero-padding) for better frequency sampling
        frame = signal[:window_size].astype(np.float32)
        window = np.hanning(window_size).astype(np.float32)
        frame_win = frame * window
    
        # choose an NFFT that's a power of two and larger than window_size
        nfft = int(2 ** np.ceil(np.log2(window_size * max(1, zero_pad_factor))))
        frequency_bins = np.fft.rfftfreq(nfft, d=1.0/self.sample_rate)
        spectrum_vals = np.fft.rfft(frame_win, n=nfft)

        # process data depending on return type
        match return_type:
            case self.FFT_RETURN_TYPE.COMPLEX:
                return spectrum_vals, frequency_bins
                pass
            case self.FFT_RETURN_TYPE.LINEAR:
                spectrum_mag = np.abs(spectrum_vals)
                #return spectrum_mag, frequency_bins
            case self.FFT_RETURN_TYPE.DECIBELS:
                spectrum_mag = self.to_decibels(np.abs(spectrum_vals))
            
        # light smoothing — keep small to avoid washing out narrow peaks
        if smoothing_sigma > 0:
            spectrum_mag = gaussian_filter1d(spectrum_mag, sigma=smoothing_sigma)
    
        return spectrum_mag,frequency_bins

    def spectrogram(
            self, 
            full_signal:np.ndarray, 
            window_size: int = 1024,
            hop_size: int = 512,
            zero_pad_factor: int = 2, # Zero padding multiplier for higher FFT bin density
            smoothing_sigma: float = 0.0,
            return_type: FFT_RETURN_TYPE = FFT_RETURN_TYPE.COMPLEX
        ):
        if len(full_signal) < window_size:
            full_signal = np.pad(full_signal,(0,window_size - len(full_signal)))

        compute = []
        for i in range(0,len(full_signal) - window_size,hop_size):
            frame = full_signal[i:i+window_size]

            # compute spectrum of this frame
            compute_mag,_ = self.spectrum(
                signal=frame,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                smoothing_sigma=smoothing_sigma,
                return_type=return_type
            )
            
            compute.append(compute_mag)

        return np.array(compute).T        

    """
    compute_cepstrum
    > compute the cepstrum of a signal
    """
    def cepstrum(
        self,
        signal: np.ndarray,
        window_size: int = 4096,
        zero_pad_factor: int = 4, # Zero padding multiplier for higher FFT bin density
        spectrum_values_linear: np.ndarray | None = None, # alternatively, pass in our spectrum
        frequency_bins: np.ndarray | None = None, # this way we don't have to redo the fft computation
        return_power_cepstrum: bool = False
    ):

        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        # If we're given the spectrum, skip the whole fft step
        if spectrum_values_linear is not None and frequency_bins is not None:
            spectrum_magnitude = spectrum_values_linear
            spectrum_frequency = frequency_bins
        else:
            if len(signal) < window_size:
                signal = np.pad(signal,(0,window_size-len(signal)))

            spectrum_magnitude,spectrum_frequency = self.spectrum(
                signal=signal,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                smoothing_sigma=0.5,
                return_type=self.FFT_RETURN_TYPE.LINEAR
            )
            spectrum_magnitude = np.abs(spectrum_magnitude)


        if spectrum_magnitude.size == 0:
            return np.array([], dtype=np.float32), np.array([],dtype=np.float32)

        if return_power_cepstrum:
            spectrum_magnitude = spectrum_magnitude ** 2

        log_mag = np.log(np.maximum(np.abs(spectrum_magnitude), 1e-12))
        cepstrum = np.abs(np.fft.irfft(log_mag))

        if return_power_cepstrum:
            cepstrum = cepstrum ** 2

        quefrency_bins = np.arange(cepstrum.size, dtype=np.float32) / self.sample_rate
        return cepstrum, quefrency_bins

    def cepstrogram(
        self, 
        full_signal:np.ndarray, 
        window_size: int = 1024,
        hop_size: int = 512,
        zero_pad_factor: int = 2, # Zero padding multiplier for higher FFT bin density
        smoothing_sigma: float = 0.0,
        return_type: FFT_RETURN_TYPE = FFT_RETURN_TYPE.COMPLEX
    ):
        if len(full_signal) < window_size:
            full_signal = np.pad(full_signal,(0,window_size - len(full_signal)))

        compute = []
        for i in range(0,len(full_signal) - window_size,hop_size):
            frame = full_signal[i:i+window_size]

            # compute spectrum of this frame
            compute_mag,_ = self.spectrum(
                signal=frame,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                smoothing_sigma=smoothing_sigma,
                return_type=return_type
            )
            
            compute.append(compute_mag)

        return np.array(compute).T

    """
    compute_CQT
    > get the Constant-Q Transform of a signal
    > This is a spectrum with logarithmic spacing instead of linear spacing
    > This gets us the frequencies of notes instead of individual frequencies
    """

    semitones_A0 = -4*12
    semitones_A8 = 4*12

    # A wrapper around librosas CQT function
    def CQT(
        self,
        signal: np.ndarray,
        window_size: int = 4096,
        zero_pad_factor: int = 4, # Zero padding multiplier for higher FFT bin density
        smoothing_sigma: int = 1,
        return_type: FFT_RETURN_TYPE = FFT_RETURN_TYPE.LINEAR,          
    ):
        if signal is None or signal.size == 0:
            return np.array([]),np.array([])

        if len(signal) < window_size:
            signal = np.pad(signal,(0,window_size-len(signal)))

        fmin = float(self.semitone_to_freq(self.semitones_A0))
        n_bins = int(self.semitones_A8 - self.semitones_A0+1)
        bins_per_octave = int(self.tet) # let's give ourself some more resolution for kicks

        hop_length = max(1,window_size)

        C = librosa.cqt(
            y=signal.astype(np.float32),
            sr=self.sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            filter_scale=zero_pad_factor,
            window='hann',
            pad_mode='constant'
        )

        # collapse time axis to a single vector (mean across frames if >1)
        if C.ndim == 2 and C.shape[1] > 1:
            C_vec = np.mean(C, axis=1)
        else:
            C_vec = C[:,0] if C.ndim == 2 else C

        semitone_bins = np.arange(self.semitones_A0, self.semitones_A8+1, dtype=np.float32)

        match return_type:
            case self.FFT_RETURN_TYPE.COMPLEX:
                return C_vec, semitone_bins
            case self.FFT_RETURN_TYPE.LINEAR:
                mags = np.abs(C_vec)
                if smoothing_sigma and smoothing_sigma > 0:
                    mags = gaussian_filter1d(mags, sigma=smoothing_sigma)
                return mags, semitone_bins
            case self.FFT_RETURN_TYPE.DECIBELS:
                mags = np.abs(C_vec)
                if smoothing_sigma and smoothing_sigma > 0:
                    mags = gaussian_filter1d(mags, sigma=smoothing_sigma)
                return self.to_decibels(mags, ref=np.max(mags) if mags.size else 1.0), semitone_bins
            case _:
                mags = np.abs(C_vec)
                if smoothing_sigma and smoothing_sigma > 0:
                    mags = gaussian_filter1d(mags, sigma=smoothing_sigma)
                return mags, semitone_bins
            
    def CQT_graph(
        self, 
        full_signal:np.ndarray, 
        window_size: int = 1024,
        hop_size: int = 512,
        zero_pad_factor: int = 2, # Zero padding multiplier for higher FFT bin density
        smoothing_sigma: float = 0.0,
        return_nearest_note: bool = False,
        return_type: FFT_RETURN_TYPE = FFT_RETURN_TYPE.DECIBELS        
    ):
        if len(full_signal) < window_size:
            full_signal = np.pad(full_signal,(0,window_size - len(full_signal)))

        compute = []
        for i in range(0,len(full_signal) - window_size,hop_size):
            frame = full_signal[i:i+window_size]

            # compute spectrum of this frame
            compute_mag,_ = self.CQT(
                signal=frame,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                return_type=return_type
            )
            
            compute.append(compute_mag)

        return np.array(compute).T
    
    """
    compute_chromagraph
    > get the chromagraph of a signal
    > computes the average energy over provided energy bins
    > returns a chromagraph
    """
    def chromagraph(
        self,
        signal: np.ndarray,
        window_size: int = 4096,
        zero_pad_factor: int = 4, # Zero padding multiplier for higher FFT bin density
        spectrum_values_db: np.ndarray | None = None, # alternatively, pass in our spectrum
        frequency_bins: np.ndarray | None = None # this way we don't have to redo the fft computation
    ):
        # If we're given the spectrum, skip the whole fft step
        if spectrum_values_db is not None and frequency_bins is not None:
            spectrum_magnitude = spectrum_values_db
            spectrum_frequency = frequency_bins
        else:
            if len(signal) < window_size:
                signal = np.pad(signal,(0,window_size-len(signal)))

            spectrum_magnitude,spectrum_frequency = self.spectrum(
                signal=signal,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                smoothing_sigma=1,
                return_type=self.FFT_RETURN_TYPE.DECIBELS
            )

        if spectrum_frequency.size == 0:
            return np.array([],dtype=np.float32), np.array([],dtype=np.int32)

        # get note chroma by mean root squaring our frequency bins
        semitone_bands = np.arange(self.semitones_A0,self.semitones_A8,1,dtype=np.int32)
        STMSP = np.zeros_like(semitone_bands,dtype=np.float32)
        for i,st in enumerate(semitone_bands):
            f0 = self.semitone_to_freq(st-1)
            f1 = self.semitone_to_freq(st)
            f2 = self.semitone_to_freq(st+1)

            low = 0.5*(f0+f1)
            high = 0.5*(f1+f2)

            # compute root mean square of this frequency band
            mask = (spectrum_frequency >= low) & (spectrum_frequency <= high)
            spec_sub_band = spectrum_magnitude[mask]

            if spec_sub_band is None:
                STMSP[i] = 0.0
                continue

            if len(spec_sub_band) == 0:
                STMSP[i] = 0.0
                continue

            rms = np.sqrt(np.sum(np.square(spec_sub_band))/len(spec_sub_band))
            #STMSP((f1,rms)) # append the current semitone and its rms
            STMSP[i] = rms

        v = np.reshape(STMSP,(-1,12)).sum(axis=0)

        # A,Bb,B,B,C#,D,Eb,E,F,F#,G,Ab
        return v

    """
    harmonic_comb_filter
    > computes a harmonic comb filter for a given signal
    > Can provide us with the most prominent notes in a given piece
    """
    def harmonic_comb_filter(
        self,
        signal: np.ndarray,
        window_size: int = 8192, # Higher window size is generally what we want
        zero_pad_factor: int = 4, # Zero padding multiplier for higher FFT bin density
        spectrum_values_linear: np.ndarray | None = None, # alternatively, pass in our spectrum
        frequency_bins: np.ndarray | None = None, # this way we don't have to redo the fft computation
        max_residuals: int = 6, # Number of spectrum residuals to check
        max_overtones: int = 12, # max overtones to check
    ):
        if spectrum_values_linear is not None and frequency_bins is not None:
            spectrum_magnitude = spectrum_values_linear
            spectrum_frequency = frequency_bins
        else:
            if len(signal) < window_size:
                signal = np.pad(signal,(0,window_size-len(signal)))
    
            spectrum_magnitude,spectrum_frequency = self.spectrum(
                signal=signal,
                window_size=window_size,
                zero_pad_factor=zero_pad_factor,
                smoothing_sigma=1,
                return_type=self.FFT_RETURN_TYPE.LINEAR
            )
    
        if spectrum_frequency.size == 0:
            return np.array([],dtype=np.float32), np.array([],dtype=np.int32)

        # Weight function dependent on overtone value to compute salience
        def weight_partial(i):
            sigma = 1.5
            return np.exp(-(i*i)/(2*sigma*sigma)) / np.sqrt(i)

        # Find max value near a given frequency with respect to our window delta
        # return freq, energy. (freq: the frequency bin with the max value, energy: the value of that freq)
        def amp_at_freq(freq,spec_mag,spec_freq,delta=1.0):
            #delta = 1.0 # 1 hz window radius -> 2 hz window size
            window_mask = np.abs(spec_freq - freq) <= delta # Get the closest frequency bins

            spec_in_window = np.abs(spec_mag[window_mask])
            freq_in_window = spec_freq[window_mask]
            amp = np.max(spec_in_window)
            freq = freq_in_window[np.argmax(spec_in_window)]
            return freq, amp
            #return np.max(np.abs(spec_mag[np.abs(spec_freq-freq) <= delta])) # Assuming this works the way I intended

        # Salience of a harmonic partial with base frequency f0
        # This salience function is just computed on vibes, LETS FIX THAT!!!
        # I think we can try to better understand what this should be represented as
        def salience(f0,spec_mag,spec_freq):
            s = 0
            fbs = [] # frequencies viewed
            for i in range(1, max_overtones+1):
                # energy, frequency bin
                en,fb = amp_at_freq(i*f0,spec_mag,spec_freq)*weight_partial(i)
                s += en
                fbs.append(fb)
            return s / max_overtones, fbs

        residual_mag = np.asarray(spectrum_magnitude,dtype=np.float32)
        residual_freq = np.asarray(spectrum_frequency,dtype=np.float32)
        selected_freqs = []
        selected_scores = []

        for _ in range(max_residuals):

            # get peaks
            peaks_idx, props = find_peaks(
                residual_mag,
                height=np.max(residual_mag)*0.01
            )
            if peaks_idx.size == 0:
                break

            candidates: list[tuple[Any,float]] = [] #Array of (frequency, salience)

            # Assume each peak could be an overtone instead of f0
            # Loop 1: Go through all peak indices
            for peak_idx in peaks_idx:
                # Loop 2: Go through all possible partials this peak could be at
                for partial_num in range(max_overtones):
                    # Given partial, compute the base frequency
                    f0 = spectrum_frequency[peak_idx]/(partial_num+1)
                    if f0 <= 0:
                        continue
                    score = salience(f0,residual_mag,residual_freq)
                    candidates.append((f0,score))

            # order array by salience value
            salience_array_sorted = sorted(candidates,key=lambda tup: tup[1], reverse=True)

            # Get the first one
            best_freq,best_salience = salience_array_sorted[0]

            # Append to our list of best-fit notes
            selected_freqs.append(best_freq)
            selected_scores.append(best_salience)

            # DEBUG: Print the salience of ALL
            for f,s in salience_array_sorted:
                print(f"Note:{self.get_note_and_difference(f)}. Salience:{s}")

            # Now delete the note from the spectrum

            

            #delta = max(1.0, best_freq*0.01)
            #mask = np.abs(residual_freq - best_freq) > delta #???
            #residual_mag = residual_mag[mask]
            #residual_freq = residual_freq[mask]

        return np.asarray(selected_freqs), np.asarray(selected_scores)




    
                