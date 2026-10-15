from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable, List
import wave
from signalprocessor import SignalProcessor
from PyQt6.QtCore import QObject, pyqtSignal, QIODevice

from PyQt6.QtMultimedia import (
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
    QAudio,
)

import numpy as np

@dataclass
class AudioMetadata:
    """Pure data"""
    filename: str
    nchannels: int
    sample_rate: int
    sample_width: int
    frame_count: int
    frame_count_per_channel: int #TODO: Implement me!

@dataclass
class PlaybackState:
    """Pure data"""
    status: QAudio.State
    current_frame: int
    duration_frames: int

class AudioDataModel(QObject):
    """Manages audio state. Signals when data changes."""
    
    data_changed = pyqtSignal()  # Emitted when ANY state changes
    
    # TODO: initialize our signal processor based on a sample rate
    def __init__(self):
        super().__init__()
        self.metadata: AudioMetadata = AudioMetadata("none",0,0,0,0,0)
        self.pcm_samples: np.ndarray | None = None
        self.playback_state = PlaybackState(QAudio.State.StoppedState, 0, 0)
        
        self._lock = threading.Lock()
        self._processor: SignalProcessor | None = None
        
        # Caches for expensive computations
        self._spectrum_cache = {}
        self._spectrum_linear_cache = {}
        self._cepstrum_cache = {}
        self._spectrogram_cache = {}
        self._cqt_cache = {}
        #self._partials_cache = {}

    def frame_to_byte_offset(self, frame: int) -> int:
        if self.metadata is None:
            return 0
        samplewidth = self.metadata.sample_width
        nchannels = self.metadata.nchannels
        return frame * nchannels * samplewidth
    
    def load_audio(self, filepath: str):
        """Load audio file"""
        with self._lock:

            self._spectrum_cache.clear()
            self._spectrum_linear_cache.clear()
            self._cepstrum_cache.clear()
            self._spectrogram_cache.clear()
            self._cqt_cache.clear()

            with wave.open(filepath, 'rb') as wav_file:
                nchannels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                
                self.metadata = AudioMetadata(
                    filename=filepath,
                    nchannels=nchannels,
                    sample_rate=sample_rate,
                    sample_width=sample_width,
                    frame_count=frame_count,
                    frame_count_per_channel=0 #for now
                )
                
                # Read all frames
                frames = wav_file.readframes(frame_count)
                pcm_int = np.frombuffer(frames, dtype=np.int16)
                
                # Reshape to (channels, samples)
                if nchannels > 1:
                    pcm_int = pcm_int.reshape(-1, nchannels).T
                else:
                    pcm_int = pcm_int.reshape(1, -1)
                
                self._processor = SignalProcessor(sample_rate)
                self.pcm_samples = self._processor.normalize_pcm(pcm_int, sample_width)
                
                self.playback_state = PlaybackState(
                    QAudio.State.StoppedState, 0, frame_count
                )
        
        self.data_changed.emit()
    
    def get_samples(self, channel: int, start_frame: int, num_frames: int) -> np.ndarray:
        with self._lock:
            if self.pcm_samples is None:
                return np.array([])
            # modify this so it accounts for us reaching the end of the array
            return self.pcm_samples[channel,start_frame:start_frame+num_frames]
        
    # just return everything (TODO: Prevent this from being used!)
    def get_pcm_bytes(self) -> bytes:
        with self._lock:
            if self.pcm_samples is None or self.metadata is None:
                return b""
            
            sample_width = self.metadata.sample_width
            nchannels = self.metadata.nchannels
            samples = self.pcm_samples

            int_samples: np.ndarray = np.zeros((nchannels,100)) #I think like this?
            if samples.dtype.kind == "f":
                if sample_width == 2:
                    int_samples = np.clip(samples * (2**15)-1,-(2**15),(2**15)-1).astype(np.int16)
                elif sample_width == 1:
                    int_samples = np.clip((samples * (2**7)+(2**7)),0,(2**8)-1).astype(np.uint8)
                elif sample_width == 4:
                    int_samples = np.clip(samples * (2**31),-(2**31),(2**31)-1).astype(np.int32)
                else:
                    raise ValueError(f"Unsupported Sample Witdh: {sample_width}")
            else:
                if sample_width == 2:
                    pass
                elif sample_width == 1:
                    pass
                elif sample_width == 4:
                    pass
                else:
                    raise ValueError(f"Unsupported Sample Witdh: {sample_width}")
                
            if nchannels > 1:
                interleaved = int_samples.T.reshape(-1)
            else:
                interleaved = int_samples[0]

            return interleaved.tobytes()
                     
    def get_spectrum(self, channel: int, frame:int, window_size: int = 4096) -> np.ndarray:
        cache_key = (channel, frame, window_size)

        # our cache system coming in clutch
        if cache_key in self._spectrum_cache:
            return self._spectrum_cache[cache_key]
        
        samples = self.get_samples(channel, frame, window_size)
        if len(samples) == 0:
            return np.zeros(window_size // 2 + 1) #nyqyist theorem type shi
        
        if self._processor is None:
            return np.zeros(window_size // 2 + 1)
        
        spec_db,spec_bin = self._processor.spectrum(
            samples, 
            window_size,
            return_type=SignalProcessor.FFT_RETURN_TYPE.DECIBELS
        )
        self._spectrum_cache[cache_key] = spec_db
        return spec_db
    
    def get_spectrogram(self, channel: int, window_size: int = 8192, hop_size: int = 8192) -> np.ndarray:
        cache_key = (channel, window_size)

        if cache_key in self._spectrogram_cache and self.pcm_samples is not None:
            return self._spectrogram_cache[cache_key]
        
        if self.pcm_samples is None or self._processor is None:
            return np.zeros(shape=(2,2))

        samples = self.get_samples(channel, 0, len(self.pcm_samples[channel]))
        spec_mag = self._processor.spectrogram(
            samples,
            window_size,
            hop_size=hop_size,
            smoothing_sigma=0.4,
            return_type=SignalProcessor.FFT_RETURN_TYPE.DECIBELS
        )
        self._spectrogram_cache[cache_key] = spec_mag
        return spec_mag

    def get_Constant_Q_Transform(self, channel: int, window_size: int = 8192, hop_size: int = 4096) -> np.ndarray:
        cache_key = (channel, window_size)

        if cache_key in self._cqt_cache and self.pcm_samples is not None:
            return self._cqt_cache[cache_key]
        
        if self.pcm_samples is None or self._processor is None:
            return np.zeros(shape=(2,2))

        samples = self.get_samples(channel, 0, len(self.pcm_samples[channel]))
        cqt_mag = self._processor.CQT_graph(
            samples,
            window_size,
            hop_size=hop_size,
            return_type=SignalProcessor.FFT_RETURN_TYPE.DECIBELS
        )
        self._cqt_cache[cache_key] = cqt_mag
        return cqt_mag        

    def get_cepstrum(self, channel: int, frame: int, window_size: int = 4096, get_power_cepstrum: bool = False) -> np.ndarray:
        cache_key = (channel, frame, window_size)

        # our cache system coming in clutch
        if cache_key in self._cepstrum_cache:
            return self._cepstrum_cache[cache_key]
        
        samples = self.get_samples(channel, frame, window_size)
        if len(samples) == 0:
            return np.zeros(window_size // 2 + 1) #nyqyist theorem type shi
        
        if self._processor is None:
            return np.zeros(window_size // 2 + 1)

        ceps, quef = self._processor.cepstrum(
            samples,
            window_size,
            return_power_cepstrum=get_power_cepstrum
        )
        self._cepstrum_cache[cache_key] = ceps
        return ceps
        
        #spec_fb,spec_ft = self._processor.compute_spectrum(samples, window_size)
        #self._spectrum_cache[cache_key] = spec_ft
        #return spec_ft

    '''def get_note_candidates(self, channel: int, frame: int, window_size: int = 8192, partial_step: float = 1.0):
        cache_key = (channel,frame,window_size)

        if cache_key in self._partials_cache:
            return self._partials_cache[cache_key]

        if self.pcm_samples is None or self._processor is None:
            return []

        samples = self.get_samples(channel,frame,window_size)
        note_candidates = self._processor.identify_partials_in_signal(
            samples,
            window_size=window_size,
        )
        #note_candidates = self._processor.identify_partials_in_signal(
        #    samples,
        #    partial_step=1,
        #    window_size=window_size,
        #    max_overtones=10,
        #    max_notes=10
        #)
        self._partials_cache[cache_key] = note_candidates
        return note_candidates'''
    
    def set_playback_state(self, status: QAudio.State, frame: int):
        with self._lock:
            self.playback_state = PlaybackState(
                status, frame, self.playback_state.duration_frames
            )
        #self.data_changed.emit() # Not nessecary