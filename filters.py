# filters.py
import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
from typing import Tuple
from config import AudioConfig
import scipy.special  # Add this import for error function
import logging

logger = logging.getLogger(__name__)

__all__ = [
    'AudioFilter',
    'BandpassFilter',
    'LowpassFilter',
    'HighpassFilter',
    'NotchFilter',
    'GaussianFilter',
    'ParabolicFilter',
    'PlateauFilter',  # Add this
    'AudioNormalizer'
]

class AudioFilter(ABC):
    def __init__(self, config: AudioConfig):
        self.config = config
        self._zi = None
        self.filter_mask = None
        self.last_size = None
        self._gain_db = 0.0  # Store gain in dB
        self._amplitude = 1.0  # Cached linear amplitude
    
    @property
    def gain_db(self) -> float:
        """Get gain in dB"""
        return self._gain_db
        
    @gain_db.setter
    def gain_db(self, db: float):
        """Set gain in dB and update cached linear amplitude"""
        self._gain_db = db
        # Convert dB to linear amplitude, handling -inf dB case
        self._amplitude = 0.0 if db <= -120 else 10 ** (db / 20.0)
        
    @property
    def amplitude(self) -> float:
        """Get linear amplitude (read-only)"""
        return self._amplitude

    @abstractmethod
    def process(self, data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        pass

    def update_parameters(self, params: dict):
        """Default parameter update implementation"""
        updated = False
        for key, value in params.items():
            if key == 'amplitude':
                # Convert linear amplitude to dB for backward compatibility
                if value <= 0:
                    self.gain_db = -120  # Effectively -inf dB
                else:
                    self.gain_db = 20 * np.log10(value)
                updated = True
            elif key == 'gain_db':
                self.gain_db = value
                updated = True
            elif hasattr(self, key):
                setattr(self, key, value)
                updated = True
        if updated:
            self._update_coefficients()

    def _ensure_filter_size(self, size: int):
        """Default implementation for IIR filters - creates frequency response mask"""
        if size == self.last_size and self.filter_mask is not None:
            return

        # Calculate number of points needed for freqz
        if size % 2 == 0:  # Even-sized array
            nfreqs = size // 2 + 1  # Include DC and Nyquist
        else:  # Odd-sized array
            nfreqs = (size + 1) // 2  # Include DC and Nyquist

        # Get frequency response of the filter
        w, h = signal.freqz(self.b, self.a, worN=nfreqs)
        freqs = w * self.config.sample_rate / (2 * np.pi)
        
        # Convert to magnitude response
        magnitude = np.abs(h)
        
        # Create full spectrum mask with symmetry
        self.filter_mask = np.zeros(size)
        midpoint = size // 2
        
        # Handle both even and odd-sized arrays consistently
        if size % 2 == 0:  # Even-sized array
            self.filter_mask[:midpoint] = magnitude[:-1]  # Exclude Nyquist
            self.filter_mask[midpoint:] = magnitude[-2::-1]  # Mirror excluding Nyquist
        else:  # Odd-sized array
            self.filter_mask[:midpoint] = magnitude[:-1]  # Exclude Nyquist
            self.filter_mask[midpoint] = magnitude[-1]  # Center point (Nyquist)
            self.filter_mask[midpoint+1:] = magnitude[-2::-1]  # Mirror excluding Nyquist
            
        self.last_size = size

class BandpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, lowcut: float, highcut: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.lowcut = lowcut
        self.highcut = highcut 
        self.order = order
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self._update_coefficients()

    def _update_coefficients(self):
        nyq = self.config.sample_rate * 0.5
        
        # Handle equal frequencies by slightly adjusting highcut
        if self.lowcut == self.highcut:
            actual_highcut = self.highcut + 0.1  # Add tiny offset
        else:
            actual_highcut = self.highcut
            
        # Use butter with proper parameters
        self.b, self.a = signal.butter(
            self.order, 
            [self.lowcut/nyq, actual_highcut/nyq], 
            btype='band'
        )
        self._zi = signal.lfilter_zi(self.b, self.a) * 0
        # Reset filter mask when coefficients change
        self.filter_mask = None
        self.last_size = None

    def process(self, data: np.ndarray) -> np.ndarray:
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered * self.amplitude

    def get_name(self) -> str:
        return f"Bandpass {self.lowcut:.0f}-{self.highcut:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'bandpass',
            'lowcut': self.lowcut,
            'highcut': self.highcut,
            'order': self.order,
            'gain_db': self.gain_db
        }

class LowpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, cutoff: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.cutoff = cutoff
        self.order = order
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self._update_coefficients()

    def _update_coefficients(self):
        nyq = self.config.sample_rate * 0.5
        normal_cutoff = self.cutoff / nyq
        
        # Use butter with proper parameters
        self.b, self.a = signal.butter(
            self.order, 
            normal_cutoff,
            btype='low'
        )
        self._zi = signal.lfilter_zi(self.b, self.a) * 0
        # Reset filter mask when coefficients change
        self.filter_mask = None
        self.last_size = None

    def process(self, data: np.ndarray) -> np.ndarray:
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered * self.amplitude

    def get_name(self) -> str:
        return f"Lowpass {self.cutoff:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'lowpass',
            'cutoff': self.cutoff,
            'order': self.order,
            'gain_db': self.gain_db
        }

class HighpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, cutoff: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.cutoff = cutoff
        self.order = order
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self._update_coefficients()

    def _update_coefficients(self):
        nyq = self.config.sample_rate * 0.5
        normal_cutoff = self.cutoff / nyq
        
        # Use butter with proper parameters
        self.b, self.a = signal.butter(
            self.order, 
            normal_cutoff,
            btype='high'
        )
        self._zi = signal.lfilter_zi(self.b, self.a) * 0
        # Reset filter mask when coefficients change
        self.filter_mask = None
        self.last_size = None

    def process(self, data: np.ndarray) -> np.ndarray:
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered * self.amplitude

    def get_name(self) -> str:
        return f"Highpass {self.cutoff:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'highpass',
            'cutoff': self.cutoff,
            'order': self.order,
            'gain_db': self.gain_db
        }

class NotchFilter(AudioFilter):
    def __init__(self, config: AudioConfig, freq: float, q: float = 30.0, amplitude: float = 1.0):
        super().__init__(config)
        self.freq = freq
        self.q = q
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self._update_coefficients()

    def _update_coefficients(self):
        nyq = self.config.sample_rate * 0.5
        normal_freq = self.freq / nyq
        self.b, self.a = signal.iirnotch(normal_freq, self.q)
        self._zi = signal.lfilter_zi(self.b, self.a) * 0
        # Reset filter mask when coefficients change
        self.filter_mask = None
        self.last_size = None

    def process(self, data: np.ndarray) -> np.ndarray:
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered * self.amplitude

    def get_name(self) -> str:
        return f"Notch {self.freq:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'notch',
            'freq': self.freq,
            'q': self.q,
            'gain_db': self.gain_db
        }

class GaussianFilter(AudioFilter):
    def __init__(self, config: AudioConfig, center_freq: float, width: float, 
                 amplitude: float = 1.0, skew: float = 0.0, kurtosis: float = 1.0):
        super().__init__(config)
        self.center_freq = center_freq
        self.width = width
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self.skew = skew  # -1000 to 1000, frequency shift in Hz
        self.kurtosis = kurtosis  # 0.2 to 5.0, wider range for more shaping
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None
        self._ensure_filter_size(self.config.fft_size)  # Initialize with default size

    def _update_coefficients(self):
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None

    def _ensure_filter_size(self, size: int):
        """Create frequency array and filter mask for current size"""
        if size != self.last_size:
            # Use full FFT frequencies
            self.frequencies = np.fft.fftfreq(size, 1 / self.config.sample_rate)
            
            # Calculate standardized frequency variable
            z = (self.frequencies - self.center_freq) / (self.width + 1e-10)
            # Square z first to ensure positive values
            z_squared = z ** 2
            # Apply kurtosis
            z_kurtosis = z_squared ** self.kurtosis

            # Incorporate skewness using the skew normal distribution
            skewness_term = 1 + scipy.special.erf(self.skew * z / np.sqrt(2))
            self.filter_mask = np.exp(-z_kurtosis / 2) * skewness_term  # Removed amplitude

            self.last_size = size

    def process(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return data
            
        # Use full FFT
        spectrum = np.fft.fft(data)
        
        # Always ensure filter mask exists and matches size
        self._ensure_filter_size(len(data))
        
        try:
            # Apply filter mask and amplitude
            filtered_spectrum = spectrum * self.filter_mask * self.amplitude
            return np.fft.ifft(filtered_spectrum).real.astype(np.float32)
        except Exception as e:
            logger.error(f"Filter processing error: {str(e)}, sizes: data={len(data)}, spectrum={len(spectrum)}, mask={len(self.filter_mask) if self.filter_mask is not None else 'None'}")
            return data

    def get_name(self) -> str:
        return f"Gaussian {self.center_freq:.0f}Hz ±{self.width:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'gaussian',
            'center_freq': self.center_freq,
            'width': self.width,
            'gain_db': self.gain_db,
            'skew': self.skew,
            'kurtosis': self.kurtosis
        }

class ParabolicFilter(AudioFilter):
    def __init__(self, config: AudioConfig, center_freq: float, width: float, 
                 amplitude: float = 1.0, skew: float = 0.0, flatness: float = 1.0):
        super().__init__(config)
        self.center_freq = center_freq
        self.width = width
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self.skew = skew
        self.flatness = flatness
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None
        self._ensure_filter_size(self.config.fft_size)  # Initialize with default size

    def _update_coefficients(self):
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None

    def _ensure_filter_size(self, size: int):
        """Create frequency array and filter mask - exact match to v5.3.1"""
        if size != self.last_size:
            # Use full FFT frequencies
            freqs = np.fft.fftfreq(size, 1 / self.config.sample_rate)
            
            # Initialize mask with zeros like v5.3.1
            mask = np.zeros_like(freqs)
            
            # Calculate mask exactly like v5.3.1
            freq_diff = np.abs(freqs - self.center_freq)
            mask_indices = freq_diff <= self.width
            mask[mask_indices] = 1 - (freq_diff[mask_indices] / self.width) ** 2
            
            # Apply symmetry for real-valued signal
            midpoint = len(mask) // 2
            if len(mask) % 2 == 0:  # Even-sized array
                mask[midpoint:] = mask[midpoint-1::-1]
            else:  # Odd-sized array
                mask[midpoint+1:] = mask[midpoint-1::-1]
            
            # Store mask without amplitude
            self.filter_mask = mask
            self.last_size = size

    def process(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return data
            
        # Use full FFT
        spectrum = np.fft.fft(data)
        
        # Always ensure filter mask exists and matches size
        self._ensure_filter_size(len(data))
        
        try:
            # Apply filter mask and amplitude
            filtered_spectrum = spectrum * self.filter_mask * self.amplitude
            return np.fft.ifft(filtered_spectrum).real.astype(np.float32)
        except Exception as e:
            logger.error(f"Filter processing error: {str(e)}")
            return data

    def get_name(self) -> str:
        return f"Parabolic {self.center_freq:.0f}Hz ±{self.width:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'parabolic',
            'center_freq': self.center_freq,
            'width': self.width,
            'gain_db': self.gain_db,
            'skew': self.skew,
            'flatness': self.flatness
        }

class PlateauFilter(AudioFilter):
    """Filter implementing plateau curve with flat center and cosine rolloff"""
    def __init__(self, config: AudioConfig, center_freq: float, width: float, 
                 flat_width: float, amplitude: float = 1.0):
        super().__init__(config)
        self.center_freq = center_freq
        # Ensure width is at least 1 Hz larger than flat_width
        self.width = max(width, flat_width + 1)
        self.flat_width = flat_width  # Width of flat section (full delta)
        # Convert amplitude to gain_db
        self.gain_db = 0.0 if amplitude == 1.0 else 20 * np.log10(amplitude)
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None
        self._ensure_filter_size(self.config.fft_size)  # Initialize with default size

    def _update_coefficients(self):
        self.frequencies = None
        self.filter_mask = None
        self.last_size = None

    def _ensure_filter_size(self, size: int):
        """Create frequency array and filter mask"""
        if size == self.last_size and self.filter_mask is not None:
            return

        # Use full FFT frequencies
        freqs = np.fft.fftfreq(size, 1 / self.config.sample_rate)
            
        # Use full width values
        delta = self.flat_width  # Full flat section width
        width = self.width  # Full width range
            
        # Calculate initial mask
        freq_diff = np.abs(freqs - self.center_freq)
        
        # Use np.where for vectorized operation
        mask = np.where(
            freq_diff < delta,
            1.0,
            np.where(
                freq_diff <= width,
                0.5 * (1 + np.cos(np.pi * (freq_diff - delta) / (width - delta))),
                0.0
            )
        )
            
        # Apply symmetry for real-valued signal
        midpoint = len(mask) // 2
        if len(mask) % 2 == 0:  # Even-sized array
            mask[midpoint:] = mask[midpoint-1::-1]
        else:  # Odd-sized array
            mask[midpoint+1:] = mask[midpoint-1::-1]
            
        # Store mask without amplitude
        self.filter_mask = mask
        self.last_size = size

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process audio data through the filter"""
        if len(data) == 0:
            return data
            
        # Use full
        spectrum = np.fft.fft(data)
        self._ensure_filter_size(len(data))
        
        try:
            # Apply filter mask and amplitude
            filtered_spectrum = spectrum * self.filter_mask * self.amplitude
            return np.fft.ifft(filtered_spectrum).real.astype(np.float32)
        except Exception as e:
            logger.error(f"Filter processing error: {str(e)}")
            return data

    def get_name(self) -> str:
        return f"Plateau {self.center_freq:.0f}Hz ±{self.width:.0f}Hz"
        
    def get_parameters(self) -> dict:
        return {
            'type': 'plateau',
            'center_freq': self.center_freq,
            'width': self.width,
            'flat_width': self.flat_width,
            'gain_db': self.gain_db
        }

class AudioNormalizer:
    @staticmethod
    def normalize_signal(signal: np.ndarray, target_amplitude: float = 1.0) -> np.ndarray:
        """
        Normalize signal to [-1,1] range then scale by target amplitude
        """
        logger.debug(f"\nDEBUG: AudioNormalizer.normalize_signal:")
        logger.debug(f"- Input signal max abs: {np.max(np.abs(signal))}")
        logger.debug(f"- Target amplitude: {target_amplitude}")
        
        # Get the maximum absolute value
        max_abs = np.max(np.abs(signal))
        
        # Avoid division by zero
        if max_abs > 0:
            # First normalize to [-1,1] range
            normalized = signal / max_abs
            # Then scale to target amplitude
            result = normalized * target_amplitude
        else:
            result = signal
            
        logger.debug(f"- Output signal max abs: {np.max(np.abs(result))}")
        return result
