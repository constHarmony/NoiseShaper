# filters.py
import numpy as np
from scipy import signal
from abc import ABC, abstractmethod
from typing import Tuple
from config import AudioConfig
import scipy.special  # Add this import for error function

class AudioFilter(ABC):
    def __init__(self, config: AudioConfig):
        self.config = config
        self._zi = None
    
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
            if hasattr(self, key):
                setattr(self, key, value)
                updated = True
        if updated:
            self._update_coefficients()

class BandpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, lowcut: float, highcut: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.lowcut = lowcut
        self.highcut = highcut 
        self.order = order
        self.amplitude = amplitude
        self._update_coefficients()  # Removed enabled flag

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
            'amplitude': self.amplitude
        }

class LowpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, cutoff: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.cutoff = cutoff
        self.order = order
        self.amplitude = amplitude
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
            'amplitude': self.amplitude
        }

class HighpassFilter(AudioFilter):
    def __init__(self, config: AudioConfig, cutoff: float, order: int = 4, amplitude: float = 1.0):
        super().__init__(config)
        self.cutoff = cutoff
        self.order = order
        self.amplitude = amplitude
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
            'amplitude': self.amplitude
        }

class NotchFilter(AudioFilter):
    def __init__(self, config: AudioConfig, freq: float, q: float = 30.0, amplitude: float = 1.0):
        super().__init__(config)
        self.freq = freq
        self.q = q
        self.amplitude = amplitude
        self._update_coefficients()

    def _update_coefficients(self):
        nyq = self.config.sample_rate * 0.5
        normal_freq = self.freq / nyq
        self.b, self.a = signal.iirnotch(normal_freq, self.q)
        self._zi = signal.lfilter_zi(self.b, self.a) * 0

    def process(self, data: np.ndarray) -> np.ndarray:
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered * self.amplitude

    def get_name(self) -> str:
        return f"Notch {self.freq:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'notch',
            'frequency': self.freq,
            'q': self.q,
            'amplitude': self.amplitude
        }

class GaussianFilter(AudioFilter):
    def __init__(self, config: AudioConfig, center_freq: float, width: float, 
                 amplitude: float = 1.0, skew: float = 0.0, kurtosis: float = 1.0):
        super().__init__(config)
        self.center_freq = center_freq
        self.width = width
        self.amplitude = amplitude
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
            self.frequencies = np.fft.rfftfreq(size, 1 / self.config.sample_rate)
            
            # Calculate standardized frequency variable
            z = (self.frequencies - self.center_freq) / (self.width + 1e-10)
            # Square z first to ensure positive values
            z_squared = z ** 2
            # Apply kurtosis
            z_kurtosis = z_squared ** self.kurtosis

            # Incorporate skewness using the skew normal distribution
            skewness_term = 1 + scipy.special.erf(self.skew * z / np.sqrt(2))
            self.filter_mask = self.amplitude * np.exp(-z_kurtosis / 2) * skewness_term

            self.last_size = size

    def process(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return data
            
        # Get spectrum first
        spectrum = np.fft.rfft(data)
        
        # Always ensure filter mask exists and matches size
        self._ensure_filter_size(len(data))
        
        # Apply filter and transform back
        try:
            filtered_spectrum = spectrum * self.filter_mask[:len(spectrum)]
            return np.fft.irfft(filtered_spectrum, n=len(data)).astype(np.float32)
        except Exception as e:
            print(f"Filter processing error: {str(e)}, sizes: data={len(data)}, spectrum={len(spectrum)}, mask={len(self.filter_mask) if self.filter_mask is not None else 'None'}")
            return data

    def get_name(self) -> str:
        return f"Gaussian {self.center_freq:.0f}Hz ±{self.width:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'gaussian',
            'center_freq': self.center_freq,
            'width': self.width,
            'amplitude': self.amplitude,
            'skew': self.skew,
            'kurtosis': self.kurtosis
        }

class ParabolicFilter(AudioFilter):
    def __init__(self, config: AudioConfig, center_freq: float, width: float, 
                 amplitude: float = 1.0, skew: float = 0.0, flatness: float = 1.0):
        super().__init__(config)
        self.center_freq = center_freq
        self.width = width
        self.amplitude = amplitude
        self.skew = skew  # -1000 to 1000, frequency shift in Hz
        self.flatness = flatness  # 0.2 to 5.0, wider range for shaping
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
            self.frequencies = np.fft.rfftfreq(size, 1 / self.config.sample_rate)
            
            # Calculate standardized frequency variable
            z = (self.frequencies - self.center_freq) / (self.width + 1e-10)
            # Square z first to ensure positive values
            z_squared = z ** 2
            # Apply flatness
            z_flatness = z_squared ** self.flatness

            # Parabolic shape with flatness adjustment
            base_shape = np.maximum(1 - z_flatness, 0)

            # Incorporate skewness using a skew factor
            skewness_term = 1 + scipy.special.erf(self.skew * z / np.sqrt(2))
            self.filter_mask = self.amplitude * base_shape * skewness_term

            self.last_size = size

    def process(self, data: np.ndarray) -> np.ndarray:
        if len(data) == 0:
            return data
            
        # Get spectrum first
        spectrum = np.fft.rfft(data)
        
        # Always ensure filter mask exists and matches size
        self._ensure_filter_size(len(data))
        
        # Apply filter and transform back
        try:
            filtered_spectrum = spectrum * self.filter_mask[:len(spectrum)]
            return np.fft.irfft(filtered_spectrum, n=len(data)).astype(np.float32)
        except Exception as e:
            print(f"Filter processing error: {str(e)}, sizes: data={len(data)}, spectrum={len(spectrum)}, mask={len(self.filter_mask) if self.filter_mask is not None else 'None'}")
            return data

    def get_name(self) -> str:
        return f"Parabolic {self.center_freq:.0f}Hz ±{self.width:.0f}Hz"

    def get_parameters(self) -> dict:
        return {
            'type': 'parabolic',
            'center_freq': self.center_freq,
            'width': self.width,
            'amplitude': self.amplitude,
            'skew': self.skew,
            'flatness': self.flatness
        }

class AudioNormalizer:
    @staticmethod
    def normalize_signal(signal: np.ndarray, target_amplitude: float = 1.0) -> np.ndarray:
        """
        Normalize signal to [-1,1] range then scale by target amplitude
        
        Args:
            signal: Input signal
            target_amplitude: Desired peak amplitude (0.0 to 1.0)
        Returns:
            Normalized and scaled signal in range [-target_amplitude, target_amplitude]
        """
        # First normalize to [-1,1]
        if np.any(signal != 0):
            max_val = np.max(np.abs(signal))
            if max_val > 0:
                signal = signal / max_val
                
        # Then scale to target amplitude
        return signal * target_amplitude
