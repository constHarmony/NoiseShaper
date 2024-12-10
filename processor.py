# processor.py
import numpy as np
from typing import Optional, List, Tuple
from config import AudioConfig
from audio_sources import AudioSource, NoiseSource
from filters import AudioFilter
from scipy import signal




class AudioProcessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.source = None
        self.filters = []
        self.window = None
        self._prev_chunk_size = None
        self._analysis_buffer = np.array([], dtype=np.float32)
        self.update_window()

    def set_source(self, source: AudioSource):
        """Set the audio source and close any existing source"""
        try:
            if self.source is not None:
                self.source.close()
            self.source = source
            
            # Re-add existing filters to noise source
            if isinstance(self.source, NoiseSource):
                for f in self.filters:
                    self.source.add_filter(f)
        except Exception as e:
            print(f"Error setting source: {e}")
            self.source = None
            raise

    def update_window(self, size: int = None):
        if size is None:
            size = self.config.fft_size  # Changed from chunk_size to fft_size
        # Create window of the appropriate size
        if self.config.window_type == 'hanning':
            self.window = np.hanning(size)
        elif self.config.window_type == 'hamming':
            self.window = np.hamming(size)
        elif self.config.window_type == 'blackman':
            self.window = np.blackman(size)
        elif self.config.window_type == 'flattop':
            self.window = signal.windows.flattop(size)
        else:  # rectangular
            self.window = np.ones(size)
        # Normalize window
        self.window = self.window / np.sqrt(np.sum(self.window**2))

    def process(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.source or not self.source.is_running:
            return np.array([]), np.array([])

        try:
            # Get data from source
            data = self.source.read()  # Use read() instead of read_analysis()
            if data.size == 0:
                return np.array([]), np.array([])

            # Ensure data size matches FFT size
            if len(data) > self.config.fft_size:
                data = data[:self.config.fft_size]
            elif len(data) < self.config.fft_size:
                data = np.pad(data, (0, self.config.fft_size - len(data)))

            # Apply windowing
            windowed_data = data * self.window
            
            # Perform FFT
            spec = np.fft.rfft(windowed_data)
            freq = np.fft.rfftfreq(len(windowed_data), 1/self.config.sample_rate)
            
            # Calculate magnitude with proper scaling
            magnitude = np.abs(spec)
            magnitude[1:-1] *= 2  # Compensate for one-sided spectrum
            
            # Convert to dB with safety checks
            with np.errstate(divide='ignore', invalid='ignore'):
                spec_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
                spec_db = np.nan_to_num(spec_db, nan=-120, posinf=-120, neginf=-120)
            
            # Apply output scaling
            spec_db += 20 * np.log10(2)  # Additional scaling factor
            spec_db = np.clip(spec_db, self.config.min_db, self.config.max_db)

            return freq, spec_db

        except Exception as e:
            print(f"Processing error: {str(e)}")
            return np.array([]), np.array([])

    def add_filter(self, filter_):
        """Add a filter to the processing chain"""
        self.filters.append(filter_)
        if isinstance(self.source, NoiseSource):
            self.source.add_filter(filter_)

    def remove_filter(self, index: int):
        """Remove a filter from the processing chain"""
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
            if isinstance(self.source, NoiseSource):
                self.source.remove_filter(index)

    def update_filter(self, index: int, params: dict):
        """Update filter parameters"""
        if 0 <= index < len(self.filters):
            params = params.copy()
            filter_type = params.pop('type', None) 
            self.filters[index].update_parameters(params)
            if isinstance(self.source, NoiseSource):
                self.source.update_filter(index, params)

    def close(self):
        """Close the audio source and clean up"""
        if self.source is not None:
            self.source.close()
            self.source = None