# processor.py
import numpy as np
from typing import Optional, List, Tuple
from config import AudioConfig
from audio_sources import AudioSource, NoiseSource, MonitoredInputSource
from filters import AudioFilter
from scipy import signal
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    # Trigger modes
    TRIGGER_RESET_HOLD = "hold_time"    # Reset after hold time
    TRIGGER_RESET_NEXT = "next_trigger" # Reset on next trigger
    TRIGGER_RESET_MANUAL = "manual"     # Reset only manually
    
    TRIGGER_EDGE_RISING = "rising"      # Trigger on rising edge
    TRIGGER_EDGE_FALLING = "falling"    # Trigger on falling edge
    TRIGGER_EDGE_BOTH = "both"          # Trigger on both edges

    def __init__(self, config: AudioConfig):
        self.config = config
        self.source = None
        self.filters = []
        self.window = None
        self._prev_chunk_size = None
        self._analysis_buffer = np.array([], dtype=np.float32)
        self._prev_spectrum = None  # Store previous spectrum for decay
        self._decay_rate = 0.1  # Decay rate per frame (adjustable)
        self._decay_enabled = False  # Add enable flag
        
        # Trigger settings
        self._trigger_enabled = False
        self._trigger_level = -45  # Changed from -60 to -45 dB threshold for trigger
        self._hold_time = 0.2  # seconds to hold the peak
        self._trigger_samples = 0  # counter for hold time
        self._triggered = False
        self._peak_spectrum = None
        self._prev_max_db = -120   # For edge detection
        self._trigger_reset_mode = self.TRIGGER_RESET_NEXT  # Default to next trigger
        self._trigger_edge_mode = self.TRIGGER_EDGE_RISING  # Default to rising edge
        
        self.update_window()

    def set_source(self, source: AudioSource):
        """Set the audio source and close any existing source"""
        try:
            if self.source is not None:
                self.source.close()
            self.source = source
            
            # If it's a NoiseSource, use its filter list directly
            if isinstance(self.source, NoiseSource):
                # Transfer any existing filters first if we have our own
                if self.filters and self.filters is not self.source.filters:
                    for f in self.filters:
                        self.source.add_filter(f)
                # Then use the same list reference
                self.filters = self.source.filters  # Direct reference to source's filter list
                logger.debug(f"DEBUG: AudioProcessor set_source - sharing filter list with source")
            
        except Exception as e:
            logger.error(f"Error setting source: {e}")
            self.source = None
            raise

    def update_window(self):
        """Update the window function based on current settings"""
        size = self.config.fft_size
        logger.debug("Updating window:")
        logger.debug(f"- Size: {size}")
        logger.debug(f"- Type: {self.config.window_type}")
        
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
        logger.debug("- Window updated successfully")

    def process(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.source or not self.source.is_running:
            return np.array([]), np.array([])

        try:
            # Get data from source
            data = self.source.read()
            if data.size == 0:
                return np.array([]), np.array([])

            # Check if FFT size has changed
            if len(self.window) != self.config.fft_size:
                logger.debug("FFT size mismatch detected:")
                logger.debug(f"- Window size: {len(self.window)}")
                logger.debug(f"- Config FFT size: {self.config.fft_size}")
                self.update_window()  # Update window if size changed

            # Debug print for data sizes
            if np.random.random() < 0.01:  # Print ~1% of the time
                logger.debug("FFT Debug:")
                logger.debug(f"- Configured FFT size: {self.config.fft_size}")
                logger.debug(f"- Window size: {len(self.window)}")
                logger.debug(f"- Input data length: {len(data)}")

            # Ensure data size matches FFT size
            if len(data) > self.config.fft_size:
                data = data[:self.config.fft_size]
                if np.random.random() < 0.01:
                    logger.debug(f"- Data truncated from {len(data)} to {self.config.fft_size}")
            elif len(data) < self.config.fft_size:
                pad_amount = self.config.fft_size - len(data)
                data = np.pad(data, (0, pad_amount))
                if np.random.random() < 0.01:
                    logger.debug(f"- Data padded from {len(data)-pad_amount} to {len(data)}")

            # Apply windowing
            windowed_data = data * self.window
            
            # Normalize window to preserve amplitude
            window_norm = np.sqrt(np.mean(self.window**2))
            if window_norm > 0:
                windowed_data = windowed_data / window_norm
            
            # Perform full FFT like DIY
            full_spec = np.fft.fft(windowed_data)
            freqs = np.fft.fftfreq(len(windowed_data), 1/self.config.sample_rate)
            
            # Convert to magnitude spectrum - use absolute value of frequencies
            magnitude = np.abs(full_spec)
            freqs = np.abs(freqs)  # Take absolute value of frequencies
            
            # Sort frequencies and magnitudes together
            sort_idx = np.argsort(freqs)
            freqs = freqs[sort_idx]
            magnitude = magnitude[sort_idx]
            
            # Take unique frequencies (combines positive and negative components)
            unique_idx = np.unique(freqs, return_index=True)[1]
            positive_freqs = freqs[unique_idx]
            positive_magnitude = magnitude[unique_idx]
            
            # Double all magnitudes except DC and Nyquist to account for negative frequencies
            if len(positive_magnitude) > 1:
                positive_magnitude[1:] *= 2  # Double all except DC
            
            # Scale to match Audacity's behavior
            positive_magnitude /= len(data)  # Normalize by FFT size like Audacity
            
            # Convert to dB with safety checks
            with np.errstate(divide='ignore', invalid='ignore'):
                spec_db = 20 * np.log10(np.maximum(positive_magnitude, 1e-10))
                spec_db = np.nan_to_num(spec_db, nan=-120, posinf=-120, neginf=-120)

            # Clip to display range without additional scaling
            spec_db = np.clip(spec_db, self.config.min_db, self.config.max_db)

            # Only apply decay and trigger in test mode
            if isinstance(self.source, MonitoredInputSource):
                # Handle trigger mode
                if self._trigger_enabled:
                    current_max_db = np.max(spec_db)
                    trigger_condition = False

                    # Check trigger conditions based on edge mode
                    if self._trigger_edge_mode == self.TRIGGER_EDGE_RISING:
                        trigger_condition = (current_max_db > self._trigger_level and 
                                          self._prev_max_db <= self._trigger_level)
                    elif self._trigger_edge_mode == self.TRIGGER_EDGE_FALLING:
                        trigger_condition = (current_max_db < self._trigger_level and 
                                          self._prev_max_db >= self._trigger_level)
                    elif self._trigger_edge_mode == self.TRIGGER_EDGE_BOTH:
                        trigger_condition = ((current_max_db > self._trigger_level and 
                                           self._prev_max_db <= self._trigger_level) or
                                          (current_max_db < self._trigger_level and 
                                           self._prev_max_db >= self._trigger_level))

                    # Store current max for next edge detection
                    self._prev_max_db = current_max_db

                    # Handle triggering
                    if trigger_condition:
                        # For next-trigger mode, clear previous peak
                        if self._trigger_reset_mode == self.TRIGGER_RESET_NEXT:
                            self._peak_spectrum = None
                        
                        self._triggered = True
                        if self._trigger_reset_mode == self.TRIGGER_RESET_HOLD:
                            self._trigger_samples = int(self._hold_time * self.config.sample_rate / len(data))
                        
                        # Update peak spectrum (initialize or update existing)
                        if self._peak_spectrum is None:
                            self._peak_spectrum = spec_db.copy()
                        else:
                            self._peak_spectrum = np.maximum(self._peak_spectrum, spec_db)
                        
                        logger.info(f"Trigger activated at {current_max_db:.1f} dB")
                    
                    # Update peak spectrum if triggered
                    if self._triggered:
                        if self._peak_spectrum is not None:
                            # Keep maximum values
                            spec_db = np.maximum(spec_db, self._peak_spectrum)
                        
                        # Handle hold time reset mode
                        if self._trigger_reset_mode == self.TRIGGER_RESET_HOLD:
                            self._trigger_samples -= 1
                            if self._trigger_samples <= 0:
                                self._triggered = False
                                self._peak_spectrum = None
                                logger.info("Trigger reset (hold time expired)")

                # Apply normal decay if enabled and not triggered
                elif self._decay_enabled and self._prev_spectrum is not None and len(self._prev_spectrum) == len(spec_db):
                    db_range = self.config.max_db - self.config.min_db
                    decay_amount = self._decay_rate * db_range / 30.0
                    mask = self._prev_spectrum > spec_db
                    decayed = self._prev_spectrum[mask] - decay_amount
                    spec_db[mask] = np.maximum(spec_db[mask], decayed)

                # Store current spectrum for next frame if decay is enabled
                if self._decay_enabled:
                    self._prev_spectrum = spec_db.copy()
                else:
                    self._prev_spectrum = None

            return positive_freqs, spec_db

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return np.array([]), np.array([])

    def add_filter(self, filter_):
        """Add a filter to the processing chain"""
        # If we have a NoiseSource, we should already be sharing its filter list
        # If not, use our own list
        self.filters.append(filter_)
        logger.debug(f"AudioProcessor add_filter - filters: {len(self.filters)}")
        # No need to add to source since we're sharing the list if it's a NoiseSource

    def remove_filter(self, index: int):
        """Remove a filter from the processing chain"""
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
            logger.debug(f"AudioProcessor remove_filter - filters: {len(self.filters)}")
            # No need to remove from source since we're sharing the list if it's a NoiseSource

    def update_filter(self, index: int, params: dict):
        """Update filter parameters"""
        if 0 <= index < len(self.filters):
            params = params.copy()
            filter_type = params.pop('type', None) 
            self.filters[index].update_parameters(params)
            logger.debug(f"AudioProcessor update_filter - filters: {len(self.filters)}")
            # No need to update source since we're sharing the list if it's a NoiseSource

    def close(self):
        """Close the audio source and clean up"""
        if self.source is not None:
            self.source.close()
            self.source = None

    def set_decay_rate(self, rate: float):
        """Set the spectrum decay rate (0.0 to 1.0)"""
        self._decay_rate = np.clip(rate, 0.0, 1.0)
        logger.debug(f"Decay rate set to: {self._decay_rate}")

    def set_decay_enabled(self, enabled: bool):
        """Enable or disable the decay effect"""
        self._decay_enabled = enabled
        if not enabled:
            self._prev_spectrum = None  # Clear previous spectrum when disabled
        logger.info(f"Decay {'enabled' if enabled else 'disabled'}")

    def set_trigger_enabled(self, enabled: bool):
        """Enable or disable the trigger mode"""
        self._trigger_enabled = enabled
        if not enabled:
            self._triggered = False
            self._peak_spectrum = None
        logger.info(f"Trigger {'enabled' if enabled else 'disabled'}")

    def set_trigger_level(self, level: float):
        """Set the trigger threshold level in dB"""
        self._trigger_level = level
        logger.debug(f"Trigger level set to: {level} dB")

    def set_hold_time(self, time: float):
        """Set how long to hold the peak after triggering (in seconds)"""
        self._hold_time = time
        logger.debug(f"Hold time set to: {time} seconds")

    def set_trigger_reset_mode(self, mode: str):
        """Set when to reset the trigger (hold_time, next_trigger, or manual)"""
        if mode in [self.TRIGGER_RESET_HOLD, self.TRIGGER_RESET_NEXT, self.TRIGGER_RESET_MANUAL]:
            self._trigger_reset_mode = mode
            logger.debug(f"Trigger reset mode set to: {mode}")

    def set_trigger_edge_mode(self, mode: str):
        """Set which edges trigger (rising, falling, or both)"""
        if mode in [self.TRIGGER_EDGE_RISING, self.TRIGGER_EDGE_FALLING, self.TRIGGER_EDGE_BOTH]:
            self._trigger_edge_mode = mode
            logger.debug(f"Trigger edge mode set to: {mode}")

    def manual_trigger_reset(self):
        """Manually reset the trigger state"""
        self._triggered = False
        self._peak_spectrum = None
        logger.info("Trigger manually reset")