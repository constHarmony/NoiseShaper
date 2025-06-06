# audio_sources.py
import numpy as np
import sounddevice as sd
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Dict, Any  # Add missing type hints
import queue
import threading
from config import AudioConfig
import time
from filters import AudioNormalizer
import os
from filters import (BandpassFilter, LowpassFilter, HighpassFilter, 
                    NotchFilter, GaussianFilter, ParabolicFilter, PlateauFilter)
import traceback
import logging

logger = logging.getLogger(__name__)

class AudioSource(ABC):
    def __init__(self):
        self.filters = []
        self._running = False
        self._lock = threading.RLock()
        self._last_chunk = None  # Store last generated/captured chunk

    @abstractmethod
    def _generate_chunk(self, frames: int) -> np.ndarray:
        """Generate or capture audio chunk - to be implemented by subclasses"""
        pass

    def read(self) -> np.ndarray:
        """Get data for FFT analysis"""
        if not self._running:
            return np.zeros(self.config.fft_size, dtype=np.float32)
            
        # Use last chunk if available
        if self._last_chunk is not None:
            return self._last_chunk
            
        # Generate new chunk if needed
        return self._generate_chunk(self.config.fft_size)

    def read_analysis(self) -> np.ndarray:
        """Get latest data for analysis"""
        return self.read()  # Use same data as read()

    def add_filter(self, filter_):
        """Add a filter to the audio source"""
        self.filters.append(filter_)

    def remove_filter(self, index: int):
        """Remove a filter at the specified index"""
        if 0 <= index < len(self.filters):
            self.filters.pop(index)

    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply all filters to the data"""
        filtered_data = data.copy()
        for filter_ in self.filters:
            filtered_data = filter_.process(filtered_data)
        return filtered_data

    def update_filter(self, index: int, params: dict):
        """Update filter parameters at the specified index"""
        if 0 <= index < len(self.filters):
            filter_type = params.pop('type', None)  # Remove and ignore type
            self.filters[index].update_parameters(params)

class MonitoredInputSource(AudioSource):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        # Use buffer sizes from config
        self._input_buffer_size = config.input_buffer_size
        self._output_buffer_size = config.output_buffer_size
        
        # Create ring buffer for main data storage
        self._ring_buffer_size = max(8192, self._input_buffer_size * 16)  # Store more data
        self._ring_buffer = np.zeros(self._ring_buffer_size, dtype=np.float32)
        self._write_pos = 0  # Position to write new data
        self._read_pos = 0   # Position to read for FFT
        self._monitor_pos = 0  # Position to read for monitoring
        self._buffer_lock = threading.Lock()
        
        # FFT processing buffers
        self._fft_buffer = queue.Queue(maxsize=32)  # For processed FFT data
        self._raw_data_queue = queue.Queue(maxsize=128)  # For raw data pending FFT
        
        # Legacy buffer for compatibility
        self.monitor_buffer = queue.Queue(maxsize=1)  # Dummy queue for compatibility
        
        self._running = False
        self._lock = threading.RLock()
        self.input_stream = None
        self.output_stream = None
        
        # Start FFT processing thread
        self._fft_thread = threading.Thread(target=self._fft_processor, daemon=True)
        
        self._setup_streams()
        self._running = True
        self._fft_thread.start()

    def _generate_chunk(self, frames: int) -> np.ndarray:
        """Generate chunk from ring buffer - implements abstract method"""
        data, new_pos = self._read_from_ring_buffer(frames, self._read_pos)
        self._read_pos = new_pos
        return data

    def _write_to_ring_buffer(self, data: np.ndarray) -> None:
        """Write data to ring buffer with wraparound"""
        with self._buffer_lock:
            data_len = len(data)
            # First write attempt
            write_len = min(data_len, self._ring_buffer_size - self._write_pos)
            self._ring_buffer[self._write_pos:self._write_pos + write_len] = data[:write_len]
            
            # If we have more data, wrap around
            if write_len < data_len:
                remaining = data_len - write_len
                self._ring_buffer[:remaining] = data[write_len:]
                self._write_pos = remaining
            else:
                self._write_pos = (self._write_pos + write_len) % self._ring_buffer_size

    def _read_from_ring_buffer(self, size: int, pos: int) -> Tuple[np.ndarray, int]:
        """Read data from ring buffer with wraparound"""
        with self._buffer_lock:
            # Calculate available data
            if self._write_pos > pos:
                available = self._write_pos - pos
            else:
                available = self._ring_buffer_size - pos + self._write_pos
            
            if available < size:
                return np.zeros(size, dtype=np.float32), pos
            
            # First read attempt
            read_len = min(size, self._ring_buffer_size - pos)
            data = self._ring_buffer[pos:pos + read_len].copy()
            
            # If we need more data, wrap around
            if read_len < size:
                remaining = size - read_len
                data = np.concatenate((data, self._ring_buffer[:remaining]))
                new_pos = remaining
            else:
                new_pos = (pos + read_len) % self._ring_buffer_size
                
            return data, new_pos

    def _input_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Input callback writing to ring buffer"""
        if status.input_overflow:
            logger.warning("Input overflow detected")
            if self.config.on_overflow:
                self.config.on_overflow()

        if self._running:
            try:
                data = indata.copy().flatten()
                
                # Write to ring buffer
                self._write_to_ring_buffer(data)
                
                # Queue for FFT processing - more aggressive about keeping FFT data
                try:
                    # Clear old data if queue is getting full
                    if self._raw_data_queue.qsize() > self._raw_data_queue.maxsize * 0.8:
                        while self._raw_data_queue.qsize() > self._raw_data_queue.maxsize * 0.2:
                            try:
                                self._raw_data_queue.get_nowait()
                            except queue.Empty:
                                break
                    
                    self._raw_data_queue.put_nowait(data.copy())
                    
                    logger.debug(f"Queued {len(data)} samples for FFT processing")
                    
                except queue.Full:
                    logger.warning("Raw queue full - dropping frame")
                    
            except Exception as e:
                logger.error(f"Input callback error: {e}")
                traceback.print_exc()

    def _fft_processor(self):
        """FFT processing thread - processes raw data into FFT chunks"""
        self._data_buffer = []  # Make it instance variable for size transitions
        last_debug = time.time()
        samples_processed = 0
        last_fft_size = self.config.fft_size
        
        while self._running:
            try:
                # Check if FFT size changed
                if self.config.fft_size != last_fft_size:
                    self.update_fft_size(self.config.fft_size)
                    last_fft_size = self.config.fft_size
                
                # Get raw data with shorter timeout
                try:
                    new_data = self._raw_data_queue.get(timeout=0.005)
                    self._data_buffer.extend(new_data)
                    samples_processed += len(new_data)
                except queue.Empty:
                    if len(self._data_buffer) < self.config.fft_size:
                        time.sleep(0.001)
                        continue
                
                # Process FFT when we have enough data
                while len(self._data_buffer) >= self.config.fft_size:
                    # Take chunk for FFT with overlap
                    fft_data = np.array(self._data_buffer[:self.config.fft_size], dtype=np.float32)
                    overlap = self.config.fft_size // 2
                    self._data_buffer = self._data_buffer[overlap:]
                    
                    # Process FFT data (apply window, etc)
                    try:
                        processed_data = self.apply_filters(fft_data)
                        
                        # Try to maintain buffer at 50% capacity
                        current_size = self._fft_buffer.qsize()
                        target_size = self._fft_buffer.maxsize // 2
                        
                        if current_size < target_size:
                            self._fft_buffer.put_nowait(processed_data)
                        else:
                            try:
                                self._fft_buffer.get_nowait()
                                self._fft_buffer.put_nowait(processed_data)
                            except queue.Empty:
                                self._fft_buffer.put_nowait(processed_data)
                    except ValueError as e:
                        logger.error(f"Processing error during size transition: {e}")
                        self._data_buffer = []  # Clear buffer on error
                        break
                        
                    # Debug info every second
                    now = time.time()
                    if now - last_debug > 1.0:
                        logger.debug("FFT Stats:")
                        logger.debug(f"- Buffer: {self._fft_buffer.qsize()}/{self._fft_buffer.maxsize}")
                        logger.debug(f"- Processing rate: {samples_processed/1.0:.1f} samples/sec")
                        logger.debug(f"- Data buffer: {len(self._data_buffer)} samples")
                        logger.debug(f"- Current FFT size: {self.config.fft_size}")
                        samples_processed = 0
                        last_debug = now
                    
            except Exception as e:
                logger.error(f"FFT processing error: {e}")
                traceback.print_exc()
                time.sleep(0.001)

    def read(self) -> np.ndarray:
        """Read data for FFT analysis"""
        if not self._running:
            return np.zeros(self.config.fft_size, dtype=np.float32)
            
        try:
            # Try to get processed FFT data with longer timeout
            try:
                return self._fft_buffer.get(timeout=0.005)  # Increased timeout
            except queue.Empty:
                pass
                
            # Debug info (~1% of the time)
            if np.random.random() < 0.01:
                with self._buffer_lock:
                    unread = (self._write_pos - self._read_pos) % self._ring_buffer_size
                    logger.debug("Buffer Status:")
                    logger.debug(f"- Ring buffer size: {self._ring_buffer_size}")
                    logger.debug(f"- Write position: {self._write_pos}")
                    logger.debug(f"- Read position: {self._read_pos}")
                    logger.debug(f"- Monitor position: {self._monitor_pos}")
                    logger.debug(f"- Unread samples: {unread}")
                    logger.debug(f"- FFT buffer size: {self._fft_buffer.qsize()}/{self._fft_buffer.maxsize}")
                    logger.debug(f"- Raw queue size: {self._raw_data_queue.qsize()}/{self._raw_data_queue.maxsize}")
                    logger.debug(f"- Processing active: {self._fft_thread.is_alive()}")
            
            # Read from ring buffer as fallback
            data, new_pos = self._read_from_ring_buffer(self.config.fft_size, self._read_pos)
            self._read_pos = new_pos
            
            # Apply filters and store
            self._last_chunk = self.apply_filters(data)
            return self._last_chunk
            
        except Exception as e:
            logger.error(f"Read error: {e}")
            traceback.print_exc()
            return np.zeros(self.config.fft_size, dtype=np.float32)

    def _output_callback(self, outdata: np.ndarray, frames: int,
                         time_info: dict, status: sd.CallbackFlags) -> None:
        """Output callback reading from ring buffer"""
        logger.debug(f"MonitoredInputSource._output_callback: called with frames={frames}, status='{status}', time_info={time_info}")
        if status:
            if status.output_underflow:
                logger.warning("MonitoredInputSource._output_callback: Output underflow detected!")
            if status.output_overflow:
                logger.warning("MonitoredInputSource._output_callback: Output overflow detected!")

        if not self._running or not self.config.monitoring_enabled:
            # logger.debug(f"MonitoredInputSource._output_callback: Not running or monitoring disabled. Filling outdata with zeros.")
            outdata.fill(0)
            return

        try:
            # logger.debug(f"MonitoredInputSource._output_callback: Reading {frames * self.config.channels} samples from ring buffer.")
            # Read from ring buffer
            data, new_pos = self._read_from_ring_buffer(
                frames * self.config.channels, 
                self._monitor_pos
            )
            self._monitor_pos = new_pos
            
            if np.all(data == 0):
                logger.warning("MonitoredInputSource._output_callback: Read only zeros from ring buffer. Potential underflow.")

            # logger.debug(f"MonitoredInputSource._output_callback: Read {len(data)} samples. Reshaping and applying volume.")
            # Apply volume and reshape
            outdata[:] = np.multiply(
                data.reshape(-1, self.config.channels), 
                self.config.monitoring_volume
            )
            # logger.debug(f"MonitoredInputSource._output_callback: Filled outdata with {len(outdata)} samples. Max abs value: {np.max(np.abs(outdata)) if outdata.size > 0 else 0}")
        except Exception as e:
            logger.error(f"Output callback error: {e}\n{traceback.format_exc()}")
            outdata.fill(0)

    def _handle_queue_data(self, data: np.ndarray, queue_obj: queue.Queue):
        """Non-blocking queue handler - drop data if queue is full"""
        try:
            queue_obj.put_nowait(data)
        except queue.Full:
            # Just drop the data if queue is full
            pass

    def _setup_streams(self):
        with self._lock:
            # Clear existing streams first
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()

            self._running = True
            try:
                # Input stream setup with higher latency for stability
                self.input_stream = sd.InputStream(
                    device=self.config.device_input,
                    channels=self.config.channels,
                    samplerate=self.config.sample_rate,
                    blocksize=self._input_buffer_size,
                    dtype=np.float32,
                    callback=self._input_callback,
                    latency='high'  # Changed to high latency
                )
                self.input_stream.start()

                # Output stream setup - match input settings
                if self.config.monitoring_enabled and self.config.device_output is not None:
                    self.output_stream = sd.OutputStream(
                        device=self.config.device_output,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=self._output_buffer_size,  # Use matching size
                        dtype=np.float32,
                        callback=self._output_callback,
                        latency='low'
                    )
                    self.output_stream.start()

            except Exception as e:
                logger.error(f"Stream setup error: {e}")
                self._running = False
                self.close()
                raise

    def update_output_device(self):
        """Update output device settings"""
        with self._lock:
            if self.output_stream is not None:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            
            if self.config.device_output is not None:
                self.output_stream = sd.OutputStream(
                    device=self.config.device_output,
                    channels=self.config.channels,
                    samplerate=self.config.sample_rate,
                    blocksize=512,  # Use smaller blocksize
                    dtype=np.float32,
                    callback=self._output_callback,
                    latency='low'
                )
                self.output_stream.start()

    def update_monitoring(self):
        """Update monitoring state"""
        if self.config.monitoring_enabled and self.config.device_output is not None:
            # Start output stream if it doesn't exist
            if self.output_stream is None:
                try:
                    self.output_stream = sd.OutputStream(
                        device=self.config.device_output,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=512,
                        dtype=np.float32,
                        callback=self._output_callback,
                        latency='low'
                    )
                    self.output_stream.start()
                except Exception as e:
                    logger.error(f"Error starting monitoring: {e}")
        else:
            # Stop and close output stream if it exists
            if self.output_stream is not None:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                finally:
                    self.output_stream = None

    def close(self):
        """Clean shutdown"""
        self._running = False  # Set False before acquiring lock
        
        # Close streams first
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        
        if self.output_stream is not None:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream is not None

        # Clear all queues without locking
        for q in [self.monitor_buffer, self._fft_buffer, self._raw_data_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        # Wait for FFT thread with timeout
        if hasattr(self, '_fft_thread') and self._fft_thread.is_alive():
            self._fft_thread.join(timeout=0.5)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return (self._running and 
                   self.input_stream is not None and 
                   self.input_stream.active)

    def update_fft_size(self, new_size: int):
        """Handle FFT size changes safely"""
        with self._buffer_lock:
            # Clear existing buffers
            while not self._fft_buffer.empty():
                try:
                    self._fft_buffer.get_nowait()
                except queue.Empty:
                    break
                    
            while not self._raw_data_queue.empty():
                try:
                    self._raw_data_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Reset data buffer in FFT processor
            self._data_buffer = []
            
            # Update ring buffer if needed
            min_size = max(8192, new_size * 4)
            if self._ring_buffer_size < min_size:
                self._ring_buffer_size = min_size
                self._ring_buffer = np.zeros(self._ring_buffer_size, dtype=np.float32)
            
            # Reset positions
            self._write_pos = 0
            self._read_pos = 0
            self._monitor_pos = 0
            
            logger.debug("\nFFT size transition:")
            logger.debug(f"- New size: {new_size}")
            logger.debug(f"- Ring buffer size: {self._ring_buffer_size}")
            logger.debug(f"- Buffers cleared")


class NoiseGenerator:
    def __init__(self):
        self._rng = np.random
        self.rng_type = 'uniform'
        self.filters = []
        logger.debug(f"NoiseGenerator init - filters: {len(self.filters)}")
        self.normalize = False
        self.parabolas = []  # For spectral generation
        self.base_amplitude = 1.0  # Add base amplitude
        
    def generate(self, frames: int, sample_rate: int, noise_type: str = 'white') -> np.ndarray:
        """Unified generate function that handles all noise types"""
        # Generate base signal
        if noise_type == 'spectral':
            data = self._generate_spectral(frames, sample_rate)
        else:  # white noise
            data = self._generate_white(frames)
            
            # Currently only apply filters in white noise mode
            # TODO: Future feature - add flag to enable filter application in spectral mode
            if self.filters:
                # logger.debug(f"NoiseGenerator generate - applying {len(self.filters)} filters")
                # Apply filter in frequency domain if any
                spectrum = np.fft.fft(data)
                for filter_ in self.filters:
                    # Get filter mask without amplitude scaling
                    filter_._ensure_filter_size(frames)
                    spectrum *= filter_.filter_mask * filter_.amplitude
                data = np.fft.ifft(spectrum).real
                
        # Apply base amplitude at the source
        return data * self.base_amplitude
        
    def generate_sequence(self, settings: dict, config: AudioConfig) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate a sequence of noise samples with silence intervals"""
        try:
            # Extract settings with defaults
            duration_ms = float(settings.get('carousel_noise_duration_ms', 10.0))
            silence_duration_ms = float(settings.get('silence_duration_ms', 190.0))
            num_samples = int(settings.get('carousel_samples', 20))
            sample_rate = int(settings.get('sample_rate', 44100))
            base_amplitude = float(settings.get('base_amplitude', 1.0))  # Should be 1.0 from UI
            normalize_value = float(settings.get('normalize_value', 0.5))
            enable_normalization = bool(settings.get('enable_normalization', True))
            global_normalization = bool(settings.get('global_normalization', True))
            enable_fade = bool(settings.get('enable_fade', True))
            fade_before_norm = bool(settings.get('fade_before_norm', False))
            fade_in_samples = int(sample_rate * float(settings.get('fade_in_duration', 0.001))) if enable_fade else 0
            fade_out_samples = int(sample_rate * float(settings.get('fade_out_duration', 0.001))) if enable_fade else 0
            fade_in_power = float(settings.get('fade_in_power', 2.0))
            fade_out_power = float(settings.get('fade_out_power', 2.0))
            
            # Calculate durations
            duration = duration_ms / 1000  # Convert to seconds
            silence_samples = int(np.ceil(sample_rate * silence_duration_ms / 1000))
            total_samples = int(duration * sample_rate)
            
            # Set seed if specified
            if not settings.get('use_random_seed', True):
                self.set_seed(settings.get('seed', None))
            
            # Set base amplitude at the source
            self.update_parameters({'amplitude': base_amplitude})
            
            # Generate raw samples first
            raw_samples = []
            for _ in range(num_samples):
                # Generate noise with specified type
                data = self.generate(total_samples, sample_rate, noise_type=settings.get('noise_type', 'white'))
                raw_samples.append(data)

            # Process samples based on fade_before_norm setting
            if fade_before_norm:
                logger.debug("\nDEBUG: Processing order: Fade then Normalize")
                # Apply fade first
                if enable_fade:
                    processed_samples = []
                    for sample in raw_samples:
                        faded = AudioExporter.apply_envelope(
                            sample,
                            fade_in_samples,
                            fade_out_samples,
                            fade_in_power,
                            fade_out_power
                        )
                        processed_samples.append(faded)
                else:
                    processed_samples = raw_samples.copy()

                # Then normalize
                if enable_normalization:
                    if not global_normalization:
                        # Global normalization
                        global_max = max(np.max(np.abs(sample)) for sample in processed_samples)
                        processed_samples = [(sample / global_max) * normalize_value for sample in processed_samples]
                    else:
                        # Per-sample normalization
                        processed_samples = [AudioNormalizer.normalize_signal(sample, normalize_value) for sample in processed_samples]
            else:
                logger.debug("\nDEBUG: Processing order: Normalize then Fade")
                # Normalize first
                if enable_normalization:
                    if not global_normalization:
                        # Global normalization
                        global_max = max(np.max(np.abs(sample)) for sample in raw_samples)
                        processed_samples = [(sample / global_max) * normalize_value for sample in raw_samples]
                    else:
                        # Per-sample normalization
                        processed_samples = [AudioNormalizer.normalize_signal(sample, normalize_value) for sample in raw_samples]
                else:
                    processed_samples = raw_samples.copy()

                # Then apply fade
                if enable_fade:
                    final_samples = []
                    for sample in processed_samples:
                        faded = AudioExporter.apply_envelope(
                            sample,
                            fade_in_samples,
                            fade_out_samples,
                            fade_in_power,
                            fade_out_power
                        )
                        final_samples.append(faded)
                    processed_samples = final_samples
            
            # Apply post-processing attenuation if enabled
            if settings.get('enable_attenuation', False):
                attenuation_db = settings.get('attenuation', 0.0)
                attenuation_factor = 10 ** (-attenuation_db / 20.0)  # Convert dB to linear scale
                processed_samples = [sample * attenuation_factor for sample in processed_samples]
                logger.debug(f"\nDEBUG: Applied {attenuation_db} dB attenuation")
                logger.debug(f"Max amplitude after attenuation: {max(np.max(np.abs(sample)) for sample in processed_samples)}")
            
            # Create silence buffer
            silence = np.zeros(silence_samples)
            
            # Add samples to settings for C++ generation
            settings['individual_samples'] = processed_samples
            settings['silence_samples'] = silence_samples  # No longer doubling for stereo
            
            return silence, processed_samples
            
        except Exception as e:
            logger.error(f"Error in generate_sequence: {str(e)}")
            raise
            
    def _generate_white(self, frames: int) -> np.ndarray:
        """Generate white noise"""
        if self.rng_type == 'uniform':
            return self._rng.uniform(-1.0, 1.0, frames)  # Use np.random directly
        else:  # standard_normal
            return self._rng.standard_normal(frames)  # Use np.random directly
            
    def _generate_spectral(self, frames: int, sample_rate: int) -> np.ndarray:
        """Generate spectral noise"""
        spectrum = self._create_parabola_spectrum(frames, sample_rate)
        if len(self.parabolas) == 0 or np.all(spectrum == 0):
            return np.zeros(frames)
            
        # Apply random phase while preserving conjugate symmetry
        if self.rng_type == 'uniform':
            # Generate random phase for positive frequencies
            midpoint = frames // 2 + 1  # Include DC and Nyquist
            phase = self._rng.uniform(0, 2 * np.pi, midpoint)
            # Make DC and Nyquist phases real (0 or π)
            phase[0] = np.pi * (phase[0] > np.pi)  # DC
            if frames % 2 == 0:  # If even length, handle Nyquist
                phase[-1] = np.pi * (phase[-1] > np.pi)
            # Create full phase array with conjugate symmetry
            full_phase = np.zeros(frames)
            full_phase[:midpoint] = phase
            if frames % 2 == 0:  # Even length
                full_phase[midpoint:] = -phase[1:-1][::-1]  # Exclude Nyquist
            else:  # Odd length
                full_phase[midpoint:] = -phase[1:][::-1]
        else:
            # Use complex Gaussian for phase
            midpoint = frames // 2 + 1
            z = self._rng.standard_normal(midpoint) + 1j * self._rng.standard_normal(midpoint)
            phase = np.angle(z)
            # Make DC and Nyquist phases real
            phase[0] = np.pi * (phase[0] > np.pi)
            if frames % 2 == 0:
                phase[-1] = np.pi * (phase[-1] > np.pi)
            # Create full phase array with conjugate symmetry
            full_phase = np.zeros(frames)
            full_phase[:midpoint] = phase
            if frames % 2 == 0:  # Even length
                full_phase[midpoint:] = -phase[1:-1][::-1]  # Exclude Nyquist
            else:  # Odd length
                full_phase[midpoint:] = -phase[1:][::-1]
            
        random_phase = np.exp(1j * full_phase)
        spectrum *= random_phase
        return np.fft.ifft(spectrum).real

    def _create_parabola_spectrum(self, size: int, sample_rate: int) -> np.ndarray:
        """Create spectrum from parabolas - using full FFT"""
        # Use full FFT frequencies
        frequencies = np.fft.fftfreq(size, 1 / sample_rate)
        spectrum = np.zeros(size, dtype=np.complex128)
        
        for params in self.parabolas:
            if not all(k in params for k in ['center_freq', 'width', 'amplitude']):
                continue
                
            center_freq = params['center_freq']
            width = params['width']
            amplitude = params['amplitude']
            
            # Calculate parabolic shape for positive frequencies
            freq_diff = np.abs(frequencies - center_freq)
            mask = freq_diff <= width
            spectrum[mask] += amplitude * (1 - (freq_diff[mask] / width) ** 2)

        # Apply symmetry for real-valued signal
        midpoint = size // 2
        if size % 2 == 0:  # Even length
            # Ensure Nyquist component is real
            spectrum[midpoint] = np.real(spectrum[midpoint])
            # Mirror the rest with complex conjugate
            spectrum[midpoint+1:] = np.conj(spectrum[1:midpoint][::-1])
        else:  # Odd length
            # Mirror with complex conjugate
            spectrum[midpoint+1:] = np.conj(spectrum[1:midpoint+1][::-1])
        
        # Ensure DC component is real
        spectrum[0] = np.real(spectrum[0])
        
        return spectrum

    def set_seed(self, seed: Optional[int]):
        """Set random seed"""
        if seed is not None:
            self._rng.seed(seed)  # Use np.random.seed directly
            
    def set_rng_type(self, rng_type: str):
        """Set RNG type"""
        self.rng_type = rng_type
        
    def update_parameters(self, params: dict):
        """Update generator parameters"""
        if 'amplitude' in params:
            self.base_amplitude = params['amplitude']  # Update to use base_amplitude
        if 'normalize' in params:
            self.normalize = params['normalize']
            
    def add_filter(self, filter_):
        """Add filter to chain"""
        self.filters.append(filter_)
        logger.debug(f"NoiseGenerator add_filter - filters: {len(self.filters)}")
        
    def remove_filter(self, index: int):
        """Remove filter from chain"""
        if 0 <= index < len(self.filters):
            self.filters.pop(index)
            logger.debug(f"NoiseGenerator remove_filter - filters: {len(self.filters)}")

    def add_parabola(self, params: dict):
        """Add spectral parabola"""
        self.parabolas.append(params.copy())

    def remove_parabola(self, index: int):
        """Remove spectral parabola"""
        if 0 <= index < len(self.parabolas):
            self.parabolas.pop(index)

    def update_parabola(self, index: int, params: dict):
        """Update parabola parameters"""
        if 0 <= index < len(self.parabolas):
            self.parabolas[index].update(params)

class AudioExporter:
    """Handles exporting audio to WAV and C++ code"""
    @staticmethod
    def apply_envelope(signal: np.ndarray, fade_in_samples: int, fade_out_samples: int, 
                      fade_in_power: float = 2.0, fade_out_power: float = 2.0) -> np.ndarray:
        """
        Apply cosine fade envelope to signal with configurable power (default 2.0)
        The envelope shape is: (0.5 * (1 - cos(pi * t))) ^ power
        """
        logger.debug("apply_envelope:")
        logger.debug(f"- Input signal max abs: {np.max(np.abs(signal))}")
        logger.debug(f"- fade_in_samples: {fade_in_samples}")
        logger.debug(f"- fade_out_samples: {fade_out_samples}")
        logger.debug(f"- fade_in_power: {fade_in_power}")
        logger.debug(f"- fade_out_power: {fade_out_power}")
        
        if fade_in_samples <= 0 and fade_out_samples <= 0:
            return signal
            
        # Create fade in envelope
        if fade_in_samples > 0:
            t_in = np.linspace(0, 1, fade_in_samples)
            fade_in = (0.5 * (1 - np.cos(np.pi * t_in))) ** fade_in_power
        else:
            fade_in = np.array([])

        # Create fade out envelope
        if fade_out_samples > 0:
            t_out = np.linspace(0, 1, fade_out_samples)
            fade_out = ((0.5 * (1 - np.cos(np.pi * t_out))) ** fade_out_power)[::-1]
        else:
            fade_out = np.array([])
        
        # Create constant middle section
        constant_len = len(signal) - len(fade_in) - len(fade_out)
        if constant_len < 0:
            raise ValueError("Fade lengths exceed signal length")
        constant = np.ones(constant_len)
        
        # Combine all sections
        envelope = np.concatenate([fade_in, constant, fade_out])
        result = signal * envelope
        
        logger.debug(f"- Output signal max abs: {np.max(np.abs(result))}")
        return result

    @staticmethod
    def export_signal(generator: NoiseGenerator, duration: float, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate and process signal for export"""
        logger.debug("AudioExporter.export_signal called with:")
        logger.debug(f"- duration: {duration}")
        logger.debug(f"- sample_rate: {sample_rate}")
        logger.debug(f"- kwargs: {kwargs}")

        # Remove duplicate parameters if present
        kwargs.pop('duration', None)
        kwargs.pop('sample_rate', None)
        
        # Set RNG type before generating signal
        if 'rng_type' in kwargs:
            generator.set_rng_type(kwargs.pop('rng_type'))
            
        # Set seed if specified
        if not kwargs.get('use_random_seed', True):
            generator.set_seed(kwargs.get('seed', None))
            
        # Set base amplitude at the source
        generator.update_parameters({'amplitude': kwargs.get('amplitude', 1.0)})
        
        total_samples = int(sample_rate * duration)
        fade_in_samples = int(sample_rate * kwargs.get('fade_in_duration', 0.001)) if kwargs.get('enable_fade', True) else 0
        fade_out_samples = int(sample_rate * kwargs.get('fade_out_duration', 0.001)) if kwargs.get('enable_fade', True) else 0
        
        # Validate fade lengths
        if fade_in_samples + fade_out_samples >= total_samples:
            raise ValueError("Total fade duration exceeds signal length")

        # 1. Generate base signal with specified noise type
        noise_type = kwargs.pop('noise_type', 'white')  # Use self.noise_type
        signal = generator.generate(total_samples, sample_rate, noise_type=noise_type)
        logger.debug(f"\nDEBUG: Signal processing steps:")
        logger.debug(f"1. Generated signal max abs: {np.max(np.abs(signal))}")
        
        # Process based on selected order
        if kwargs.get('fade_before_norm', False):
            logger.debug("2. Processing order: Fade then Normalize")
            # "Fade then Normalize" order
            if kwargs.get('enable_fade', True):
                signal = AudioExporter.apply_envelope(
                    signal,
                    fade_in_samples,
                    fade_out_samples,
                    kwargs.get('fade_in_power', 2.0),
                    kwargs.get('fade_out_power', 2.0)
                )
                logger.debug(f"3. After fade max abs: {np.max(np.abs(signal))}")
            
            if kwargs.get('enable_normalization', True):
                signal = AudioNormalizer.normalize_signal(signal, kwargs.get('normalize_value', 1.0))
                logger.debug(f"4. After normalization max abs: {np.max(np.abs(signal))}")
        else:
            logger.debug("2. Processing order: Normalize then Fade")
            # "Normalize then Fade" order
            if kwargs.get('enable_normalization', True):
                signal = AudioNormalizer.normalize_signal(signal, kwargs.get('normalize_value', 1.0))
                logger.debug(f"3. After normalization max abs: {np.max(np.abs(signal))}")
                
            if kwargs.get('enable_fade', True):
                signal = AudioExporter.apply_envelope(
                    signal,
                    fade_in_samples,
                    fade_out_samples,
                    kwargs.get('fade_in_power', 2.0),
                    kwargs.get('fade_out_power', 2.0)
                )
                logger.debug(f"4. After fade max abs: {np.max(np.abs(signal))}")
        
        # Apply post-processing attenuation if enabled
        if kwargs.get('enable_attenuation', False):
            attenuation_db = kwargs.get('attenuation', 0.0)
            attenuation_factor = 10 ** (-attenuation_db / 20.0)  # Convert dB to linear scale
            signal = signal * attenuation_factor
            logger.debug(f"\nDEBUG: Applied {attenuation_db} dB attenuation")
            logger.debug(f"Max amplitude after attenuation: {np.max(np.abs(signal))}")
        
        return signal

    @staticmethod
    def generate_cpp_code(signal: np.ndarray, settings: dict) -> str:
        if settings.get('carousel_enabled', False):
            # Get carousel template from settings
            template = settings.get('carousel_template')
            if not template:
                raise ValueError("No carousel template provided in settings")

            # Get samples and compute silence samples (mono samples)
            samples = settings.get('individual_samples', [])
            if not samples:
                raise ValueError("No samples available for carousel")

            # Use mono silence samples directly
            silence_samples = settings.get('silence_samples', 0)

            # Process template:
            import re
            
            # Create format dict with arithmetic evaluation
            format_dict = {
                'num_buffers': len(samples),
                'samples_per_buffer': len(samples[0]),
                'silence_samples': silence_samples,
                'generator_type': settings.get('source_type', 'Unknown'),
                'buffer_array_name': template['buffer_array_name'],
                'silence_buffer_name': template['silence_buffer_name'],
                'silence_data': '0',
                'buffer_name': template['buffer_name_format'],  # Add buffer_name format
                'data': '',  # Will be replaced per buffer
                'index': 0  # Will be updated per buffer
            }
            
            # Process the main template first to find any iterator templates
            template_text = template['template_text']
            
            # Find any lines containing both @{buffer_name} and @{index}
            # These will be treated as buffer declaration iterators
            lines = template_text.split('\n')
            processed_lines = []
            
            for line in lines:
                if '@{buffer_name}' in line:
                    # This is a buffer iterator line - expand it for all buffers
                    buffer_lines = []
                    for i, sample in enumerate(samples):
                        # Update format dict for this buffer
                        format_dict['index'] = i
                        format_dict['data'] = ', '.join(map(str, np.clip(sample * 32767.0, -32767, 32767).astype(np.int16)))
                        
                        # Process line template
                        processed_line = line
                        
                        # Process all placeholders with arithmetic first
                        arithmetic_pattern = r'@\{([^}]+(?:[+\-*/][^}]+)*)\}'
                        while re.search(arithmetic_pattern, processed_line):
                            match = re.search(arithmetic_pattern, processed_line)
                            if not match:
                                break
                            placeholder = match.group(0)
                            param = match.group(1)
                            try:
                                if any(op in param for op in ['*', '/', '+', '-', '(', ')']):
                                    # Split on operators while keeping them
                                    parts = re.split(r'([+\-*/()])', param)
                                    # Process each part
                                    processed_parts = []
                                    for part in parts:
                                        part = part.strip()
                                        if part in format_dict:
                                            processed_parts.append(str(format_dict[part]))
                                        elif part in ['+', '-', '*', '/', '(', ')']:
                                            processed_parts.append(part)
                                        else:
                                            processed_parts.append(part)
                                    expr = ''.join(processed_parts)
                                    result = eval(expr)
                                    processed_line = processed_line.replace(placeholder, str(result))
                                else:
                                    processed_line = processed_line.replace(placeholder, str(format_dict.get(param, param)))
                            except Exception as e:
                                logger.error(f"Error evaluating arithmetic expression in template: {e}")
                                processed_line = processed_line.replace(placeholder, str(format_dict.get(param, param)))
                        
                        # Then process any remaining simple placeholders
                        for match in re.finditer(r'@\{([^}]+)\}', processed_line):
                            placeholder = match.group(0)
                            param = match.group(1)
                            if param in format_dict:
                                processed_line = processed_line.replace(placeholder, str(format_dict[param]))
                        
                        buffer_lines.append(processed_line)
                    processed_lines.extend(buffer_lines)
                else:
                    processed_lines.append(line)
            
            # Join processed lines back together
            template_text = '\n'.join(processed_lines)
            
            # Prepare buffer list (still needed for array initialization)
            buffer_list_items = []
            for i in range(len(samples)):
                format_dict['index'] = i
                buffer_name = format_dict['buffer_name']
                # Process the buffer name with the same placeholder system
                processed_name = buffer_name
                arithmetic_pattern = r'@\{([^}]+(?:[+\-*/][^}]+)*)\}'
                while re.search(arithmetic_pattern, processed_name):
                    match = re.search(arithmetic_pattern, processed_name)
                    if not match:
                        break
                    placeholder = match.group(0)
                    param = match.group(1)
                    try:
                        if any(op in param for op in ['*', '/', '+', '-', '(', ')']):
                            parts = re.split(r'([+\-*/()])', param)
                            processed_parts = []
                            for part in parts:
                                part = part.strip()
                                if part in format_dict:
                                    processed_parts.append(str(format_dict[part]))
                                elif part in ['+', '-', '*', '/', '(', ')']:
                                    processed_parts.append(part)
                                else:
                                    processed_parts.append(part)
                            expr = ''.join(processed_parts)
                            result = eval(expr)
                            processed_name = processed_name.replace(placeholder, str(result))
                        else:
                            processed_name = processed_name.replace(placeholder, str(format_dict.get(param, param)))
                    except Exception as e:
                        logger.error(f"Error evaluating arithmetic expression in template: {e}")
                        processed_name = processed_name.replace(placeholder, str(format_dict.get(param, param)))
                buffer_list_items.append(processed_name)
            format_dict['buffer_list'] = ', '.join(buffer_list_items)
            
            # Process remaining placeholders
            placeholders = []
            for i, match in enumerate(re.finditer(r'@\{([^}]+)\}', template_text)):
                placeholder = match.group(0)
                param = match.group(1)
                marker = f'<<PLACEHOLDER_{i}>>'
                template_text = template_text.replace(placeholder, marker)
                placeholders.append((marker, param))

            # Process each placeholder
            for marker, param in placeholders:
                try:
                    # Check if param contains arithmetic operations
                    if any(op in param for op in ['*', '/', '+', '-', '(', ')']):
                        # Split param into variable and arithmetic part
                        parts = param.split('*', 1) if '*' in param else param.split('/', 1) if '/' in param else param.split('+', 1) if '+' in param else param.split('-', 1)
                        if len(parts) == 2:
                            var_name = parts[0].strip()
                            if var_name in format_dict:
                                # Evaluate arithmetic expression
                                expr = f"{format_dict[var_name]}{param[len(var_name):]}"
                                result = eval(expr)
                                template_text = template_text.replace(marker, str(result))
                                continue
                    # If no arithmetic or variable not found, use normal formatting
                    template_text = template_text.replace(marker, str(format_dict.get(param, param)))
                except Exception as e:
                    logger.error(f"Error evaluating arithmetic expression in template: {e}")
                    template_text = template_text.replace(marker, str(format_dict.get(param, param)))

            return template_text

        else:
            # Use template from settings
            template = settings.get('cpp_template')
            if not template:
                raise ValueError("No C++ template provided in settings")
            
            # Generate code using template
            signal = np.clip(signal, -1.0, 1.0)
            int16_data = (signal * 32767.0).astype(np.int16)
            array_data = ", ".join(map(str, int16_data))
            
            # Process template:
            template_text = template['template_text']
            placeholders = []
            import re
            
            # Find and store all @{...} patterns
            for i, match in enumerate(re.finditer(r'@\{([^}]+)\}', template_text)):
                placeholder = match.group(0)
                param = match.group(1)
                marker = f'<<PLACEHOLDER_{i}>>'
                template_text = template_text.replace(placeholder, marker)
                placeholders.append((marker, param))
            
            # Create format dict with arithmetic evaluation
            format_dict = {
                'var_name': template.get('var_name', 'audioData'),
                'length_name': template.get('length_name', 'AUDIO_LENGTH'),
                'length': len(int16_data),
                'array_data': array_data
            }
            
            # Process each placeholder
            for marker, param in placeholders:
                try:
                    # Check if param contains arithmetic operations
                    if any(op in param for op in ['*', '/', '+', '-', '(', ')']):
                        # Split param into variable and arithmetic part
                        parts = param.split('*', 1) if '*' in param else param.split('/', 1) if '/' in param else param.split('+', 1) if '+' in param else param.split('-', 1)
                        if len(parts) == 2:
                            var_name = parts[0].strip()
                            if var_name in format_dict:
                                # Evaluate arithmetic expression
                                expr = f"{format_dict[var_name]}{param[len(var_name):]}"
                                result = eval(expr)
                                template_text = template_text.replace(marker, str(result))
                                continue
                    # If no arithmetic or variable not found, use normal formatting
                    template_text = template_text.replace(marker, str(format_dict.get(param, param)))
                except Exception as e:
                    logger.error(f"Error evaluating arithmetic expression in template: {e}")
                    template_text = template_text.replace(marker, str(format_dict.get(param, param)))
            
            return template_text

class NoiseSource(AudioSource):
    def __init__(self, config: AudioConfig, noise_type: str = 'white'):
        super().__init__()
        self.config = config
        self.noise_type = noise_type
        
        # Use appropriate buffer sizes
        self._buffer_size = config.spectral_size if noise_type == 'spectral' else config.buffer_size
        
        # Create unified generator and share filter list
        self.generator = NoiseGenerator()
        # Share filter list - generator will use our list
        self.generator.filters = self.filters  # Direct reference to AudioSource's filter list
        logger.debug(f"DEBUG: NoiseSource init - filters: {len(self.filters)}, generator filters: {len(self.generator.filters)}")
        
        # Audio device handling
        self._running = False
        self._lock = threading.RLock()
        self.stream = None
        
        # Initialize synthesis buffer
        self._synthesis_buffer = np.array([], dtype=np.float32)
        
        # Start if output enabled
        if config.output_device_enabled:
            self._setup_stream()
        else:
            self._running = True
        # Add this to propagate RNG type to export
        self.rng_type = 'standard_normal'  # Default to standard normal
        
        # Keep track of parabolas locally
        self._parabolas = []

    def _generate_chunk(self, frames: int) -> np.ndarray:
        """Generate noise chunk with proper buffering"""
        # logger.debug(f"NoiseSource._generate_chunk: called with frames={frames}")
        if frames <= 0:
            logger.warning(f"NoiseSource._generate_chunk: requested frames is {frames}, returning empty array.")
            return np.array([], dtype=np.float32)
            
        if self.noise_type == 'spectral':
            # logger.debug(f"NoiseSource._generate_chunk (spectral): current _synthesis_buffer len={len(self._synthesis_buffer)}")
            # Fill synthesis buffer until we have enough frames
            while len(self._synthesis_buffer) < frames:
                # Always generate using spectral_size for consistency
                # logger.debug(f"NoiseSource._generate_chunk (spectral): _synthesis_buffer needs more data (has {len(self._synthesis_buffer)}, needs {frames}). Generating {self.config.spectral_size} samples.")
                data = self.generator.generate(self.config.spectral_size, self.config.sample_rate, noise_type='spectral')
                # logger.debug(f"NoiseSource._generate_chunk (spectral): generator.generate returned {len(data)} samples.")
                if data.size > 0:
                    # Apply amplitude scaling based on mode
                    amplitude = self.config.amp_spectral if self.noise_type == 'spectral' else self.config.amp_whitenoise
                    data = data * amplitude
                    self._synthesis_buffer = np.concatenate((self._synthesis_buffer, data))
                    # logger.debug(f"NoiseSource._generate_chunk (spectral): _synthesis_buffer len after append: {len(self._synthesis_buffer)}")
                else:
                    logger.warning("NoiseSource._generate_chunk (spectral): generator.generate returned 0 samples. Breaking fill loop.")
                    break
                    
            if len(self._synthesis_buffer) >= frames:
                # Extract required frames
                output_data = self._synthesis_buffer[:frames]
                # Update buffer
                self._synthesis_buffer = self._synthesis_buffer[frames:]
                self._last_chunk = output_data
                # logger.debug(f"NoiseSource._generate_chunk (spectral): returning {len(output_data)} samples. _synthesis_buffer remaining: {len(self._synthesis_buffer)}")
                return output_data
            else:
                logger.warning(f"NoiseSource._generate_chunk (spectral): not enough data in _synthesis_buffer (has {len(self._synthesis_buffer)}, needs {frames}). Returning zeros.")
                return np.zeros(frames, dtype=np.float32)
        else:
            # White noise can be generated at exact frame size
            # logger.debug(f"NoiseSource._generate_chunk (whitenoise): generating {frames} samples.")
            data = self.generator.generate(frames, self.config.sample_rate, noise_type=self.noise_type)  # Use self.noise_type
            # logger.debug(f"NoiseSource._generate_chunk (whitenoise): generator.generate returned {len(data)} samples.")
            # Apply amplitude scaling based on mode
            amplitude = self.config.amp_spectral if self.noise_type == 'spectral' else self.config.amp_whitenoise
            data = data * amplitude
            self._last_chunk = data
            # logger.debug(f"NoiseSource._generate_chunk (whitenoise): returning {len(data)} samples.")
            return data

    def read(self) -> np.ndarray:
        """Get data for analysis"""
        if not self._running:
            return np.zeros(self.config.fft_size, dtype=np.float32)
            
        # Generate new chunk for analysis
        data = self._generate_chunk(self.config.fft_size)
        self._last_chunk = data  # Store for potential audio output
        return data

    def read_analysis(self) -> np.ndarray:
        """Get latest generated data for analysis"""
        if self._last_chunk is not None:
            return self._last_chunk
        return self.read()

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Audio output callback - only active if device is enabled"""
        logger.debug(f"NoiseSource._audio_callback: called with frames={frames}, status={status}, time_info={time_info}")
        if status:
            if status.output_underflow:
                logger.warning("NoiseSource._audio_callback: Output underflow detected!")
            if status.output_overflow:
                logger.warning("NoiseSource._audio_callback: Output overflow detected!")
            if status.input_underflow: # Though less likely for an output callback, log if present
                logger.warning("NoiseSource._audio_callback: Input underflow detected on output stream status!")
            if status.input_overflow: # Though less likely for an output callback, log if present
                logger.warning("NoiseSource._audio_callback: Input overflow detected on output stream status!")
            if status.priming_output: # Log if output is being primed
                 logger.info("NoiseSource._audio_callback: Output stream is priming.")


        if not self._running:
            # logger.debug("NoiseSource._audio_callback: not running, filling outdata with zeros.")
            outdata.fill(0)
            return

        try:
            # Generate fresh chunk for audio
            # logger.debug(f"NoiseSource._audio_callback: calling _generate_chunk for {frames} frames.")
            audio_data = self._generate_chunk(frames)
            # logger.debug(f"NoiseSource._audio_callback: _generate_chunk returned {len(audio_data)} frames.")
            self._last_chunk = audio_data  # Store for visualization
            
            if self.config.monitoring_enabled:
                # Clip the audio data to prevent overflow
                scaled_data = audio_data.reshape(-1, self.config.channels) * self.config.monitoring_volume
                np.clip(scaled_data, -1.0, 1.0, out=outdata)
                # logger.debug(f"NoiseSource._audio_callback: monitoring enabled, filled outdata with {len(outdata)} samples. Max abs value: {np.max(np.abs(outdata)) if outdata.size > 0 else 0}")
            else:
                # logger.debug("NoiseSource._audio_callback: monitoring disabled, filling outdata with zeros.")
                outdata.fill(0)

        except Exception as e:
            logger.error(f"Audio callback error: {e}\n{traceback.format_exc()}")
            outdata.fill(0)
            if self.config.on_underflow:
                self.config.on_underflow()

    def _setup_stream(self):
        with self._lock:
            self._running = True
            self._audio_buffer = np.array([], dtype=np.float32)
            
            if (self.config.device_output is not None and 
                self.config.output_device_enabled):
                try:
                    # Pre-fill buffer with validation
                    test_data = self._generate_chunk(self._buffer_size)
                    if np.any(np.isnan(test_data)) or np.any(np.isinf(test_data)):
                        raise ValueError("Invalid audio data generated during setup")
                    self._audio_buffer = test_data
                    
                    safe_blocksize = self.config.output_buffer_size
                    
                    logger.debug(f"Using blocksize: {safe_blocksize}")
                    
                    self.stream = sd.OutputStream(
                        device=self.config.device_output,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=safe_blocksize,
                        dtype=np.float32,
                        callback=self._audio_callback,
                        latency='high'  # Use high latency for stability
                    )
                    self.stream.start()
                    
                except Exception as e:
                    logger.error(f"Error opening audio output: {e}")
                    self.stream = None
                    # Re-raise the error to propagate it to the UI
                    raise

    def update_output_device(self):
        """Update output device settings"""
        with self._lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            if self.config.device_output is not None:
                self.stream = sd.OutputStream(
                    device=self.config.device_output,
                    channels=self.config.channels,
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.output_buffer_size,  # Fix: use output_buffer_size
                    dtype=np.float32,
                    callback=self._audio_callback
                )
                self.stream.start()

    def update_monitoring(self):
        """Update monitoring state based on config - just update internal state"""
        # No need to start/stop stream, just let the callback handle it
        pass

    def close(self):
        """Clean up resources"""
        self._running = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._last_chunk = None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running and (self.stream is None or self.stream.active)

    def add_filter(self, filter_):
        """Add a filter to shared filter list"""
        super().add_filter(filter_)  # This adds to self.filters
        logger.debug(f"DEBUG: NoiseSource add_filter - filters: {len(self.filters)}, generator filters: {len(self.generator.filters)}")
        # No need to update generator's filter list since it's using the same list

    def remove_filter(self, index: int):
        """Remove filter from shared filter list"""
        super().remove_filter(index)  # This removes from self.filters
        logger.debug(f"DEBUG: NoiseSource remove_filter - filters: {len(self.filters)}, generator filters: {len(self.generator.filters)}")
        # No need to update generator's filter list since it's using the same list

    def update_filter(self, index: int, params: dict):
        """Update filter in shared filter list"""
        super().update_filter(index, params)  # This updates in self.filters
        # No need to update generator's filter list since it's using the same list

    def add_parabola(self, params: dict):
        """Add a new parabola component"""
        self._parabolas.append(params.copy())  # Store locally
        self.generator.parabolas = self._parabolas  # Update generator

    def remove_parabola(self, index: int):
        """Remove a parabola component"""
        if 0 <= index < len(self._parabolas):
            self._parabolas.pop(index)
            self.generator.parabolas = self._parabolas  # Update generator

    def update_parabola(self, index: int, params: dict):
        """Update parabola parameters"""
        if 0 <= index < len(self._parabolas):
            self._parabolas[index] = params.copy()
            self.generator.parabolas = self._parabolas  # Update generator

    def export_sequence(self, settings: dict) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Export a sequence of noise samples with silence intervals"""
        # Add noise type to settings
        settings['noise_type'] = self.noise_type
        logger.debug(f"\nDEBUG: NoiseSource.export_sequence:")
        logger.debug(f"- noise_type: {self.noise_type}")
        return self.generator.generate_sequence(settings, self.config)

    def export_signal(self, duration: float, sample_rate: int, amplitude: float,
                     enable_fade: bool = True, fade_in_duration: float = 0.001,
                     fade_out_duration: float = 0.001, 
                     fade_in_power: float = 0.5, fade_out_power: float = 0.5, 
                     enable_normalization: bool = True,
                     normalize_value: float = 1.0,
                     fade_before_norm: bool = False,
                     rng_type: str = 'standard_normal',
                     use_random_seed: bool = True,
                     seed: Optional[int] = None,
                     enable_attenuation: bool = False,
                     attenuation: float = 0.0) -> np.ndarray:
        """Export signal with specified parameters"""
        logger.debug("\nDEBUG: NoiseSource.export_signal called with:")
        logger.debug(f"- duration: {duration}")
        logger.debug(f"- sample_rate: {sample_rate}")
        logger.debug(f"- amplitude: {amplitude}")
        logger.debug(f"- enable_fade: {enable_fade}")
        logger.debug(f"- fade_in_duration: {fade_in_duration}")
        logger.debug(f"- fade_out_duration: {fade_out_duration}")
        logger.debug(f"- fade_in_power: {fade_in_power}")
        logger.debug(f"- fade_out_power: {fade_out_power}")
        logger.debug(f"- enable_normalization: {enable_normalization}")
        logger.debug(f"- normalize_value: {normalize_value}")
        logger.debug(f"- fade_before_norm: {fade_before_norm}")
        logger.debug(f"- rng_type: {rng_type}")
        logger.debug(f"- use_random_seed: {use_random_seed}")
        logger.debug(f"- seed: {seed}")
        logger.debug(f"- enable_attenuation: {enable_attenuation}")
        logger.debug(f"- attenuation: {attenuation}")
        logger.debug(f"- noise_type: {self.noise_type}")
        logger.debug(f"- num_parabolas: {len(self._parabolas)}")
        
        kwargs = {
            'amplitude': amplitude,
            'enable_fade': enable_fade,
            'fade_in_duration': fade_in_duration,
            'fade_out_duration': fade_out_duration,
            'fade_in_power': fade_in_power,
            'fade_out_power': fade_out_power,
            'enable_normalization': enable_normalization,
            'normalize_value': normalize_value,
            'fade_before_norm': fade_before_norm,
            'rng_type': rng_type,
            'use_random_seed': use_random_seed,
            'seed': seed,
            'enable_attenuation': enable_attenuation,
            'attenuation': attenuation,
            'noise_type': self.noise_type  # Add noise type to kwargs
        }
        
        # Share filter list with generator
        self.generator.filters = self.filters
        
        # If in spectral mode, make sure generator has the parabola components
        if self.noise_type == 'spectral':
            # Use our stored parabolas
            self.generator.parabolas = self._parabolas
            
        return AudioExporter.export_signal(self.generator, duration, sample_rate, **kwargs)

    def get_individual_samples(self, sequence: np.ndarray, settings: dict) -> List[np.ndarray]:
        """Extract individual samples from sequence"""
        sample_rate = settings.get('sample_rate', 44100)
        noise_samples = int(sample_rate * settings.get('noise_duration_ms', 10.0) / 1000)
        silence_samples = int(sample_rate * settings.get('silence_duration_ms', 190.0) / 1000)
        total_samples = noise_samples + silence_samples
        samples = []
        offset = 0
        while offset + noise_samples <= len(sequence):
            samples.append(sequence[offset:offset + noise_samples])
            offset += total_samples
        return samples

    def set_spectral_normalization(self, enabled: bool):
        """Set whether spectral synthesis should normalize output"""
        self.generator.update_parameters({'normalize': enabled})

    def set_filter_normalization(self, enabled: bool):
        """Set whether white noise filtering should normalize output"""
        self.generator.update_parameters({'normalize': enabled})

    def set_rng_type(self, rng_type: str):
        """Set the RNG distribution type"""
        self.rng_type = rng_type
        self.generator.set_rng_type(rng_type)