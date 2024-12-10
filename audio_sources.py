# audio_sources.py
import numpy as np
import sounddevice as sd
from abc import ABC, abstractmethod
from typing import Optional
import queue
import threading
from config import AudioConfig
import time  # Add this import at the top
from filters import AudioNormalizer  # Add this import
import os  # Add this import at the top


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
        self._synthesis_buffer = np.array([], dtype=np.float32)
        
        # Increase queue sizes significantly to handle larger buffers
        max_queue = max(1024, self._input_buffer_size * 4)  # Increased multiplier
        self.monitor_buffer = queue.Queue(maxsize=max_queue)
        self.fft_buffer = queue.Queue(maxsize=16)  # Increased size
        self._raw_data_queue = queue.Queue(maxsize=64)  # Increased size
        
        self._running = False  # Must be False before starting thread
        self._lock = threading.RLock()  # Use RLock instead of Lock
        self.input_stream = None
        self.output_stream = None
        
        # Start FFT processing thread
        self._fft_thread = threading.Thread(target=self._fft_processor, daemon=True)
        
        self._setup_streams()
        self._running = True  # Set True after initialization
        self._fft_thread.start()

    def _generate_chunk(self, frames: int) -> np.ndarray:
        """Optimized chunk generation"""
        try:
            # Pre-allocate output array
            output_data = np.zeros(frames, dtype=np.float32)
            samples_needed = frames
            offset = 0
            
            # Get data from synthesis buffer first
            if len(self._synthesis_buffer) > 0:
                samples_to_copy = min(len(self._synthesis_buffer), samples_needed)
                output_data[:samples_to_copy] = self._synthesis_buffer[:samples_to_copy]
                self._synthesis_buffer = self._synthesis_buffer[samples_to_copy:]
                samples_needed -= samples_to_copy
                offset = samples_to_copy

            # Fill remaining samples from monitor buffer
            while samples_needed > 0:
                try:
                    data = self.monitor_buffer.get_nowait()
                    samples_to_copy = min(len(data), samples_needed)
                    output_data[offset:offset + samples_to_copy] = data[:samples_to_copy]
                    
                    # Store any remaining data in synthesis buffer
                    if samples_to_copy < len(data):
                        self._synthesis_buffer = np.concatenate((
                            self._synthesis_buffer, 
                            data[samples_to_copy:]
                        ))
                    
                    offset += samples_to_copy
                    samples_needed -= samples_to_copy
                except queue.Empty:
                    break

            self._last_chunk = output_data
            return output_data

        except Exception as e:
            print(f"Generate chunk error: {e}")
            return np.zeros(frames, dtype=np.float32)

    def read(self) -> np.ndarray:
        """Read data for FFT analysis with consistent buffering"""
        if not self._running:
            return np.zeros(self.config.fft_size, dtype=np.float32)
            
        try:
            # Always try to get fresh data first
            data = self.monitor_buffer.get_nowait()
            
            # Add to FFT buffer
            try:
                self.fft_buffer.put_nowait(data)
            except queue.Full:
                # If full, remove oldest data
                try:
                    self.fft_buffer.get_nowait()
                    self.fft_buffer.put_nowait(data)
                except queue.Empty:
                    pass

            # Use the latest data
            self._last_chunk = data
            return self.apply_filters(data)
            
        except queue.Empty:
            # If no new data, use last chunk if available
            if self._last_chunk is not None:
                return self.apply_filters(self._last_chunk)
            return np.zeros(self.config.fft_size, dtype=np.float32)
        
    def read_analysis(self) -> np.ndarray:
        """Get data for FFT analysis"""
        try:
            # Get incoming audio data
            data = self.monitor_buffer.get_nowait()
            
            # Copy to FFT buffer
            try:
                self.fft_buffer.put_nowait(data)
            except queue.Full:
                pass
                
            # Return data for analysis
            return data
        except queue.Empty:
            return np.zeros(self.config.fft_size, dtype=np.float32)

    def _fft_processor(self):
        """FFT processing thread - completely separate from audio path"""
        data_buffer = []
        
        while self._running:
            try:
                # Get raw data
                new_data = self._raw_data_queue.get(timeout=0.1)
                data_buffer.extend(new_data)
                
                # Process FFT when we have enough data
                while len(data_buffer) >= self.config.fft_size:
                    # Take chunk for FFT
                    fft_data = np.array(data_buffer[:self.config.fft_size], dtype=np.float32)
                    data_buffer = data_buffer[self.config.fft_size:]  # Keep remainder
                    
                    try:
                        self.fft_buffer.put_nowait(fft_data)
                    except queue.Full:
                        break  # Skip if buffer is full

                # Keep buffer size under control
                if len(data_buffer) > self.config.fft_size * 2:
                    data_buffer = data_buffer[-self.config.fft_size:]
                    
            except queue.Empty:
                continue  # No data available
            except Exception as e:
                print(f"FFT processing error: {e}")
                time.sleep(0.1)

    def _handle_queue_data(self, data: np.ndarray, queue_obj: queue.Queue):
        """Non-blocking queue handler - drop data if queue is full"""
        try:
            queue_obj.put_nowait(data)
        except queue.Full:
            # Just drop the data if queue is full
            pass

    def _input_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Optimized input callback with better buffer management"""
        if status and status.input_overflow:
            if self.config.on_overflow:
                self.config.on_overflow()

        if self._running:
            try:
                data = indata.copy().flatten()
                
                # Drop older data if monitor buffer is too full
                if self.monitor_buffer.qsize() > self.monitor_buffer.maxsize * 0.8:
                    try:
                        while self.monitor_buffer.qsize() > self.monitor_buffer.maxsize * 0.5:
                            self.monitor_buffer.get_nowait()
                    except queue.Empty:
                        pass

                # Always try to add to monitor buffer
                try:
                    self.monitor_buffer.put_nowait(data)
                except queue.Full:
                    # Clear half the queue if full
                    try:
                        for _ in range(self.monitor_buffer.maxsize // 2):
                            self.monitor_buffer.get_nowait()
                        self.monitor_buffer.put_nowait(data)
                    except (queue.Empty, queue.Full):
                        pass

                # Directly add to raw data queue for FFT processing
                try:
                    self._raw_data_queue.put_nowait(data.copy())
                except queue.Full:
                    # Clear old data if full
                    try:
                        self._raw_data_queue.get_nowait()
                        self._raw_data_queue.put_nowait(data.copy())
                    except queue.Empty:
                        pass

            except Exception as e:
                print(f"Input callback error: {e}")

    def _output_callback(self, outdata: np.ndarray, frames: int,
                         time_info: dict, status: sd.CallbackFlags) -> None:
        if status.output_underflow:
            # Signal underflow through config callback
            if hasattr(self.config, 'on_underflow') and self.config.on_underflow:
                self.config.on_underflow()
        if status:
            print(f'Output callback status: {status}')

        if not self._running:
            outdata.fill(0)
            return

        # Check if monitoring is enabled before attempting to get data
        if self.config.monitoring_enabled:
            try:
                data = self.monitor_buffer.get_nowait()
            except queue.Empty:
                data = np.zeros(frames * self.config.channels, dtype=np.float32)
            except Exception as e:
                print(f"Output callback error: {e}")
                data = np.zeros(frames * self.config.channels, dtype=np.float32)

            # Ensure data has the correct shape
            if data.size < frames * self.config.channels:
                # Pad data if it's too short
                data = np.pad(data, (0, frames * self.config.channels - data.size), mode='constant')
            elif data.size > frames * self.config.channels:
                # Trim data if it's too long
                data = data[:frames * self.config.channels]

            outdata[:] = np.multiply(data.reshape(-1, self.config.channels), self.config.monitoring_volume)
        else:
            outdata.fill(0)

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
                    device=self.config.device_input_index,
                    channels=self.config.channels,
                    samplerate=self.config.sample_rate,
                    blocksize=self._input_buffer_size,
                    dtype=np.float32,
                    callback=self._input_callback,
                    latency='high'  # Changed to high latency
                )
                self.input_stream.start()

                # Output stream setup - match input settings
                if self.config.monitoring_enabled and self.config.device_output_index is not None:
                    self.output_stream = sd.OutputStream(
                        device=self.config.device_output_index,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=self._output_buffer_size,  # Use matching size
                        dtype=np.float32,
                        callback=self._output_callback,
                        latency='low'
                    )
                    self.output_stream.start()

            except Exception as e:
                print(f"Stream setup error: {e}")
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
            
            if self.config.device_output_index is not None:
                self.output_stream = sd.OutputStream(
                    device=self.config.device_output_index,
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
        if self.config.monitoring_enabled and self.config.device_output_index is not None:
            # Start output stream if it doesn't exist
            if self.output_stream is None:
                try:
                    self.output_stream = sd.OutputStream(
                        device=self.config.device_output_index,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=512,
                        dtype=np.float32,
                        callback=self._output_callback,
                        latency='low'
                    )
                    self.output_stream.start()
                except Exception as e:
                    print(f"Error starting monitoring: {e}")
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
            self.output_stream = None

        # Clear all queues without locking
        for q in [self.monitor_buffer, self.fft_buffer, self._raw_data_queue]:
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


class NoiseGenerator:
    def __init__(self):
        self._rng = np.random.default_rng()
        self.rng_type = 'standard_normal'
        self.amplitude = 0.5  # This is the base generator amplitude (matches DIY)

    @abstractmethod
    def generate(self, frames: int, sample_rate: int) -> np.ndarray:
        pass

    @abstractmethod
    def update_parameters(self, params: dict):
        pass

    @abstractmethod
    def set_seed(self, seed: Optional[int]):
        """Set the random seed for noise generation"""
        pass

    def set_rng_type(self, rng_type: str):
        """Set the RNG distribution type ('standard_normal' or 'uniform')"""
        self.rng_type = rng_type

class WhiteNoiseGenerator(NoiseGenerator):
    def __init__(self):
        super().__init__()  # Get RNG from parent
        self.filters = []
        self.amplitude = 0.5  # Match DIY example default amplitude
        self.rng_type = 'standard_normal'  # Default to standard normal
        
    def generate(self, frames: int, sample_rate: int) -> np.ndarray:
        # Generate base noise using selected distribution
        if self.rng_type == 'uniform':
            data = self._rng.uniform(-1.0, 1.0, frames).astype(np.float32)
        else:  # standard_normal
            data = self._rng.standard_normal(frames).astype(np.float32)
            # Clip to reasonable range for normal distribution
            data = np.clip(data, -3.0, 3.0) / 3.0

        # Apply filters
        for filter_ in self.filters:
            data = filter_.process(data)
            
        # Apply amplitude scaling - no normalization in edit mode
        data *= self.amplitude
            
        return data

    def update_parameters(self, params: dict):
        if 'amplitude' in params:
            self.amplitude = params['amplitude']

    def add_filter(self, filter_):
        self.filters.append(filter_)

    def remove_filter(self, index: int):
        if 0 <= index < len(self.filters):
            self.filters.pop(index)

    def update_filter(self, index: int, params: dict):
        if 0 <= index < len(self.filters):
            self.filters[index].update_parameters(params)

    def set_seed(self, seed: Optional[int]):
        """Set or reset RNG seed"""
        self._rng = np.random.default_rng(seed)

class SpectralNoiseGenerator(NoiseGenerator):
    def __init__(self):
        super().__init__()  # Get RNG from parent
        self.parabolas = []
        self.amplitude = 0.5  # Match DIY example default amplitude
        self.rng_type = 'standard_normal'  # Default to standard normal
        
    def generate(self, frames: int, sample_rate: int) -> np.ndarray:
        # Generate spectrum
        spectrum = self._create_parabola_spectrum(frames, sample_rate)
        if len(self.parabolas) == 0 or np.all(spectrum == 0):
            # Return silence instead of raising error
            return np.zeros(frames, dtype=np.float32)
            
        # Apply random phase using selected distribution
        if self.rng_type == 'uniform':
            phase = self._rng.uniform(0, 2 * np.pi, len(spectrum))
        else:  # standard_normal
            phase = np.angle(self._rng.standard_normal(len(spectrum)) + 
                           1j * self._rng.standard_normal(len(spectrum)))
        random_phase = np.exp(1j * phase)
        
        spectrum *= random_phase
        
        # Convert to time domain
        data = np.fft.irfft(spectrum).astype(np.float32)
        
        # Apply amplitude scaling - no normalization in edit mode
        data *= self.amplitude
            
        return data

    def update_parameters(self, params: dict):
        if 'amplitude' in params:
            self.amplitude = params['amplitude']

    def _create_parabola_spectrum(self, size: int, sample_rate: int) -> np.ndarray:
        frequencies = np.fft.rfftfreq(size, 1 / sample_rate)
        spectrum = np.zeros(len(frequencies), dtype=np.complex128)
        
        for params in self.parabolas:
            if not all(k in params for k in ['center_freq', 'width', 'amplitude']):
                continue
                
            center_freq = params['center_freq']
            width = params['width']
            amplitude = params['amplitude']
            
            freq_diff = np.abs(frequencies - center_freq)
            mask = freq_diff <= width
            spectrum[mask] += amplitude * (1 - (freq_diff[mask] / width) ** 2)

        return spectrum

    def add_parabola(self, params: dict):
        self.parabolas.append(params.copy())

    def remove_parabola(self, index: int):
        if 0 <= index < len(self.parabolas):
            self.parabolas.pop(index)

    def update_parabola(self, index: int, params: dict):
        if 0 <= index < len(self.parabolas):
            self.parabolas[index].update(params)

    def set_seed(self, seed: Optional[int]):
        """Set or reset RNG seed"""
        self._rng = np.random.default_rng(seed)

class AudioExporter:
    """Handles exporting audio to WAV and C++ code"""
    @staticmethod
    def apply_envelope(signal: np.ndarray, fade_in_samples: int, fade_out_samples: int, 
                      fade_in_power: float = 2.0, fade_out_power: float = 2.0) -> np.ndarray:
        """
        Apply cosine fade envelope to signal with configurable power (default 2.0 to match DIY example)
        The envelope shape is: (0.5 * (1 - cos(pi * t))) ^ power
        """
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
        return signal * envelope

    @staticmethod
    def export_signal(generator: NoiseGenerator, duration: float, sample_rate: int, **kwargs) -> np.ndarray:
        """Generate and process signal for export"""
        total_samples = int(sample_rate * duration)
        fade_in_samples = int(sample_rate * kwargs.get('fade_in_duration', 0.001)) if kwargs.get('enable_fade', True) else 0
        fade_out_samples = int(sample_rate * kwargs.get('fade_out_duration', 0.001)) if kwargs.get('enable_fade', True) else 0
        
        # Ensure even number of samples
        if fade_in_samples % 2 != 0:
            fade_in_samples -= 1
        if fade_out_samples % 2 != 0:
            fade_out_samples -= 1
            
        # Validate fade lengths
        if fade_in_samples + fade_out_samples >= total_samples:
            raise ValueError("Total fade duration exceeds signal length")

        # Generate base signal (will already have generator.amplitude applied)
        signal = generator.generate(total_samples, sample_rate)
        
        # Don't raise error for all-zero signals anymore
        if np.any(np.isnan(signal)):
            raise ValueError("Generated signal contains invalid values")

        # Apply processing chain
        if kwargs.get('enable_normalization', True):
            signal = AudioNormalizer.normalize_signal(signal, kwargs.get('normalize_value', 1.0))

        if kwargs.get('enable_fade', True) and (fade_in_samples > 0 or fade_out_samples > 0):
            signal = AudioExporter.apply_envelope(signal, fade_in_samples, fade_out_samples,
                                               kwargs.get('fade_in_power', 0.5), kwargs.get('fade_out_power', 0.5))
        
        # Final output amplitude scaling
        return signal * kwargs.get('amplitude', 1.0)

    @staticmethod
    def generate_cpp_code(signal: np.ndarray, settings: dict) -> str:
        """Generate C++ code with template and audio data"""
        # Convert signal to int16
        signal = np.clip(signal, -1.0, 1.0)
        audio_data = (signal * 32767.0).astype(np.int16)
        length = len(audio_data)

        # Format audio data array (10 items per line)
        data_lines = []
        for i in range(0, length, 10):
            line_data = audio_data[i:i+10]
            data_lines.append("    " + ", ".join(map(str, line_data)))

        # Get template settings and create format dictionary
        template = settings.get('cpp_template', {})
        template_text = template.get('template_text', '')
        
        # Use only the standard variables
        format_dict = {
            'length': length,
            'var_name': template.get('var_name', 'audioData'),
            'length_name': template.get('length_name', 'AUDIO_LENGTH'),
            'array_data': ",\n".join(data_lines)
        }

        # Replace template placeholders using format dictionary
        try:
            code = template_text.format(**format_dict)
        except KeyError as e:
            raise KeyError(f"Missing template variable: {e}")
        
        return code

class NoiseSource(AudioSource):
    def __init__(self, config: AudioConfig, noise_type: str = 'white'):
        super().__init__()
        self.config = config
        self.noise_type = noise_type
        
        # Use appropriate buffer sizes
        if (noise_type == 'spectral'):
            self.generator = SpectralNoiseGenerator()
            self._buffer_size = config.spectral_size  # Keep spectral size fixed
        else:
            self.generator = WhiteNoiseGenerator()
            self.generator.filters = self.filters
            self._buffer_size = config.buffer_size  # Use configurable size
        
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

    def _generate_chunk(self, frames: int) -> np.ndarray:
        """Generate noise chunk with proper buffering"""
        if frames <= 0:
            return np.array([], dtype=np.float32)
            
        if self.noise_type == 'spectral':
            # Fill synthesis buffer until we have enough frames
            while len(self._synthesis_buffer) < frames:
                # Always generate using spectral_size for consistency
                data = self.generator.generate(self.config.spectral_size, self.config.sample_rate)
                if data.size > 0:
                    self._synthesis_buffer = np.concatenate((self._synthesis_buffer, data))
                else:
                    break
                    
            if len(self._synthesis_buffer) >= frames:
                # Extract required frames
                output_data = self._synthesis_buffer[:frames]
                # Update buffer
                self._synthesis_buffer = self._synthesis_buffer[frames:]
                self._last_chunk = output_data
                return output_data
            else:
                return np.zeros(frames, dtype=np.float32)
        else:
            # White noise can be generated at exact frame size
            data = self.generator.generate(frames, self.config.sample_rate)
            self._last_chunk = data
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
        if not self._running:
            outdata.fill(0)
            return

        try:
            # Generate fresh chunk for audio
            audio_data = self._generate_chunk(frames)
            self._last_chunk = audio_data  # Store for visualization
            
            if self.config.monitoring_enabled:
                # Clip the audio data to prevent overflow
                scaled_data = audio_data.reshape(-1, self.config.channels) * self.config.monitoring_volume
                np.clip(scaled_data, -1.0, 1.0, out=outdata)
            else:
                outdata.fill(0)

        except Exception as e:
            print(f"Audio callback error: {e}")
            outdata.fill(0)
            if self.config.on_underflow:
                self.config.on_underflow()

    def _setup_stream(self):
        with self._lock:
            self._running = True
            self._audio_buffer = np.array([], dtype=np.float32)
            
            if (self.config.device_output_index is not None and 
                self.config.output_device_enabled):
                try:
                    # Pre-fill buffer with validation
                    test_data = self._generate_chunk(self._buffer_size)
                    if np.any(np.isnan(test_data)) or np.any(np.isinf(test_data)):
                        raise ValueError("Invalid audio data generated during setup")
                    self._audio_buffer = test_data
                    
                    safe_blocksize = self.config.output_buffer_size
                    
                    print(f"Using blocksize: {safe_blocksize}")  # Diagnostic print
                    
                    self.stream = sd.OutputStream(
                        device=self.config.device_output_index,
                        channels=self.config.channels,
                        samplerate=self.config.sample_rate,
                        blocksize=safe_blocksize,
                        dtype=np.float32,
                        callback=self._audio_callback,
                        latency='high'  # Use high latency for stability
                    )
                    self.stream.start()
                    
                except Exception as e:
                    print(f"Warning: Could not open audio output: {e}")
                    self.stream = None

    def update_output_device(self):
        """Update output device settings"""
        with self._lock:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            if self.config.device_output_index is not None:
                self.stream = sd.OutputStream(
                    device=self.config.device_output_index,
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
        # No need to add to generator since it shares the same list

    def remove_filter(self, index: int):
        """Remove filter from shared filter list"""
        super().remove_filter(index)  # This removes from self.filters
        # No need to remove from generator since it shares the same list

    def update_filter(self, index: int, params: dict):
        """Update filter in shared filter list"""
        super().update_filter(index, params)  # This updates in self.filters
        # No need to update in generator since it shares the same list

    def add_parabola(self, params: dict):
        """Add a new parabola component"""
        self.generator.add_parabola(params)

    def remove_parabola(self, index: int):
        """Remove a parabola component"""
        self.generator.remove_parabola(index)

    def update_parabola(self, index: int, params: dict):
        """Update parabola parameters"""
        self.generator.update_parabola(index, params)

    def export_signal(self, duration: float, sample_rate: int, amplitude: float,
                     enable_fade: bool = True, fade_in_duration: float = 0.001,
                     fade_out_duration: float = 0.001, 
                     fade_in_power: float = 0.5, fade_out_power: float = 0.5, 
                     enable_normalization: bool = True,
                     normalize_value: float = 1.0) -> np.ndarray:
        """Generate and process signal for export using same pipeline as real-time"""
        # Set generator normalization settings
        self.generator.normalize = enable_normalization
        self.generator.normalize_value = normalize_value
        
        # Package all parameters into kwargs dict
        kwargs = {
            'amplitude': amplitude,
            'enable_fade': enable_fade,
            'fade_in_duration': fade_in_duration,
            'fade_out_duration': fade_out_duration,
            'fade_in_power': fade_in_power,
            'fade_out_power': fade_out_power,
            'enable_normalization': enable_normalization,
            'normalize_value': normalize_value
        }
        
        return AudioExporter.export_signal(self.generator, duration, sample_rate, **kwargs)

    def generate_cpp_code(self, signal: np.ndarray, sample_rate: int) -> str:
        """Generate Arduino/ESP32 compatible C++ code with bounds checking"""
        return AudioExporter.generate_cpp_code(signal, sample_rate)

    def set_spectral_normalization(self, enabled: bool):
        """Set whether spectral synthesis should normalize output"""
        if isinstance(self.generator, SpectralNoiseGenerator):
            self.generator.update_parameters({'normalize': enabled})

    def set_filter_normalization(self, enabled: bool):
        """Set whether white noise filtering should normalize output"""
        if isinstance(self.generator, WhiteNoiseGenerator):
            self.generator.update_parameters({'normalize': enabled})

    def set_rng_type(self, rng_type: str):
        """Set the RNG distribution type"""
        self.rng_type = rng_type
        self.generator.set_rng_type(rng_type)