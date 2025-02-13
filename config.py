# config.py
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Dict, Any
import json
import os
from pathlib import Path
import logging

# Configure logging with a default level
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.CRITICAL  # Set default to CRITICAL to match AudioConfig default
)
logger = logging.getLogger(__name__)

VERSION = "1.1.1"  # Updated version number

@dataclass
class AudioConfig:
    # Logging settings
    log_level: str = 'ERROR'  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Audio I/O settings
    chunk_size: int = 1024     # Processing chunk size - controlled by UI
    buffer_size: int = 1024    # Universal buffer size - controlled by UI
    input_buffer_size: int = 1024  # Input device specific - controlled by UI
    output_buffer_size: int = 1024  # Output device specific - controlled by UI
    spectral_size: int = 8192  # Increased for better frequency resolution
    sample_rate: int = 44100
    channels: int = 1
    
    # Analysis settings
    fft_size: int = 2048    
    window_type: str = 'hanning'
    scale_type: str = 'linear'
    averaging_count: int = 4
    min_db: float = -90
    max_db: float = 0
    trigger_level: float = -45  # Changed from -60 to -45
    
    # Device enabled flags
    input_device_enabled: bool = False
    output_device_enabled: bool = False
    
    # Monitoring settings
    monitoring_enabled: bool = False
    monitoring_volume: float = 0.2  # Changed from 0.5 to 0.2
    
    # Status callbacks
    on_overflow: Optional[Callable] = None
    on_underflow: Optional[Callable] = None

    # RNG settings
    rng_type: str = 'uniform'  # Default RNG type (uniform or standard_normal)

    # Export dialog settings
    export_format: str = 'wav'
    export_duration: float = 1.0
    export_amplitude: float = 1.0
    fade_in_duration: float = 0.001
    fade_out_duration: float = 0.001
    fade_in_power: float = 0.5
    fade_out_power: float = 0.5
    enable_fade: bool = True
    enable_normalization: bool = True
    normalize_value: float = 0.5
    fade_before_norm: bool = True  # Default to "Fade then Normalize"
    
    # Separate amplitude parameters for each mode
    amp_whitenoise: float = 1.0
    amp_spectral: float = 1.0
    
    # Carousel mode settings
    carousel_enabled: bool = False
    carousel_samples: int = 20
    carousel_noise_duration_ms: float = 10.0
    carousel_silence_duration_ms: float = 190.0
    carousel_individual_files: bool = False
    
    # Export folder settings
    last_export_folder: str = ""
    last_export_template: str = ""
    export_individual_files: bool = False

    def __post_init__(self):
        # Set the global logging level when AudioConfig is instantiated
        logging.getLogger().setLevel(self.log_level)

    def to_dict(self) -> dict:
        """Convert config to dictionary, excluding callbacks"""
        d = asdict(self)
        # Remove callback functions
        d.pop('on_overflow', None)
        d.pop('on_underflow', None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'AudioConfig':
        """Create config from dictionary, ignoring unknown fields"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

class SettingsManager:
    def __init__(self, app_name: str = "spectrum_analyzer"):
        self.app_name = app_name
        self.settings_file = self._get_settings_path()
        self.config = AudioConfig()
        self._initialize_default_settings()
        # Apply initial log level
        logging.getLogger().setLevel(self.config.log_level)

    def _get_settings_path(self) -> Path:
        """Get platform-specific settings path"""
        if os.name == 'nt':  # Windows
            base_path = Path(os.getenv('APPDATA'))
        else:  # Unix/Linux/Mac
            base_path = Path.home() / '.config'
        return base_path / self.app_name / 'settings.json'

    def _initialize_default_settings(self):
        """Initialize default settings structure"""
        self.default_settings = {
            'analyzer': {
                'fft_size': self.config.fft_size,
                'window_type': self.config.window_type,
                'scale_type': self.config.scale_type,
                'averaging_count': self.config.averaging_count
            },
            'source': {
                'source_type': 'White Noise',
                'monitoring_enabled': self.config.monitoring_enabled,
                'monitoring_volume': int(self.config.monitoring_volume * 100),
                'output_device_index': None,
                'input_device_index': None,
                'amp_whitenoise': self.config.amp_whitenoise,
                'amp_spectral': self.config.amp_spectral,
                'carousel_template': {
                    'template_text': (
                        "#define SAMPLE_RATE 44100\n"
                        "#define NUM_BUFFERS @{num_buffers}\n"
                        "#define MONO_SAMPLES @{samples_per_buffer}  // Samples per buffer\n"
                        "#define STEREO_SAMPLES (MONO_SAMPLES * 2)\n"
                        "#define SILENCE_SAMPLES @{silence_samples * 2}\n\n"
                        "// Noise samples for carousel playback\n"
                        "// Generated with @{generator_type}\n\n"
                        "int16_t @{buffer_name}[@{samples_per_buffer}] = {@{data}};\n\n"
                        "int16_t @{silence_buffer_name}[SILENCE_SAMPLES] = {@{silence_data}};\n"
                        "int16_t* @{buffer_array_name}[NUM_BUFFERS] = {@{buffer_list}};\n"
                        "int currentBufferIndex = 0;\n"
                    ),
                    'buffer_name_format': 'buffer@{index+1}',
                    'buffer_array_name': 'noiseBuffers',
                    'silence_buffer_name': 'silenceBuffer'
                }
            },
            'audio': {
                'chunk_size': self.config.chunk_size,
                'buffer_size': self.config.buffer_size,
                'input_buffer_size': self.config.input_buffer_size,
                'output_buffer_size': self.config.output_buffer_size
            }
        }

    def save_settings(self, settings: Dict[str, Any]):
        """Save settings to file"""
        try:
            # Ensure directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Remove device indices before saving
            save_settings = self._remove_device_indices(settings)
            
            with open(self.settings_file, 'w') as f:
                json.dump(save_settings, f, indent=4)
                
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                return self._merge_settings(self.default_settings, loaded)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
        return self.default_settings.copy()

    def _merge_settings(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge settings, with override taking priority"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        return result

    def _remove_device_indices(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Create copy of settings without device indices"""
        settings = settings.copy()
        if 'source' in settings:
            settings['source'] = settings['source'].copy()
            settings['source']['input_device_index'] = None
            settings['source']['output_device_index'] = None
        return settings

    def apply_to_config(self, settings: Dict[str, Any]):
        """Apply loaded settings to AudioConfig instance"""
        if 'analyzer' in settings:
            analyzer = settings['analyzer']
            self.config.fft_size = analyzer.get('fft_size', self.config.fft_size)
            self.config.window_type = analyzer.get('window_type', self.config.window_type)
            self.config.scale_type = analyzer.get('scale_type', self.config.scale_type)
            self.config.averaging_count = analyzer.get('averaging_count', self.config.averaging_count)

        if 'source' in settings:
            source = settings['source']
            self.config.monitoring_enabled = source.get('monitoring_enabled', self.config.monitoring_enabled)
            self.config.monitoring_volume = source.get('monitoring_volume', self.config.monitoring_volume) / 100.0
            self.config.amp_whitenoise = source.get('amp_whitenoise', self.config.amp_whitenoise)
            self.config.amp_spectral = source.get('amp_spectral', self.config.amp_spectral)

        if 'audio' in settings:
            audio = settings['audio']
            self.config.chunk_size = audio.get('chunk_size', self.config.chunk_size)
            self.config.buffer_size = audio.get('buffer_size', self.config.buffer_size)
            self.config.input_buffer_size = audio.get('input_buffer_size', self.config.input_buffer_size)
            self.config.output_buffer_size = audio.get('output_buffer_size', self.config.output_buffer_size)
            
        # Apply log level if it has changed
        if self.config.log_level != logging.getLogger().getEffectiveLevel():
            logging.getLogger().setLevel(self.config.log_level)

    def get_config(self) -> AudioConfig:
        """Get current AudioConfig instance"""
        return self.config