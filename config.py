# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioConfig:
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
    min_db: float = -80
    max_db: float = 0

    # Device settings
    device_input_index: Optional[int] = None
    device_output_index: Optional[int] = None
    
    # Device enabled flags
    input_device_enabled: bool = False
    output_device_enabled: bool = False
    
    # Monitoring settings
    monitoring_enabled: bool = False
    monitoring_volume: float = 0.5
    
    # Status callbacks
    on_overflow: Optional[callable] = None
    on_underflow: Optional[callable] = None

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
    normalize_value: float = 1.0
    
    # Separate amplitude parameters for each mode
    amp_whitenoise: float = 0.5
    amp_spectral: float = 1.0