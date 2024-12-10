import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from config import AudioConfig

class SettingsManager:
    def __init__(self, app_name: str = "spectrum_analyzer"):
        self.app_name = app_name
        self.default_settings = {
            'analyzer': {
                'fft_size': 2048,
                'window_type': 'hanning',
                'scale_type': 'linear',
                'averaging_count': 4
            },
            'source': {
                'source_type': 'White Noise',
                'monitoring_enabled': False,
                'monitoring_volume': 20,
                'output_device_index': None,
                'input_device_index': None
            },
            'filters': {
                'filters': []
            }
        }
        self.program_settings = self.default_settings.copy()
        self.settings_file = self._get_settings_path()

    def _get_settings_path(self) -> Path:
        """Get platform-specific settings path"""
        if os.name == 'nt':  # Windows
            base_path = Path(os.getenv('APPDATA'))
        else:  # Unix/Linux/Mac
            base_path = Path.home() / '.config'
            
        return base_path / self.app_name / 'settings.json'

    def save_program_settings(self, settings: Dict[str, Any]):
        """Save program settings without device indices"""
        # Deep copy settings and remove device indices
        save_settings = self._remove_device_indices(settings)
        
        # Ensure directory exists
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save settings
        with open(self.settings_file, 'w') as f:
            json.dump(save_settings, f, indent=4)

    def load_program_settings(self) -> Dict[str, Any]:
        """Load program settings, falling back to defaults if needed"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                return self._merge_settings(self.default_settings, loaded)
        except Exception as e:
            print(f"Error loading program settings: {e}")
        return self.default_settings.copy()

    def load_profile(self, filename: str) -> Dict[str, Any]:
        """Load user profile, preserving device settings from program settings"""
        try:
            with open(filename, 'r') as f:
                profile = json.load(f)
            
            # Get current device settings
            current_devices = self._get_device_indices(self.program_settings)
            
            # Merge profile with defaults, giving priority to profile
            merged = self._merge_settings(self.default_settings, profile)
            
            # Restore current device settings
            merged = self._restore_device_indices(merged, current_devices)
            
            return merged
            
        except Exception as e:
            print(f"Error loading profile: {e}")
            return self.program_settings.copy()

    def _merge_settings(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge settings, with override taking priority"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_settings(result[key], value)
            else:
                result[key] = value
        return result

    def _get_device_indices(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device indices from settings"""
        source = settings.get('source', {})
        return {
            'input_device_index': source.get('input_device_index'),
            'output_device_index': source.get('output_device_index')
        }

    def _remove_device_indices(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Create copy of settings without device indices"""
        settings = settings.copy()
        if 'source' in settings:
            settings['source'] = settings['source'].copy()
            settings['source']['input_device_index'] = None
            settings['source']['output_device_index'] = None
        return settings

    def _restore_device_indices(self, settings: Dict[str, Any], devices: Dict[str, Any]) -> Dict[str, Any]:
        """Restore device indices to settings"""
        settings = settings.copy()
        if 'source' in settings:
            settings['source'] = settings['source'].copy()
            settings['source']['input_device_index'] = devices['input_device_index']
            settings['source']['output_device_index'] = devices['output_device_index']
        return settings

def save_profile(config: AudioConfig, filepath: str):
    profile_data = {
        # ...existing code...
        'export': {
            'format': config.export_format,
            'duration': config.export_duration,
            'amplitude': config.export_amplitude,
            'fade_in_duration': config.fade_in_duration,
            'fade_out_duration': config.fade_out_duration,
            'fade_in_power': config.fade_in_power,
            'fade_out_power': config.fade_out_power,
            'enable_fade': config.enable_fade,
            'enable_normalization': config.enable_normalization,
            'normalize_value': config.normalize_value,
        }
    }
    with open(filepath, 'w') as f:
        json.dump(profile_data, f, indent=4)

def load_profile(config: AudioConfig, filepath: str):
    with open(filepath, 'r') as f:
        profile_data = json.load(f)
        
    if 'export' in profile_data:
        export_settings = profile_data['export']
        config.export_format = export_settings.get('format', config.export_format)
        config.export_duration = export_settings.get('duration', config.export_duration)
        config.export_amplitude = export_settings.get('amplitude', config.export_amplitude)
        config.fade_in_duration = export_settings.get('fade_in_duration', config.fade_in_duration)
        config.fade_out_duration = export_settings.get('fade_out_duration', config.fade_out_duration)
        config.fade_in_power = export_settings.get('fade_in_power', config.fade_in_power)
        config.fade_out_power = export_settings.get('fade_out_power', config.fade_out_power)
        config.enable_fade = export_settings.get('enable_fade', config.enable_fade)
        config.enable_normalization = export_settings.get('enable_normalization', config.enable_normalization)
        config.normalize_value = export_settings.get('normalize_value', config.normalize_value)