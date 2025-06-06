# ui_components.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QMenuBar, QMenu, QStatusBar, QMainWindow,
    QMessageBox, QDialog, QFileDialog, QDialogButtonBox, QScrollArea, 
    QFrame, QSizePolicy, QLineEdit, QGridLayout, QListWidget, QListWidgetItem,
    QPlainTextEdit, QInputDialog, QTabWidget  # Add QTabWidget here
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from typing import Dict, Any, List, Tuple, Optional  # Add these imports
import sounddevice as sd
from config import AudioConfig, SettingsManager
from audio_sources import MonitoredInputSource
from PyQt6.QtGui import QDoubleValidator
import numpy as np
import os  # Add this import
import threading
import logging
from PyQt6.QtCore import QSettings
import time
import config

logger = logging.getLogger(__name__)

# At module level, before classes
def update_device_list(combo: QComboBox, input_devices: bool = False):
    """Helper function to update device list in combo boxes"""
    combo.clear()
    combo.addItem("No Audio Device", None)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        channels = device['max_input_channels'] if input_devices else device['max_output_channels']
        if channels > 0:
            name = f"{device['name']} ({('In' if input_devices else 'Out')}: {channels})"
            combo.addItem(name, i)

class ParameterControl(QWidget):
    """Combined slider and spinbox control for parameters"""
    valueChanged = pyqtSignal(float)

    def __init__(self, min_val: float, max_val: float, value: float, decimals: int = 1, suffix: str = "", step: float = None, linked_param: str = None, linked_control = None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.slider_scale = 10 ** decimals
        self.linked_param = linked_param
        self.linked_control = linked_control
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Create spinbox
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(value)
        self.spinbox.setDecimals(decimals)
        self.spinbox.setSuffix(suffix)
        if step:
            self.spinbox.setSingleStep(step)
        self.spinbox.setFixedWidth(90)

        # Create slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        slider_scale = 10 ** decimals
        self.slider.setRange(int(min_val * slider_scale), int(max_val * slider_scale))
        self.slider.setValue(int(value * slider_scale))
        self.slider_scale = slider_scale

        # Connect signals
        self.spinbox.valueChanged.connect(self._spinbox_changed)
        self.slider.valueChanged.connect(self._slider_changed)

        # Add widgets to layout
        layout.addWidget(self.spinbox)
        layout.addWidget(self.slider, stretch=1)

    def _spinbox_changed(self, value):
        if self._validate_against_linked(value):
            self.slider.blockSignals(True)
            self.slider.setValue(int(value * self.slider_scale))
            self.slider.blockSignals(False)
            self.valueChanged.emit(value)
        else:
            # Revert to last valid value
            valid_value = self._get_valid_value(value)
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(valid_value)
            self.spinbox.blockSignals(False)

    def _slider_changed(self, value):
        actual = value / self.slider_scale
        if self._validate_against_linked(actual):
            self.spinbox.blockSignals(True)
            self.spinbox.setValue(actual)
            self.spinbox.blockSignals(False)
            self.valueChanged.emit(actual)
        else:
            # Revert to last valid value
            valid_value = self._get_valid_value(actual)
            self.slider.blockSignals(True)
            self.slider.setValue(int(valid_value * self.slider_scale))
            self.slider.blockSignals(False)

    def _validate_against_linked(self, value):
        if not self.linked_control or not self.linked_param:
            return True
            
        if self.linked_param == 'lowcut':
            return value <= self.linked_control.value()
        elif self.linked_param == 'highcut':
            return value >= self.linked_control.value()
        return True

    def _get_valid_value(self, attempted_value):
        if not self.linked_control or not self.linked_param:
            return attempted_value
            
        if self.linked_param == 'lowcut':
            return min(attempted_value, self.linked_control.value())
        elif self.linked_param == 'highcut':
            return max(attempted_value, self.linked_control.value())
        return attempted_value

    def value(self) -> float:
        return self.spinbox.value()

    def setValue(self, value: float):
        self.spinbox.setValue(value)

class BufferSettingsDialog(QDialog):
    def __init__(self, config: AudioConfig, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)  # Just use Window type
        self.config = config
        self.setWindowTitle("Audio Buffer Settings")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # Input buffer size
        self.input_buffer = QComboBox()
        self.input_buffer.addItems(['256', '512', '1024', '2048', '4096'])
        self.input_buffer.setCurrentText(str(self.config.input_buffer_size))
        layout.addRow("Input Buffer:", self.input_buffer)
        
        # Output buffer size
        self.output_buffer = QComboBox()
        self.output_buffer.addItems(['256', '512', '1024', '2048', '4096'])
        self.output_buffer.setCurrentText(str(self.config.output_buffer_size))
        layout.addRow("Output Buffer:", self.output_buffer)
        
        # Processing chunk size
        self.chunk_size = QComboBox()
        self.chunk_size.addItems(['128', '256', '512', '1024', '2048'])
        self.chunk_size.setCurrentText(str(self.config.chunk_size))
        layout.addRow("Chunk Size:", self.chunk_size)

        # Add help text
        help_text = QLabel(
            "Note: Increase buffer sizes if you experience audio glitches.\n"
            "Larger buffers increase latency but improve stability."
        )
        help_text.setWordWrap(True)
        layout.addRow(help_text)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self) -> dict:
        return {
            'input_buffer_size': int(self.input_buffer.currentText()),
            'output_buffer_size': int(self.output_buffer.currentText()),
            'chunk_size': int(self.chunk_size.currentText())
        }

class MonitoringPanel(QGroupBox):
    monitoring_changed = pyqtSignal(bool)
    volume_changed = pyqtSignal(float)
    settings_clicked = pyqtSignal()

    def __init__(self, config: AudioConfig):
        super().__init__("Audio Monitoring")
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)
        layout.setSpacing(4)  # Increased from 1
        layout.setContentsMargins(6, 8, 6, 8)  # Increased from 1,1,1,1
        layout.setVerticalSpacing(6)  # Add vertical spacing between rows
        layout.setHorizontalSpacing(8)  # Add horizontal spacing between label and field

        # Monitoring checkbox
        self.monitor_checkbox = QCheckBox("Enable Monitoring")
        self.monitor_checkbox.setChecked(self.config.monitoring_enabled)
        self.monitor_checkbox.stateChanged.connect(self.on_monitor_toggled)
        layout.addRow(self.monitor_checkbox)

        # Device selector - use DeviceComboBox instead of QComboBox
        self.device_combo = DeviceComboBox(input_devices=False)
        self.device_combo.currentIndexChanged.connect(lambda: self.monitoring_changed.emit(self.monitor_checkbox.isChecked()))
        layout.addRow("Output Device:", self.device_combo)

        # Volume slider with marks
        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        initial_volume = int(self.config.monitoring_volume * 100)
        self.volume_slider.setValue(initial_volume)  # Use config value
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        
        volume_label = QLabel("Volume:")
        volume_value = QLabel(f"{initial_volume}%")  # Initial value
        self.volume_slider.valueChanged.connect(
            lambda v: volume_value.setText(f"{v}%"))
        
        volume_layout.addWidget(volume_label)
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(volume_value)
        layout.addRow(volume_layout)

        # Settings button and status indicators
        settings_layout = QHBoxLayout()
        self.settings_button = QPushButton("Buffer Settings...")
        self.settings_button.clicked.connect(self.settings_clicked.emit)
        settings_layout.addWidget(self.settings_button)

        # Status indicators moved here
        self.overflow_indicator = QLabel("OF")
        self.underflow_indicator = QLabel("UF")
        # Add tooltips for each indicator
        self.overflow_indicator.setToolTip(
            "Input Overflow - CPU isn't consuming sound device input data fast enough\n"
            "May introduce clicks/pops in the monitored audio\n"
            "Click to reset indicator"
        )
        self.underflow_indicator.setToolTip(
            "Output Underflow - CPU isn't producing data fast enough for sound device\n"
            "May introduce clicks/pops in the output audio\n"
            "Click to reset indicator"
        )
        for indicator in [self.overflow_indicator, self.underflow_indicator]:
            indicator.setStyleSheet("""
                QLabel {
                    color: white;
                    background: gray;
                    padding: 2px 5px;
                    border-radius: 3px;
                    font-weight: bold;
                }
            """)
            indicator.setCursor(Qt.CursorShape.PointingHandCursor)
            indicator.mousePressEvent = lambda _, i=indicator: self._reset_indicator(i)
            settings_layout.addWidget(indicator)

        # Hide indicators by default
        self.overflow_indicator.setVisible(False)
        self.underflow_indicator.setVisible(False)

        settings_layout.addStretch()
        layout.addRow(settings_layout)

    def _reset_indicator(self, indicator):
        """Reset indicator when clicked"""
        indicator.setStyleSheet("""
            QLabel {
                color: white;
                background: gray;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)

    def set_overflow(self):
        """Set overflow indicator"""
        self.overflow_indicator.setStyleSheet("""
            QLabel {
                color: white;
                background: red;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)

    def set_underflow(self):
        """Set underflow indicator"""
        self.underflow_indicator.setStyleSheet("""
            QLabel {
                color: white;
                background: #FF6600;
                padding: 2px 5px;
                border-radius: 3px;
                font-weight: bold;
            }
        """)

    def update_device_list(self):
        update_device_list(self.device_combo, input_devices=False)

    def on_monitor_toggled(self, enabled: bool):
        self.monitoring_changed.emit(enabled)
        self.config.monitoring_enabled = enabled

    def on_volume_changed(self, value: int):
        volume = value / 100.0
        self.volume_changed.emit(volume)
        self.config.monitoring_volume = volume

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current panel settings"""
        return {
            'monitoring_enabled': self.monitor_checkbox.isChecked(),
            'monitoring_volume': self.volume_slider.value(),
            'output_device': self.device_combo.get_device_info()
        }

    def apply_settings(self, settings: Dict[str, Any]):
        """Apply loaded settings"""
        if 'monitoring_enabled' in settings:
            self.monitor_checkbox.setChecked(settings['monitoring_enabled'])
        if 'monitoring_volume' in settings:
            # Convert volume to integer percentage if it's a float
            volume = settings['monitoring_volume']
            if isinstance(volume, float):
                volume = int(volume * 100)
            self.volume_slider.setValue(volume)
        if 'output_device' in settings:
            self.device_combo.set_device_from_info(settings['output_device'])

class InputDevicePanel(QGroupBox):
    device_changed = pyqtSignal(int)

    def __init__(self, config: AudioConfig):
        super().__init__("Input Device")
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # Device selector - use DeviceComboBox instead of QComboBox
        self.device_combo = DeviceComboBox(input_devices=True)
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        layout.addRow("Input Device:", self.device_combo)

        # Input channel selector
        self.channel_combo = QComboBox()
        self.channel_combo.setVisible(False)  # Initially hidden
        layout.addRow("Input Channel:", self.channel_combo)

    def update_device_list(self):
        update_device_list(self.device_combo, input_devices=True)

    def on_device_changed(self):
        device_idx = self.device_combo.currentData()
        # Only track if device is enabled, not the specific index
        self.config.input_device_enabled = (device_idx is not None)
        
        if device_idx is not None:
            # Update channel selector
            device_info = sd.query_devices(device_idx)
            self.channel_combo.clear()
            channels = device_info['max_input_channels']
            if channels > 0:
                for i in range(channels):
                    self.channel_combo.addItem(f"Channel {i+1}", i)
                self.channel_combo.setVisible(True)
                self.channel_combo.setEnabled(channels > 1)  # Only enable selection if multiple channels
            else:
                self.channel_combo.setVisible(False)
        else:
            # Hide channel selector when no device is selected
            self.channel_combo.setVisible(False)
            
        # Always emit the signal to trigger UI updates
        self.device_changed.emit(-1 if device_idx is None else device_idx)

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current panel settings"""
        return {
            'input_device': self.device_combo.get_device_info()
        }

    def apply_settings(self, settings: Dict[str, Any]):
        """Apply loaded settings"""
        if 'input_device' in settings:
            self.device_combo.set_device_from_info(settings['input_device'])

class SourcePanel(QGroupBox):
    source_changed = pyqtSignal()
    export_requested = pyqtSignal(dict)  # Add new signal

    def __init__(self, config: AudioConfig):
        super().__init__("Audio Source")
        self.config = config
        self.is_playing = False
        self.current_source = None
        self.export_settings = {}
        
        # Set fixed size policy for the panel
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        # Create panels with fixed size policies
        self.monitoring_panel = MonitoringPanel(config)
        self.monitoring_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        
        self.input_device_panel = InputDevicePanel(config)
        self.input_device_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.input_device_panel.hide()

        # Initialize cpp_template using default template
        default_template = CppTemplate.get_default_templates()[0]
        self.cpp_template = {
            'template_text': default_template.before + default_template.after,
            'var_name': default_template.var_name,
            'length_name': default_template.length_name
        }
        
        # Initialize carousel_template
        self.carousel_template = SettingsManager().default_settings['source']['carousel_template']
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # Increased spacing between sections
        layout.setContentsMargins(6, 8, 6, 8)  # Increased margins
        
        # Source type selector
        source_layout = QHBoxLayout()
        source_layout.setSpacing(8)  # Space between label and combo
        source_label = QLabel("Source Type:")
        self.source_type = QComboBox()
        self.source_type.addItems(["White Noise", "Spectral Synthesis", "Test Mode"])
        self.source_type.currentTextChanged.connect(self.on_source_type_changed)
        source_layout.addWidget(source_label)
        source_layout.addWidget(self.source_type)
        layout.addLayout(source_layout)
        
        # Add spacing before monitoring panel
        layout.addSpacing(4)
        
        # Add monitoring panel
        layout.addWidget(self.monitoring_panel)
        
        # Add spacing before input device panel
        layout.addSpacing(4)
        
        # Add input device panel (initially hidden)
        layout.addWidget(self.input_device_panel)
        
        # Connect device change signals
        self.monitoring_panel.device_combo.currentIndexChanged.connect(self.on_output_device_changed)
        self.input_device_panel.device_combo.currentIndexChanged.connect(self.on_input_device_changed)
        
        # Export button container - separate from mode selector
        export_container = QWidget()
        export_layout = QHBoxLayout(export_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        
        # Export button - only for noise modes
        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export_noise)
        self.export_button.setFixedWidth(100)  # Set fixed width for consistency
        
        # Create invisible placeholder with same size
        self.export_placeholder = QWidget()
        self.export_placeholder.setFixedWidth(100)
        self.export_placeholder.hide()
        
        export_layout.addWidget(self.export_button)
        export_layout.addWidget(self.export_placeholder)
        export_layout.addStretch()  # Add stretch to keep button left-aligned
        
        layout.addWidget(export_container)

        # Create container widget for RNG controls
        self.rng_container = QWidget()
        rng_layout = QHBoxLayout(self.rng_container)
        rng_layout.setContentsMargins(0, 0, 0, 0)
        rng_layout.addWidget(QLabel("RNG Type:"))
        self.rng_type = QComboBox()
        self.rng_type.addItems(["Uniform", "Standard Normal"])
        # Set initial value from config
        initial_rng = "Uniform" if self.config.rng_type == 'uniform' else "Standard Normal"
        self.rng_type.setCurrentText(initial_rng)
        self.rng_type.currentTextChanged.connect(self.on_rng_type_changed)
        rng_layout.addWidget(self.rng_type)
        layout.addWidget(self.rng_container)

        # Create container widget for amplitude controls
        self.amp_container = QWidget()
        amp_layout = QHBoxLayout(self.amp_container)
        amp_layout.setContentsMargins(0, 0, 0, 0)
        amp_layout.addWidget(QLabel("Amplitude:"))
        self.amplitude_control = QDoubleSpinBox()
        self.amplitude_control.setRange(0.0, 1.0)
        self.amplitude_control.setValue(self.config.amp_whitenoise)
        self.amplitude_control.setSingleStep(0.1)
        self.amplitude_control.valueChanged.connect(self.on_amplitude_changed)
        amp_layout.addWidget(self.amplitude_control)
        layout.addWidget(self.amp_container)

        # Play/Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        # Connect signals
        self.source_type.currentTextChanged.connect(self.on_source_type_changed)
        self.monitoring_panel.monitoring_changed.connect(self.on_monitoring_changed)
        self.monitoring_panel.volume_changed.connect(self.on_volume_changed)
        self.input_device_panel.device_changed.connect(self.on_device_changed)  # Add this line
        
        # Initialize UI state based on current source type
        self.on_source_type_changed(self.source_type.currentText())

    def _settings_changed(self, old_settings: dict, new_settings: dict) -> bool:
        """Deep compare export settings to detect changes"""
        if old_settings is None:
            logger.debug("Export settings changed: old settings were None")
            return True
            
        # List of all settings to compare
        basic_settings = [
            'duration', 'sample_rate', 'base_amplitude', 'amplitude',
            'attenuation', 'enable_attenuation',
            'export_wav', 'export_cpp',
            'enable_fade_in', 'enable_fade_out', 'enable_fade',
            'fade_in_duration', 'fade_out_duration',
            'fade_in_power', 'fade_out_power',
            'enable_normalization', 'normalize_value',
            'fade_before_norm', 'use_random_seed', 'seed',
            'rng_type', 'folder_path', 'wav_filename', 'cpp_filename',
            'header_filename', 'carousel_enabled', 'carousel_samples',
            'carousel_noise_duration_ms', 'silence_duration_ms',
            'export_combined', 'export_individual', 'global_normalization'
        ]
        
        # Compare basic settings
        for key in basic_settings:
            old_value = old_settings.get(key)
            new_value = new_settings.get(key)
            if old_value != new_value:
                logger.debug(f"Setting changed: {key} from {old_value} to {new_value}")
                return True
            
        # Compare templates if they exist
        old_cpp = old_settings.get('cpp_template')
        new_cpp = new_settings.get('cpp_template')
        
        # If either template is None, but they're different, settings have changed
        if (old_cpp is None) != (new_cpp is None):
            logger.debug("One template is None while other isn't - settings changed")
            return True

        # If both templates exist, compare their fields
        if old_cpp is not None and new_cpp is not None:
            cpp_fields = ['template_text', 'var_name', 'length_name']
            for field in cpp_fields:
                if old_cpp.get(field) != new_cpp.get(field):
                    logger.debug(f"C++ template changed: {field}")
                    return True
            
        # Compare carousel template if present
        old_carousel = old_settings.get('carousel_template')
        new_carousel = new_settings.get('carousel_template')
        
        # If either carousel template is None, but they're different, settings changed
        if (old_carousel is None) != (new_carousel is None):
            logger.debug("One carousel template is None while other isn't - settings changed")
            return True

        # If both carousel templates exist, compare their fields
        if old_carousel is not None and new_carousel is not None:
            carousel_fields = ['template_text', 'buffer_name_format', 'buffer_array_name', 'silence_buffer_name']
            for field in carousel_fields:
                if old_carousel.get(field) != new_carousel.get(field):
                    logger.debug(f"Carousel template changed: {field}")
                    return True
            
        # Compare filters if present
        old_filters = old_settings.get('filters', [])
        new_filters = new_settings.get('filters', [])
        if len(old_filters) != len(new_filters):
            logger.debug(f"Number of filters changed from {len(old_filters)} to {len(new_filters)}")
            return True
            
        for old_filter, new_filter in zip(old_filters, new_filters):
            for key in ['type', 'cutoff', 'order', 'amplitude', 'q', 'lowcut', 'highcut',
                       'center_freq', 'width', 'skew', 'kurtosis', 'flatness', 'flat_width']:
                if old_filter.get(key) != new_filter.get(key):
                    logger.debug(f"Filter setting changed: {key}")
                    return True
                    
        logger.debug("No export settings changed")
        return False

    def export_noise(self):
        """Trigger appropriate export dialog based on source type"""
        try:
            logger.debug("Starting export_noise with export_settings: %s", self.export_settings)  # Debug print
            source_type = self.get_source_type()
            dialog = ExportDialog(self, mode=source_type)
            
            # Apply stored settings and current mode's amplitude
            if self.export_settings:
                logger.debug("Applying saved settings: %s", self.export_settings)  # Debug print
                dialog.apply_saved_settings(self.export_settings)
            else:
                logger.debug("No saved settings found, using current amplitude")  # Debug print
                # Initialize with current amplitude
                current_amp = (self.config.amp_whitenoise 
                             if source_type == "White Noise" 
                             else self.config.amp_spectral)
                dialog.amplitude.setValue(current_amp)
            
            # Connect amplitude control to live amplitude
            dialog.amplitude.valueChanged.connect(self.amplitude_control.setValue)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                settings = dialog.get_settings()
                logger.debug("Got new settings from dialog: %s", settings)  # Debug print
                # Check if settings have changed using deep comparison
                if self._settings_changed(self.export_settings, settings):
                    logger.debug("Settings changed, storing new settings")  # Debug print
                    # Store settings for reuse
                    self.export_settings = settings.copy()  # Make a copy to avoid reference issues
                    # Save to QSettings immediately
                    qsettings = QSettings('YourOrg', 'SpectrumAnalyzer')
                    qsettings.setValue('export_settings', self.export_settings)
                    # Mark project as dirty
                    parent = self.parent()
                    while parent:
                        if hasattr(parent, 'mark_unsaved_changes'):
                            parent.mark_unsaved_changes()
                            break
                        parent = parent.parent()
                self.export_requested.emit(settings)
        except Exception as e:
            logger.error("Export dialog error: %s", e, exc_info=True)  # Added exc_info for full traceback

    def on_source_type_changed(self, new_type: str):
        """Handle source type changes with mode-specific amplitude"""
        is_test_mode = new_type == "Test Mode"
        
        # Update visibility of export button
        self.export_button.setVisible(not is_test_mode)
        self.export_placeholder.setVisible(is_test_mode)
        
        # Update amplitude based on mode
        if new_type == "White Noise":
            self.amplitude_control.setValue(self.config.amp_whitenoise)
        elif new_type == "Spectral Synthesis":
            self.amplitude_control.setValue(self.config.amp_spectral)
        
        # Stop playback if running
        if self.is_playing:
            self.toggle_playback()
            
        # Show/hide panels based on mode
        self.input_device_panel.setVisible(is_test_mode)
        self.rng_container.setVisible(not is_test_mode)
        self.amp_container.setVisible(not is_test_mode)
        
        # Update play button state for test mode
        if is_test_mode:
            device_idx = self.input_device_panel.device_combo.currentData()
            self.play_button.setEnabled(device_idx is not None)
        else:
            self.play_button.setEnabled(True)

    def on_input_device_changed(self):
        """Handle input device changes"""
        device_idx = self.input_device_panel.device_combo.currentData()
        self.config.input_device_enabled = (device_idx is not None)
        
        # Update play button state in test mode
        if self.source_type.currentText() == "Test Mode":
            self.play_button.setEnabled(device_idx is not None)
            
        # Stop playback if device is removed while playing
        if device_idx is None and self.is_playing:
            self.toggle_playback()

    def on_output_device_changed(self):
        """Handle output device changes"""
        if self.current_source and hasattr(self.current_source, 'update_output_device'):
            self.current_source.update_output_device()

    def get_current_settings(self) -> Dict[str, Any]:
        """Get current panel settings"""
        settings = {
            'source_type': self.source_type.currentText(),
            'monitoring_enabled': self.monitoring_panel.monitor_checkbox.isChecked(),
            'monitoring_volume': self.monitoring_panel.volume_slider.value(),
            'output_device': self.monitoring_panel.device_combo.get_device_info(),
            'input_device': self.input_device_panel.device_combo.get_device_info(),
            'amp_whitenoise': self.config.amp_whitenoise,
            'amp_spectral': self.config.amp_spectral,
            'rng_type': self.rng_type.currentText().lower().replace(' ', '_'),
            'cpp_template': self.cpp_template,
            'carousel_template': self.carousel_template
        }
        return settings

    def apply_settings(self, settings: Dict[str, Any]):
        """Apply loaded settings"""
        if 'source_type' in settings:
            index = self.source_type.findText(settings['source_type'])
            if index >= 0:
                self.source_type.setCurrentIndex(index)
                
        # Apply monitoring panel settings
        monitoring_settings = {
            'monitoring_enabled': settings.get('monitoring_enabled', False),
            'monitoring_volume': settings.get('monitoring_volume', 50),
            'output_device': settings.get('output_device', None)
        }
        self.monitoring_panel.apply_settings(monitoring_settings)
        
        # Apply input device panel settings
        input_settings = {
            'input_device': settings.get('input_device', None)
        }
        self.input_device_panel.apply_settings(input_settings)
        
        # Apply amplitude settings
        if 'amp_whitenoise' in settings:
            self.config.amp_whitenoise = settings['amp_whitenoise']
            if self.source_type.currentText() == "White Noise":
                self.amplitude_control.setValue(settings['amp_whitenoise'])
                
        if 'amp_spectral' in settings:
            self.config.amp_spectral = settings['amp_spectral']
            if self.source_type.currentText() == "Spectral Synthesis":
                self.amplitude_control.setValue(settings['amp_spectral'])
                
        # Apply RNG type if present
        if 'rng_type' in settings:
            rng_type = settings['rng_type']
            if isinstance(rng_type, str):
                rng_type = rng_type.title().replace('_', ' ')
            index = self.rng_type.findText(rng_type)
            if index >= 0:
                self.rng_type.setCurrentIndex(index)
                self.config.rng_type = settings['rng_type']
                
        # Apply template settings
        if 'cpp_template' in settings:
            self.cpp_template = settings['cpp_template']
        if 'carousel_template' in settings:
            self.carousel_template = settings['carousel_template']

    def on_device_changed(self):
        """Handle device selection changes"""
        device_idx = self.input_device_panel.device_combo.currentData()
        
        # Stop playback if no device is selected and we're currently playing
        if device_idx is None and self.is_playing:
            self.toggle_playback()  # Stop playback
            
        # Only track if device is enabled, not the specific index
        self.config.input_device_enabled = (device_idx is not None)
        
        # Update current source if running
        if self.current_source and hasattr(self.current_source, 'update_output_device'):
            self.current_source.update_output_device()
        
        # Update play button state if in test mode
        if self.source_type.currentText() == "Test Mode":
            self.play_button.setEnabled(device_idx is not None)

    def toggle_playback(self):
        """Toggle playback state"""
        main_window = self.window()
        if not hasattr(main_window, 'start_playback_change'):
            # Fallback or log error if main window is not as expected
            logger.error("Could not find main window to manage dirty state.")
            return
            
        main_window.start_playback_change()
        try:
            if self.is_playing:
                logger.debug("Stopping playback")
                # Update button first
                self.play_button.setText("Play")
                self.play_button.setChecked(False)
                self.is_playing = False
                
                # Then emit signal to stop processing
                self.source_changed.emit()
                
                # Finally clean up source
                if self.current_source:
                    try:
                        self.current_source.close()
                    finally:
                        self.current_source = None
                        
                # Re-enable device selection
                self.monitoring_panel.device_combo.setEnabled(True)
                self.input_device_panel.device_combo.setEnabled(True)

            else:
                logger.debug("Starting playback")
                # Get the current monitoring device
                output_device_idx = self.monitoring_panel.device_combo.currentData()
                input_device_idx = self.input_device_panel.device_combo.currentData()
                
                # In test mode, require input device
                if self.source_type.currentText() == "Test Mode":
                    if input_device_idx is None:
                        QMessageBox.warning(self, "Error", "Please select an input device")
                        self.play_button.setChecked(False)
                        return
                    device_idx = input_device_idx
                else:
                    device_idx = output_device_idx  # For noise sources, use output device
                
                # Update config before creating new source
                self.config.device_input = input_device_idx
                self.config.device_output = output_device_idx
                self.config.input_device_enabled = (input_device_idx is not None)
                self.config.output_device_enabled = (output_device_idx is not None)
                self.config.monitoring_enabled = self.monitoring_panel.monitor_checkbox.isChecked()
                self.config.monitoring_volume = self.monitoring_panel.volume_slider.value() / 100.0
                
                # Update state before emitting signal
                self.play_button.setText("Stop")
                self.play_button.setChecked(True)
                self.is_playing = True
                
                # Disable device selection during playback
                self.monitoring_panel.device_combo.setEnabled(False)
                self.input_device_panel.device_combo.setEnabled(False)
                
                # Finally emit signal to start processing
                self.source_changed.emit()

        except Exception as e:
            logger.error(f"Playback toggle error: {str(e)}")
            error_msg = str(e)
            # Check if it's a PortAudio error and provide a more user-friendly message
            if "PaErrorCode" in error_msg:
                if "Invalid sample rate" in error_msg:
                    error_msg = "The selected audio device does not support the current sample rate (44.1kHz).\nPlease select a different audio device."
                elif "Illegal combination of I/O devices" in error_msg:
                    error_msg = "The selected audio device configuration is not supported.\nPlease try a different audio device."
                else:
                    error_msg = f"Audio device error: {error_msg}"
            QMessageBox.critical(self, "Error", error_msg)
            self.is_playing = False
            self.play_button.setText("Play")
            self.play_button.setChecked(False)
            if self.current_source:
                try:
                    self.current_source.close()
                finally:
                    self.current_source = None
            # Re-enable device selection on error
            self.monitoring_panel.device_combo.setEnabled(True)
            self.input_device_panel.device_combo.setEnabled(True)
        finally:
            main_window.end_playback_change()

    def handle_source_reference(self, source):
        """Store reference to current source"""
        self.current_source = source

    def on_volume_changed(self, value):
        """Handle volume slider changes"""
        self.config.monitoring_volume = value / 100.0

    def on_monitoring_changed(self, enabled: bool):
        """Handle monitoring toggle"""
        try:
            self.config.monitoring_enabled = enabled
            if self.current_source:
                self.current_source.update_output_device()
        except Exception as e:
            logger.error(f"Monitoring toggle error: {e}")
            error_msg = str(e)
            # Check if it's a PortAudio error and provide a more user-friendly message
            if "PaErrorCode" in error_msg:
                if "Invalid sample rate" in error_msg:
                    error_msg = "The selected audio device does not support the current sample rate (44.1kHz).\nPlease select a different audio device."
                elif "Illegal combination of I/O devices" in error_msg:
                    error_msg = "The selected audio device configuration is not supported.\nPlease try a different audio device."
                else:
                    error_msg = f"Audio device error: {error_msg}"
            QMessageBox.critical(self, "Error", error_msg)
            # Reset checkbox state
            self.monitoring_panel.monitor_checkbox.setChecked(not enabled)
            self.config.monitoring_enabled = not enabled
            
            # Stop playback if running and re-enable device selection
            if self.is_playing:
                self.toggle_playback()  # This will stop playback and re-enable device selection

    def get_source_type(self) -> str:
        """Returns the current source type"""
        return self.source_type.currentText()

    def on_amplitude_changed(self, value: float):
        """Handle amplitude changes - store to appropriate config parameter"""
        source_type = self.source_type.currentText()
        if source_type == "White Noise":
            self.config.amp_whitenoise = value
        elif source_type == "Spectral Synthesis":
            self.config.amp_spectral = value
            
        if self.current_source and hasattr(self.current_source, 'generator'):
            self.current_source.generator.update_parameters({'amplitude': value})

    def on_rng_type_changed(self, rng_type: str):
        """Handle RNG type changes"""
        if self.current_source and hasattr(self.current_source, 'set_rng_type'):
            self.current_source.set_rng_type(rng_type.lower().replace(' ', '_'))

class AnalyzerPanel(QGroupBox):
    settings_changed = pyqtSignal()

    def __init__(self, config: AudioConfig):
        super().__init__("Analyzer Settings")
        self.config = config
        self.decay_rate = None
        # Set fixed size policy for the entire panel
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(2)  # Reduced spacing between sections
        layout.setContentsMargins(4, 6, 4, 6)  # Reduced margins

        # Create a grid layout for the main controls
        grid = QGridLayout()
        grid.setSpacing(4)  # Minimal spacing between elements

        # FFT Size selection
        fft_label = QLabel("FFT Size:")
        self.fft_size = QComboBox()
        self.fft_size.addItems(['128', '256', '512', '1024', '2048', '4096', '8192'])
        self.fft_size.setCurrentText(str(self.config.fft_size))
        self.fft_size.currentTextChanged.connect(self.on_fft_size_changed)
        grid.addWidget(fft_label, 0, 0)
        grid.addWidget(self.fft_size, 0, 1)

        # Scale type selection
        scale_label = QLabel("Scale:")
        self.scale_type = QComboBox()
        self.scale_type.addItems(["Linear", "Logarithmic"])
        self.scale_type.setCurrentText(self.config.scale_type.title())
        self.scale_type.currentTextChanged.connect(self.on_scale_changed)
        grid.addWidget(scale_label, 1, 0)
        grid.addWidget(self.scale_type, 1, 1)

        # Window type selection
        window_label = QLabel("Window:")
        self.window_type = QComboBox()
        self.window_type.addItems(["Hanning", "Hamming", "Blackman", "Rectangular", "Flattop"])
        self.window_type.setCurrentText(self.config.window_type.title())
        self.window_type.currentTextChanged.connect(self.on_window_changed)
        grid.addWidget(window_label, 2, 0)
        grid.addWidget(self.window_type, 2, 1)

        layout.addLayout(grid)

        # Averaging control in a horizontal layout
        averaging_layout = QHBoxLayout()
        averaging_layout.setSpacing(4)
        averaging_label = QLabel("Averaging:")
        self.averaging = ParameterControl(1, 32, self.config.averaging_count, decimals=0, suffix=" frames")
        self.averaging.valueChanged.connect(self.on_settings_changed)
        self.averaging.setFixedHeight(24)  # Reduced height
        averaging_layout.addWidget(averaging_label)
        averaging_layout.addWidget(self.averaging)
        layout.addLayout(averaging_layout)

        # Add decay rate control
        self.decay_group = QGroupBox("Spectrum Decay")
        decay_layout = QVBoxLayout()
        decay_layout.setSpacing(1)
        decay_layout.setContentsMargins(2, 4, 2, 2)
        
        # Add enable checkbox in a more compact layout
        decay_header = QHBoxLayout()
        decay_header.setSpacing(2)
        self.decay_enabled = QCheckBox("Enable")
        self.decay_enabled.setChecked(False)
        self.decay_enabled.stateChanged.connect(self.on_decay_enabled_changed)
        decay_header.addWidget(self.decay_enabled)
        decay_layout.addLayout(decay_header)
        
        # Add rate control with reduced height
        self.decay_rate = ParameterControl(0.01, 1.0, 0.1, decimals=2, suffix=" decay")
        self.decay_rate.setEnabled(False)
        self.decay_rate.setFixedHeight(24)
        decay_layout.addWidget(self.decay_rate)
        
        self.decay_group.setLayout(decay_layout)
        layout.addWidget(self.decay_group)

        # Add trigger controls with reduced size
        self.trigger_group = QGroupBox("Trigger")
        trigger_layout = QVBoxLayout()
        trigger_layout.setSpacing(1)
        trigger_layout.setContentsMargins(2, 4, 2, 2)
        
        # Add trigger enable checkbox in a compact header
        trigger_header = QHBoxLayout()
        trigger_header.setSpacing(2)
        self.trigger_enabled = QCheckBox("Enable")
        self.trigger_enabled.setChecked(False)
        self.trigger_enabled.stateChanged.connect(self.on_trigger_enabled_changed)
        trigger_header.addWidget(self.trigger_enabled)
        trigger_layout.addLayout(trigger_header)
        
        # Add trigger level control with reduced height
        self.trigger_level = ParameterControl(-120, 0, -45, decimals=1, suffix=" dB")
        self.trigger_level.setEnabled(False)
        self.trigger_level.setFixedHeight(24)
        trigger_layout.addWidget(self.trigger_level)
        
        # Add trigger mode controls in a grid layout
        trigger_grid = QGridLayout()
        trigger_grid.setSpacing(2)

        mode_label = QLabel("Reset:")
        self.trigger_reset_mode = QComboBox()
        self.trigger_reset_mode.addItems(["Hold Time", "Next Trigger", "Manual"])
        self.trigger_reset_mode.setEnabled(False)
        self.trigger_reset_mode.setFixedHeight(22)
        self.trigger_reset_mode.currentTextChanged.connect(self.on_trigger_reset_mode_changed)
        trigger_grid.addWidget(mode_label, 0, 0)
        trigger_grid.addWidget(self.trigger_reset_mode, 0, 1)

        edge_label = QLabel("Edge:")
        self.trigger_edge_mode = QComboBox()
        self.trigger_edge_mode.addItems(["Rising", "Falling", "Both"])
        self.trigger_edge_mode.setEnabled(False)
        self.trigger_edge_mode.setFixedHeight(22)
        self.trigger_edge_mode.currentTextChanged.connect(self.on_trigger_edge_mode_changed)
        trigger_grid.addWidget(edge_label, 1, 0)
        trigger_grid.addWidget(self.trigger_edge_mode, 1, 1)
        
        trigger_layout.addLayout(trigger_grid)

        # Add hold time control with reduced height
        self.hold_time = ParameterControl(0.05, 1.0, 0.2, decimals=2, suffix=" sec")
        self.hold_time.valueChanged.connect(self.on_hold_time_changed)
        self.hold_time.setEnabled(False)
        self.hold_time.setFixedHeight(24)
        trigger_layout.addWidget(self.hold_time)
        
        # Add manual reset button with reduced height
        self.reset_button = QPushButton("Reset Trigger")
        self.reset_button.setEnabled(False)
        self.reset_button.setFixedHeight(22)
        self.reset_button.clicked.connect(self.on_manual_reset)
        trigger_layout.addWidget(self.reset_button)
        
        self.trigger_group.setLayout(trigger_layout)
        layout.addWidget(self.trigger_group)

        # Initially hide decay and trigger controls
        self.decay_group.hide()
        self.trigger_group.hide()

        self.setLayout(layout)

    def show_test_mode_controls(self, show: bool = True):
        """Show or hide test mode specific controls (decay and trigger)"""
        if show:
            self.decay_group.show()
            self.trigger_group.show()
        else:
            # Disable controls when hiding
            if self.decay_enabled.isChecked():
                self.decay_enabled.setChecked(False)
            if self.trigger_enabled.isChecked():
                self.trigger_enabled.setChecked(False)
            self.decay_group.hide()
            self.trigger_group.hide()
            
        # No need to force layout updates or window resizing

    def on_decay_enabled_changed(self, state: int):
        """Handle decay enable/disable"""
        enabled = state == Qt.CheckState.Checked.value
        self.decay_rate.setEnabled(enabled)
        
        # Get the main window (which has the processor)
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            logger.debug(f"Setting decay enabled: {enabled}")
            main_window.processor.set_decay_enabled(enabled)
            self.on_settings_changed()
        else:
            logger.warning("Could not find processor to set decay enabled")

    def on_decay_changed(self, value: float):
        """Handle decay rate changes"""
        # Get the main window (which has the processor)
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            logger.debug(f"Setting decay rate to: {value}")
            main_window.processor.set_decay_rate(value)
            self.on_settings_changed()
        else:
            logger.warning("Could not find processor to set decay rate")

    def on_scale_changed(self, new_scale: str):
        """Explicitly handle scale changes"""
        try:
            # Update config
            self.config.scale_type = new_scale.lower()
            # Emit signal for parent to handle
            self.settings_changed.emit()
        except Exception as e:
            logger.error(f"Scale change error: {e}")

    def on_window_changed(self, window_type: str):
        """Explicitly handle window type changes"""
        try:
            # Update config
            self.config.window_type = window_type.lower()  # Convert to lowercase to match config expectations
            # Emit signal for parent to handle
            self.settings_changed.emit()
        except Exception as e:
            logger.error(f"Window change error: {e}")

    def on_settings_changed(self):
        """Signal that settings have changed"""
        logger.debug(f"Settings changed - Current FFT size: {self.config.fft_size}")
        self.settings_changed.emit()
        # Notify parent of changes
        if hasattr(self.parent(), 'mark_unsaved_changes'):
            self.parent().mark_unsaved_changes()

    def on_trigger_enabled_changed(self, state: int):
        """Handle trigger enable/disable"""
        enabled = state == Qt.CheckState.Checked.value
        self.trigger_level.setEnabled(enabled)
        self.hold_time.setEnabled(enabled)
        self.trigger_reset_mode.setEnabled(enabled)
        self.trigger_edge_mode.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            logger.debug(f"Setting trigger enabled: {enabled}")
            main_window.processor.set_trigger_enabled(enabled)
            self.on_settings_changed()

    def on_trigger_reset_mode_changed(self, mode: str):
        """Handle trigger reset mode changes"""
        # Convert UI text to processor mode
        mode_map = {
            "Hold Time": "hold_time",
            "Next Trigger": "next_trigger",
            "Manual": "manual"
        }
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            main_window.processor.set_trigger_reset_mode(mode_map[mode])
            # Show/hide hold time based on mode
            self.hold_time.setEnabled(mode == "Hold Time" and self.trigger_enabled.isChecked())
            self.on_settings_changed()

    def on_trigger_edge_mode_changed(self, mode: str):
        """Handle trigger edge mode changes"""
        # Convert UI text to processor mode
        mode_map = {
            "Rising": "rising",
            "Falling": "falling",
            "Both": "both"
        }
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            main_window.processor.set_trigger_edge_mode(mode_map[mode])
            self.on_settings_changed()

    def on_manual_reset(self):
        """Handle manual trigger reset"""
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            main_window.processor.manual_trigger_reset()

    def on_trigger_level_changed(self, value: float):
        """Handle trigger level changes"""
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            main_window.processor.set_trigger_level(value)
            self.on_settings_changed()

    def on_hold_time_changed(self, value: float):
        """Handle hold time changes"""
        main_window = self.window()
        if hasattr(main_window, 'processor'):
            main_window.processor.set_hold_time(value)
            self.on_settings_changed()

    def get_current_settings(self) -> Dict[str, Any]:
        return {
            'fft_size': int(self.fft_size.currentText()),
            'window_type': self.window_type.currentText().lower(),
            'scale_type': self.scale_type.currentText().lower(),
            'averaging_count': self.averaging.value(),
            'decay_enabled': self.decay_enabled.isChecked(),
            'decay_rate': self.decay_rate.value(),
            'trigger_enabled': self.trigger_enabled.isChecked(),
            'trigger_level': self.trigger_level.value(),
            'hold_time': self.hold_time.value(),
            'trigger_reset_mode': self.trigger_reset_mode.currentText(),
            'trigger_edge_mode': self.trigger_edge_mode.currentText()
        }

    def apply_settings(self, settings: Dict[str, Any]):
        if 'fft_size' in settings:
            self.fft_size.setCurrentText(str(settings['fft_size']))
        if 'window_type' in settings:
            index = self.window_type.findText(settings['window_type'].title())
            if index >= 0:
                self.window_type.setCurrentIndex(index)
        if 'scale_type' in settings:
            index = self.scale_type.findText(settings['scale_type'].title())
            if index >= 0:
                self.scale_type.setCurrentIndex(index)
        if 'averaging_count' in settings:
            self.averaging.setValue(settings['averaging_count'])
        if 'decay_enabled' in settings:
            self.decay_enabled.setChecked(settings['decay_enabled'])
        if 'decay_rate' in settings:
            self.decay_rate.setValue(settings['decay_rate'])
        if 'trigger_enabled' in settings:
            self.trigger_enabled.setChecked(settings['trigger_enabled'])
        if 'trigger_level' in settings:
            self.trigger_level.setValue(settings['trigger_level'])
        if 'hold_time' in settings:
            self.hold_time.setValue(settings['hold_time'])
        if 'trigger_reset_mode' in settings:
            index = self.trigger_reset_mode.findText(settings['trigger_reset_mode'])
            if index >= 0:
                self.trigger_reset_mode.setCurrentIndex(index)
        if 'trigger_edge_mode' in settings:
            index = self.trigger_edge_mode.findText(settings['trigger_edge_mode'])
            if index >= 0:
                self.trigger_edge_mode.setCurrentIndex(index)

    def on_fft_size_changed(self, size: str):
        """Handle FFT size changes"""
        try:
            new_size = int(size)
            logger.debug("FFT size changed:")
            logger.debug(f"- Old size: {self.config.fft_size}")
            logger.debug(f"- New size: {new_size}")
            self.config.fft_size = new_size
            self.on_settings_changed()
        except Exception as e:
            logger.error(f"FFT size change error: {e}")

class FilterParamDialog(QDialog):
    def __init__(self, filter_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Edit Filter Parameters")
        self.params = filter_params
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)
        self.param_widgets = {}

        # Create widgets for each parameter
        for param, value in self.params.items():
            if param == 'type':
                continue  # Skip the type parameter
            if isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setRange(0.1, 20000.0)
                widget.setValue(value)
                if 'freq' in param or 'cut' in param:
                    widget.setSuffix(" Hz")
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setRange(1, 20000)
                widget.setValue(value)
            else:
                continue

            self.param_widgets[param] = widget
            layout.addRow(f"{param.title()}:", widget)

        # Add OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self) -> dict:
        return {
            'type': self.params['type'],
            **{name: widget.value() for name, widget in self.param_widgets.items()}
        }

class FilterWidget(QFrame):
    """Individual filter widget that shows controls for a single filter"""
    parameterChanged = pyqtSignal(dict)
    removeRequested = pyqtSignal()

    def __init__(self, filter_type: str, params: dict, parent=None):
        super().__init__(parent)
        self.filter_type = filter_type
        self.params = params.copy()
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with type and remove button
        header = QHBoxLayout()
        type_label = QLabel(self.filter_type.title())
        type_label.setStyleSheet("font-weight: bold;")
        header.addWidget(type_label)
        
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.clicked.connect(self.removeRequested.emit)
        header.addWidget(remove_btn)
        layout.addLayout(header)

        # Parameter controls
        self.param_widgets = {}
        params_layout = QFormLayout()
        
        # Create appropriate controls based on filter type
        if self.filter_type in ['lowpass', 'highpass']:
            self.param_widgets['cutoff'] = ParameterControl(20.0, 20000.0, self.params.get('cutoff', 1000.0), 0, " Hz")
            self.param_widgets['order'] = ParameterControl(1, 8, self.params.get('order', 4), 0)
            self.param_widgets['gain_db'] = ParameterControl(-120.0, 0.0, self.params.get('gain_db', 0.0), 1, " dB", 0.1)
            params_layout.addRow("Cutoff:", self.param_widgets['cutoff'])
            params_layout.addRow("Order:", self.param_widgets['order'])
            params_layout.addRow("Gain:", self.param_widgets['gain_db'])

        elif self.filter_type == 'bandpass':
            # Create controls with linking
            self.param_widgets['lowcut'] = ParameterControl(
                20.0, 20000.0, self.params.get('lowcut', 100.0), 0, " Hz", 
                linked_param='lowcut'
            )
            self.param_widgets['highcut'] = ParameterControl(
                20.0, 20000.0, self.params.get('highcut', 1000.0), 0, " Hz",
                linked_param='highcut'
            )
            # Link the controls to each other
            self.param_widgets['lowcut'].linked_control = self.param_widgets['highcut']
            self.param_widgets['highcut'].linked_control = self.param_widgets['lowcut']
            self.param_widgets['order'] = ParameterControl(1, 8, self.params.get('order', 4), 0)
            self.param_widgets['gain_db'] = ParameterControl(-120.0, 0.0, self.params.get('gain_db', 0.0), 1, " dB", 0.1)
            params_layout.addRow("Low Cut:", self.param_widgets['lowcut'])
            params_layout.addRow("High Cut:", self.param_widgets['highcut'])
            params_layout.addRow("Order:", self.param_widgets['order'])
            params_layout.addRow("Gain:", self.param_widgets['gain_db'])

        elif self.filter_type == 'notch':
            self.param_widgets['freq'] = ParameterControl(20.0, 20000.0, self.params.get('freq', 1000.0), 0, " Hz")
            self.param_widgets['q'] = ParameterControl(0.1, 100.0, self.params.get('q', 30.0), 1)
            self.param_widgets['gain_db'] = ParameterControl(-120.0, 0.0, self.params.get('gain_db', 0.0), 1, " dB", 0.1)
            params_layout.addRow("Frequency:", self.param_widgets['freq'])
            params_layout.addRow("Q Factor:", self.param_widgets['q'])
            params_layout.addRow("Gain:", self.param_widgets['gain_db'])

        elif self.filter_type in ['gaussian', 'parabolic']:
            self.param_widgets['center_freq'] = ParameterControl(20.0, 20000.0, self.params.get('center_freq', 1000.0), 0, " Hz")
            self.param_widgets['width'] = ParameterControl(1.0, 5000.0, self.params.get('width', 100.0), 0, " Hz")
            self.param_widgets['gain_db'] = ParameterControl(-120.0, 0.0, self.params.get('gain_db', 0.0), 1, " dB", 0.1)
            self.param_widgets['skew'] = ParameterControl(-5.0, 5.0, self.params.get('skew', 0.0), 2)
            if self.filter_type == 'gaussian':
                self.param_widgets['kurtosis'] = ParameterControl(0.2, 5.0, self.params.get('kurtosis', 1.0), 2)
            else:  # parabolic
                self.param_widgets['flatness'] = ParameterControl(0.2, 5.0, self.params.get('flatness', 1.0), 2)
            
            params_layout.addRow("Center:", self.param_widgets['center_freq'])
            params_layout.addRow("Width:", self.param_widgets['width'])
            params_layout.addRow("Gain:", self.param_widgets['gain_db'])
            params_layout.addRow("Skew:", self.param_widgets['skew'])
            if self.filter_type == 'gaussian':
                params_layout.addRow("Kurtosis:", self.param_widgets['kurtosis'])
            else:  # parabolic
                params_layout.addRow("Flatness:", self.param_widgets['flatness'])

        elif self.filter_type == 'plateau':
            self.param_widgets['center_freq'] = ParameterControl(20.0, 20000.0, self.params.get('center_freq', 1000.0), 0, " Hz")
            self.param_widgets['width'] = ParameterControl(1.0, 10000.0, self.params.get('width', 100.0), 0, " Hz")  # Increased max to 10kHz
            self.param_widgets['flat_width'] = ParameterControl(1.0, 10000.0, self.params.get('flat_width', 50.0), 0, " Hz")  # Also increase flat width max
            self.param_widgets['gain_db'] = ParameterControl(-120.0, 0.0, self.params.get('gain_db', 0.0), 1, " dB", 0.1)
            
            params_layout.addRow("Center:", self.param_widgets['center_freq'])
            params_layout.addRow("Total Width:", self.param_widgets['width'])
            params_layout.addRow("Flat Width:", self.param_widgets['flat_width'])
            params_layout.addRow("Gain:", self.param_widgets['gain_db'])

        # Add params_layout to main layout
        layout.addLayout(params_layout)

        # Connect value changed signals
        for param, widget in self.param_widgets.items():
            widget.valueChanged.connect(lambda v, p=param: self.on_param_changed(p, v))

    def on_param_changed(self, param: str, value: float):
        self.params[param] = value
        self.parameterChanged.emit(self.params)

    def get_parameters(self) -> dict:
        return {'type': self.filter_type, **self.params}

class FilterPanel(QGroupBox):
    filter_updated = pyqtSignal(int, dict)
    filter_removed = pyqtSignal(int)
    filter_parameters = pyqtSignal(dict)

    def __init__(self, config: AudioConfig, processor=None):
        super().__init__("Filters")
        self.config = config
        self.processor = processor
        self.filters = []  # Initialize the filters list
        self.init_ui()
        # Set size policy to maintain consistent sizing
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(100)  # Set reasonable minimum height

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(4)  # Increased spacing
        main_layout.setContentsMargins(6, 8, 6, 8)  # Increased margins
        
        # Replace button layout with combo box and single add button
        add_layout = QHBoxLayout()
        add_layout.setSpacing(4)  # Increased spacing
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Plateau", "Parabolic", "Gaussian", "Bandpass", "Lowpass", "Highpass", "Notch"])
        add_layout.addWidget(self.filter_type, stretch=1)
        
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self.add_filter(self.filter_type.currentText().lower()))
        add_layout.addWidget(add_btn)
        
        main_layout.addLayout(add_layout)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container widget for filters
        self.filter_container = QWidget()
        self.filter_layout = QVBoxLayout(self.filter_container)
        self.filter_layout.setSpacing(4)  # Increased spacing between filters
        self.filter_layout.setContentsMargins(4, 4, 4, 4)  # Increased margins
        self.filter_layout.addStretch()  # Push filters to top
        
        # Add container to scroll area
        scroll.setWidget(self.filter_container)
        
        # Add scroll area to main layout, with stretch
        main_layout.addWidget(scroll, stretch=1)

    def add_filter(self, filter_type: str):
        """Add a new filter with default parameters"""
        default_params = {
            'lowpass': {'type': 'lowpass', 'cutoff': 1000},
            'highpass': {'type': 'highpass', 'cutoff': 100},
            'bandpass': {'type': 'bandpass', 'lowcut': 100, 'highcut': 1000},
            'notch': {'type': 'notch', 'freq': 1000, 'q': 30.0},
            'gaussian': {'type': 'gaussian', 'center_freq': 1000, 'width': 100, 'amplitude': 1.0},
            'parabolic': {'type': 'parabolic', 'center_freq': 1000, 'width': 100, 'amplitude': 1.0},
            'plateau': {'type': 'plateau', 'center_freq': 10000, 'width': 5000, 'flat_width': 2000, 'amplitude': 0.5}
        }

        params = default_params[filter_type]
        widget = FilterWidget(filter_type, params)
        widget.parameterChanged.connect(
            lambda p, idx=len(self.filters): self.filter_updated.emit(idx, p))
        widget.removeRequested.connect(
            lambda idx=len(self.filters): self.remove_filter(idx))
        
        # Insert before the stretch
        self.filter_layout.insertWidget(len(self.filters), widget)
        self.filters.append(widget)
        
        # Emit signal with parameters for new filter
        self.filter_parameters.emit(params)

    def remove_filter(self, index: int):
        if 0 <= index < len(self.filters):
            widget = self.filters.pop(index)
            self.filter_layout.removeWidget(widget)
            widget.deleteLater()
            self.filter_removed.emit(index)
            
            # Update remaining filters' callbacks
            for i, filter_widget in enumerate(self.filters):
                filter_widget.parameterChanged.disconnect()
                filter_widget.removeRequested.disconnect()
                filter_widget.parameterChanged.connect(
                    lambda p, idx=i: self.filter_updated.emit(idx, p))
                filter_widget.removeRequested.connect(
                    lambda idx=i: self.remove_filter(idx))

    def get_current_settings(self) -> Dict[str, Any]:
        return {
            'filters': [f.get_parameters() for f in self.filters]
        }

    def apply_settings(self, settings: Dict[str, Any]):
        """Update UI to reflect loaded filter settings without managing filters"""
        # Clear existing filter widgets
        while self.filters:
            widget = self.filters.pop(0)
            self.filter_layout.removeWidget(widget)
            widget.deleteLater()
            
        # Add filter widgets from settings
        for filter_params in settings.get('filters', []):
            filter_type = filter_params['type']
            widget = FilterWidget(filter_type, filter_params)
            widget.parameterChanged.connect(
                lambda p, idx=len(self.filters): self.filter_updated.emit(idx, p))
            widget.removeRequested.connect(
                lambda idx=len(self.filters): self.remove_filter(idx))
            self.filter_layout.insertWidget(len(self.filters), widget)
            self.filters.append(widget)
            
        logger.debug(f"DEBUG: FilterPanel apply_settings - created {len(self.filters)} filter widgets")

class ParabolaWidget(QFrame):
    """Individual parabola control widget"""
    parameterChanged = pyqtSignal(dict)
    removeRequested = pyqtSignal()

    def __init__(self, params: dict, parent=None):
        super().__init__(parent)
        self.params = params.copy()
        self.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Raised)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Header with remove button
        header = QHBoxLayout()
        type_label = QLabel(f"Spectral Component {self.params.get('id', '')}")  # Updated label
        type_label.setStyleSheet("font-weight: bold;")
        header.addWidget(type_label)
        
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.clicked.connect(self.removeRequested.emit)
        header.addWidget(remove_btn)
        layout.addLayout(header)

        # Parameter controls
        params_layout = QFormLayout()
        self.param_widgets = {}
        
        # Create controls with simplified parameters
        self.param_widgets['center_freq'] = ParameterControl(20.0, 20000.0, self.params.get('center_freq', 1000.0), 0, " Hz")
        self.param_widgets['width'] = ParameterControl(1.0, 5000.0, self.params.get('width', 100.0), 0, " Hz")
        self.param_widgets['amplitude'] = ParameterControl(0.0, 3.0, self.params.get('amplitude', 0.5), 2, "", 0.1)  # Changed max from 1.0 to 3.0
        
        params_layout.addRow("Center:", self.param_widgets['center_freq'])
        params_layout.addRow("Width:", self.param_widgets['width'])
        params_layout.addRow("Amplitude:", self.param_widgets['amplitude'])
        
        layout.addLayout(params_layout)

        # Connect signals
        for param, widget in self.param_widgets.items():
            widget.valueChanged.connect(lambda v, p=param: self.on_param_changed(p, v))

    def on_param_changed(self, param: str, value: float):
        self.params[param] = value
        self.parameterChanged.emit(self.params)

    def get_parameters(self) -> dict:
        return self.params

class SpectralComponentsPanel(QGroupBox):
    parabola_updated = pyqtSignal(int, dict)
    parabola_removed = pyqtSignal(int)
    parabola_added = pyqtSignal(dict)

    def __init__(self, processor=None):
        super().__init__("Spectral Components")
        self.processor = processor
        self.parabolas = []
        self.init_ui()
        # Set size policy to maintain consistent sizing
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMinimumHeight(100)  # Set reasonable minimum height

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(4)  # Increased spacing
        main_layout.setContentsMargins(6, 8, 6, 8)  # Increased margins

        # Add button
        add_btn = QPushButton("Add Component")
        main_layout.addWidget(add_btn)
        add_btn.clicked.connect(lambda: self.add_parabola())

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container widget for components
        self.parabola_container = QWidget()
        self.parabola_layout = QVBoxLayout(self.parabola_container)
        self.parabola_layout.setSpacing(4)  # Increased spacing
        self.parabola_layout.setContentsMargins(4, 4, 4, 4)  # Increased margins
        self.parabola_layout.addStretch()
        
        # Add container to scroll area
        scroll.setWidget(self.parabola_container)
        
        # Add scroll area to main layout, with stretch
        main_layout.addWidget(scroll, stretch=1)

    def add_parabola(self, params=None):
        """Add a new parabola component. If params is provided, use those values."""
        # Create default parameters if none provided or if called from button click
        if params is None or isinstance(params, bool):
            params = {
                'id': len(self.parabolas) + 1,
                'center_freq': 1000.0,
                'width': 100.0,
                'amplitude': 0.5,
                'slope': 1.0,
                'phase': 0.0
            }
        else:
            # Ensure ID is set when loading from settings
            if 'id' not in params:
                params['id'] = len(self.parabolas) + 1
            # Ensure required parameters have defaults
            params.setdefault('center_freq', 1000.0)
            params.setdefault('width', 100.0)
            params.setdefault('amplitude', 0.5)
            params.setdefault('slope', 1.0)
            params.setdefault('phase', 0.0)

        widget = ParabolaWidget(params)
        widget.parameterChanged.connect(
            lambda p, idx=len(self.parabolas): self.parabola_updated.emit(idx, p))
        widget.removeRequested.connect(
            lambda idx=len(self.parabolas): self.remove_parabola(idx))
        
        self.parabola_layout.insertWidget(len(self.parabolas), widget)
        self.parabolas.append(widget)
        self.parabola_added.emit(params)

    def remove_parabola(self, index: int):
        """Remove a parabola component and update the UI"""
        if 0 <= index < len(self.parabolas):
            widget = self.parabolas.pop(index)
            self.parabola_layout.removeWidget(widget)
            widget.deleteLater()
            self.parabola_removed.emit(index)
            
            # Update remaining parabolas' callbacks
            for i, parabola_widget in enumerate(self.parabolas):
                parabola_widget.parameterChanged.disconnect()
                parabola_widget.removeRequested.disconnect()
                parabola_widget.parameterChanged.connect(
                    lambda p, idx=i: self.parabola_updated.emit(idx, p))
                parabola_widget.removeRequested.connect(
                    lambda idx=i: self.remove_parabola(idx))

class StatusBar(QStatusBar):
    def __init__(self):
        super().__init__()
        self.setSizeGripEnabled(False)

class CppTemplate:
    """Represents a C++ code template with before/after sections"""
    def __init__(self, name: str, before: str = "", after: str = "", 
                 var_name: str = "audioData", length_name: str = "AUDIO_LENGTH"):
        self.name = name
        self.before = before
        self.after = after
        self.var_name = var_name
        self.length_name = length_name
        
    @classmethod
    def get_default_templates(cls) -> List['CppTemplate']:
        # Header file template (.h)
        header_template = cls(
            name="Header File",
            before="// Auto-generated audio data header\n\n",
            after="#define SAMPLE_RATE 44100\n"
                "#define @{length_name} @{length}  // Array length\n\n"
                 "// Audio samples normalized to int16 (-32768 to 32767)\n"
                 "int16_t @{var_name}[@{length_name}] = {\n"
                 "@{array_data}\n"
                 "};\n"
        )
        
        return [header_template]

class ExportDialog(QDialog):
    def __init__(self, parent=None, mode="White Noise"):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)  # Add this line to make dialog movable
        self.setWindowTitle("Export Audio")
        self.mode = mode
        self.folder_path = None  # Initialize folder_path
        
        # Get templates from parent
        parent_cpp = getattr(self.parent(), 'cpp_template', None)
        parent_carousel = getattr(self.parent(), 'carousel_template', None)
        
        logger.debug("ExportDialog init - Parent templates:")
        logger.debug("Parent cpp_template: %s", parent_cpp)
        logger.debug("Parent carousel_template: %s", parent_carousel)
        
        # Initialize templates with defaults if parent templates are None
        default_cpp_template = {
            'template_text': '// Auto-generated audio data header\n\n#define @{length_name} @{length}  // Array length\n\n// Audio samples normalized to int16 (-32768 to 32767)\nint16_t @{var_name}[@{length_name}] = {\n@{array_data}\n};\n',
            'var_name': 'audioData',
            'length_name': 'AUDIO_LENGTH'
        }
        
        default_carousel_template = config.SettingsManager().default_settings['source']['carousel_template']

        self.cpp_template = parent_cpp.copy() if parent_cpp is not None else default_cpp_template.copy()
        self.carousel_template = parent_carousel.copy() if parent_carousel is not None else default_carousel_template.copy()
        
        # Initialize export settings with templates
        self.export_settings = {
            'cpp_template': self.cpp_template,
            'carousel_template': self.carousel_template
        }
        
        logger.debug("ExportDialog init - Initialized templates:")
        logger.debug("Initial cpp_template: %s", self.cpp_template)
        logger.debug("Initial carousel_template: %s", self.carousel_template)
        
        self.filter_settings = []  # Initialize filter_settings
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # Increased spacing between groups
        layout.setContentsMargins(8, 10, 8, 10)  # Increased margins

        # Source Settings group
        source_group = QGroupBox("Source Settings")
        source_layout = QFormLayout()
        source_layout.setSpacing(4)  # Increased spacing
        source_layout.setContentsMargins(6, 8, 6, 8)  # Increased margins
        source_layout.setHorizontalSpacing(8)  # Space between labels and fields
        source_layout.setVerticalSpacing(4)  # Space between rows

        # Duration control
        self.duration = QDoubleSpinBox()
        self.duration.setRange(1.0, 3600.0)  # Changed minimum from 0.1 to 1.0
        self.duration.setValue(10.0)
        self.duration.setSuffix(" milliseconds")
        source_layout.addRow("Duration:", self.duration)

        # Base amplitude (moved here from post process)
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.0, 1.0)
        self.amplitude.setValue(self.parent().amplitude_control.value() if hasattr(self.parent(), 'amplitude_control') else 0.5)
        self.amplitude.setSingleStep(0.1)
        self.amplitude.setDecimals(2)
        self.amplitude.setSuffix("x")
        source_layout.addRow("Base Amplitude:", self.amplitude)

        # RNG Settings
        seed_layout = QHBoxLayout()
        self.use_random_seed = QCheckBox("Random Seed")
        self.use_random_seed.setChecked(True)
        self.use_random_seed.toggled.connect(self.toggle_seed_input)
        
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999999)
        self.seed_input.setValue(12345)
        self.seed_input.setEnabled(False)
        
        self.regen_seed = QPushButton("New Seed")
        self.regen_seed.clicked.connect(self.generate_random_seed)
        self.regen_seed.setEnabled(False)
        
        seed_layout.addWidget(self.use_random_seed)
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.regen_seed)
        source_layout.addRow(seed_layout)
        
        # Set the layout for source group
        source_group.setLayout(source_layout)
        layout.addWidget(source_group)  # Add to main layout

        # 2. Processing Settings group
        process_group = QGroupBox("Processing Settings")
        process_layout = QFormLayout(process_group)
        
        # Processing order selection with clearer names
        self.process_order = QComboBox()
        # Add items in order matching config default (True = "Fade then Normalize")
        self.process_order.addItems([
            "Fade then Normalize",
            "Normalize then Fade"
        ])
        process_layout.addRow("Process Order:", self.process_order)

        # Normalization settings
        self.enable_normalization = QCheckBox("Enable Normalization")
        self.enable_normalization.setChecked(self.parent().config.enable_normalization)
        self.enable_normalization.stateChanged.connect(self._update_normalization_controls)
        # Also connect to carousel normalization
        self.enable_normalization.stateChanged.connect(self._update_carousel_normalization)
        self.normalize_value = QDoubleSpinBox()
        self.normalize_value.setRange(0.0, 2.0)
        self.normalize_value.setValue(self.parent().config.normalize_value)
        self.normalize_value.setSingleStep(0.1)
        self.normalize_value.setDecimals(2)
        self.normalize_value.setSuffix("x")
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(self.enable_normalization)
        norm_layout.addWidget(self.normalize_value)
        process_layout.addRow("Normalization:", norm_layout)

        # Connect normalization checkbox to control global normalization
        self.enable_normalization.toggled.connect(self._update_normalization_controls)

        # Fade settings
        fade_group = QGroupBox("Fade Settings")
        fade_layout = QGridLayout()
        
        self.enable_fade_in = QCheckBox("Fade In")
        self.enable_fade_in.setChecked(True)  # Set checked by default
        self.fade_in = QDoubleSpinBox()
        self.fade_in.setRange(0.1, 1000.0)
        self.fade_in.setValue(1.0)
        self.fade_in.setSuffix(" ms")
        self.fade_in_power = QDoubleSpinBox()
        self.fade_in_power.setRange(0.1, 5.0)
        self.fade_in_power.setValue(2.0)
        self.fade_in_power.setSuffix("x")
        
        self.enable_fade_out = QCheckBox("Fade Out")
        self.enable_fade_out.setChecked(True)  # Set checked by default
        self.fade_out = QDoubleSpinBox()
        self.fade_out.setRange(0.1, 1000.0)
        self.fade_out.setValue(1.0)
        self.fade_out.setSuffix(" ms")
        self.fade_out_power = QDoubleSpinBox()
        self.fade_out_power.setRange(0.1, 5.0)
        self.fade_out_power.setValue(2.0)
        self.fade_out_power.setSuffix("x")

        # Grid layout for fade controls
        fade_layout.addWidget(self.enable_fade_in, 0, 0)
        fade_layout.addWidget(self.fade_in, 0, 1)
        fade_layout.addWidget(self.fade_in_power, 0, 2)
        fade_layout.addWidget(self.enable_fade_out, 1, 0)
        fade_layout.addWidget(self.fade_out, 1, 1)
        fade_layout.addWidget(self.fade_out_power, 1, 2)
        fade_group.setLayout(fade_layout)
        process_layout.addRow(fade_group)
        layout.addWidget(process_group)  # Add to main layout using addWidget

        # 3. Post Processing group
        post_group = QGroupBox("Post Processing")
        post_layout = QFormLayout(post_group)

        # Attenuation
        attn_layout = QHBoxLayout()
        self.enable_attenuation = QCheckBox("Additional Attenuation")
        self.enable_attenuation.setChecked(False)  # Explicitly set to unchecked by default
        self.attenuation = QSpinBox()
        self.attenuation.setRange(0, 96)
        self.attenuation.setValue(12)  # Set default value to 12 dB
        self.attenuation.setSuffix(" dB")
        self.attenuation.setEnabled(False)  # Explicitly disable by default
        self.enable_attenuation.toggled.connect(self.attenuation.setEnabled)
        attn_layout.addWidget(self.enable_attenuation)
        attn_layout.addWidget(self.attenuation)
        post_layout.addRow(attn_layout)
        layout.addWidget(post_group)  # Add to main layout using addWidget

        # 4. Output Settings group
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout(output_group)

        # File path controls
        folder_layout = QHBoxLayout()
        self.folder_path_label = QLabel("None")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(self.folder_path_label, stretch=1)
        folder_layout.addWidget(browse_btn)
        output_layout.addRow("Folder:", folder_layout)

        # WAV output
        wav_layout = QHBoxLayout()
        self.wav_filename = QLineEdit("noise.wav")
        self.export_wav = QCheckBox("Export WAV")
        self.export_wav.setChecked(True)
        wav_layout.addWidget(self.wav_filename, stretch=1)
        wav_layout.addWidget(self.export_wav)
        output_layout.addRow("WAV:", wav_layout)
        # Connect WAV export state changes
        self.export_wav.stateChanged.connect(self._update_wav_export_options)

        # C++ output
        cpp_layout = QHBoxLayout()
        self.cpp_filename = QLineEdit("noise.h")
        self.export_cpp = QCheckBox("Export C++")
        self.export_cpp.setChecked(True)  # Set checked by default
        cpp_layout.addWidget(self.cpp_filename, stretch=1)
        cpp_layout.addWidget(self.export_cpp)
        output_layout.addRow("C++:", cpp_layout)

        # Create tabs for template modes
        self.mode_tabs = QTabWidget()
        
        # Create both tabs first
        carousel_tab = QWidget()
        carousel_layout = QFormLayout(carousel_tab)
        
        single_tab = QWidget()
        single_layout = QFormLayout(single_tab)

        # Add carousel tab FIRST with all its content
        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 100)
        self.num_samples.setValue(20)
        carousel_layout.addRow("Number of Samples:", self.num_samples)

        # Silence duration
        self.silence_duration = QDoubleSpinBox()
        self.silence_duration.setRange(0.0, 1000.0)
        self.silence_duration.setValue(190.0)
        self.silence_duration.setSuffix(" ms")
        carousel_layout.addRow("Silence Duration:", self.silence_duration)

        # Add global normalization here instead
        self.global_norm = QCheckBox("Use Global Normalization")
        self.global_norm.setToolTip(
            "Global: Normalize across all samples\n"
            "Per-sample: Normalize each sample individually"
        )
        self.global_norm.setChecked(self.parent().config.enable_normalization)  # Use config value
        # Initially disabled if normalization is not enabled
        self.global_norm.setEnabled(self.enable_normalization.isChecked())
        carousel_layout.addRow("", self.global_norm)

        # Add template button for carousel mode
        template_btn = QPushButton("Edit Carousel Template...")
        template_btn.clicked.connect(lambda: self.edit_template(True))  # Pass True for carousel mode
        carousel_layout.addRow(template_btn)

        # Export options
        export_group = QGroupBox("WAV Export Format")  # Changed group title
        export_options = QVBoxLayout()
        self.export_combined = QCheckBox("Export Combined Sequence WAV")  # Updated label
        self.export_individual = QCheckBox("Export Individual WAV Files")  # Updated label
        self.export_combined.setChecked(True)
        export_options.addWidget(self.export_combined)
        export_options.addWidget(self.export_individual)
        export_group.setLayout(export_options)
        carousel_layout.addRow(export_group)

        # Initial state update for WAV export options
        self._update_wav_export_options(Qt.CheckState.Checked if self.export_wav.isChecked() else Qt.CheckState.Unchecked)

        self.mode_tabs.addTab(carousel_tab, "Carousel Mode")
        
        # Add content to single file tab
        single_tab = QWidget()
        single_layout = QFormLayout(single_tab)
        cpp_template_btn = QPushButton("Edit C++ Template...")
        cpp_template_btn.clicked.connect(lambda: self.edit_template(False))  # Pass False for single file mode
        single_layout.addRow(cpp_template_btn)
        single_tab.setLayout(single_layout)
        self.mode_tabs.addTab(single_tab, "Single File")

        # Set Carousel tab as default
        self.mode_tabs.setCurrentIndex(0)

        output_layout.addWidget(self.mode_tabs)
        layout.addWidget(output_group)

        # Final buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)  # Add to main layout using addWidget

        # Initial validation
        self.validate_export_options()

    def _update_fade_controls(self, which: str, enabled: bool):
        """Enable/disable fade controls based on checkbox"""
        if which == 'in':
            self.fade_in.setEnabled(enabled)
            self.fade_in_power.setEnabled(enabled)
        else:
            self.fade_out.setEnabled(enabled)
            self.fade_out_power.setEnabled(enabled)

    def validate_export_options(self):
        """Enable OK button only if at least one export option is selected"""
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setEnabled(self.export_wav.isChecked() or self.export_cpp.isChecked())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if (folder):
            self.folder_path = folder
            # Show abbreviated path if too long
            max_length = 40
            display_path = folder
            if len(folder) > max_length:
                display_path = "..." + folder[-(max_length-3):]
            self.folder_path_label.setText(display_path)

    def validate_and_accept(self):
        """Validate settings before accepting"""
        if not self.folder_path:
            QMessageBox.warning(self, "Error", "Please select an output folder")
            return

        # Add correct extensions if not present
        if self.export_wav.isChecked():
            wav_name = self.wav_filename.text()
            if not wav_name.lower().endswith('.wav'):
                self.wav_filename.setText(wav_name + '.wav')

        if self.export_cpp.isChecked():
            cpp_name = self.cpp_filename.text()
            if not cpp_name.lower().endswith(('.cpp', '.h')):
                self.cpp_filename.setText(cpp_name + '.h')

        # Validate fade times against total duration
        total_fade = 0
        if self.enable_fade_in.isChecked():
            total_fade += self.fade_in.value()
        if self.enable_fade_out.isChecked():
            total_fade += self.fade_out.value()
            
        if (total_fade >= self.duration.value()):
            QMessageBox.warning(self, "Error", 
                "Total fade duration cannot be larger than output duration.\n"
                f"Current fade: {total_fade}ms, Duration: {self.duration.value()}ms")
            return

        self.accept()

    def toggle_seed_input(self, random_seed: bool):
        """Toggle seed input and regen button based on checkbox"""
        self.seed_input.setEnabled(not random_seed)  # Fix: use 'not' instead of '!'
        self.regen_seed.setEnabled(not random_seed)

    def generate_random_seed(self):
        """Generate a new random seed"""
        self.seed_input.setValue(np.random.randint(0, 1000000000))

    def get_settings(self) -> dict:
        """Get export settings with fixed sample rate and calculated amplitude"""
        try:
            logger.debug("Starting get_settings")
            self._ensure_file_extensions()
            basic_settings = self._get_basic_settings()
            logger.debug("Got basic settings: %s", basic_settings)
            
            file_settings = self._get_file_settings()
            logger.debug("Got file settings: %s", file_settings)
            
            carousel_settings = self._get_carousel_settings()
            logger.debug("Got carousel settings: %s", carousel_settings)
            
            filter_settings = self._get_filter_settings()
            logger.debug("Got filter settings: %s", filter_settings)
            
            settings = {
                **basic_settings,
                **file_settings,
                **carousel_settings,
                'filters': filter_settings,
                'source_type': self.mode
            }
            logger.debug("Returning combined settings: %s", settings)
            return settings
        except Exception as e:
            logger.error("Error in get_settings: %s", e, exc_info=True)
            raise

    def _ensure_file_extensions(self):
        """Ensure proper file extensions for export files"""
        if self.export_wav.isChecked() and not self.wav_filename.text().lower().endswith('.wav'):
            self.wav_filename.setText(self.wav_filename.text() + '.wav')
        if self.export_cpp.isChecked() and not self.cpp_filename.text().lower().endswith(('.cpp', '.h')):
            self.cpp_filename.setText(self.cpp_filename.text() + '.h')

    def _get_basic_settings(self) -> dict:
        """Get basic export settings including amplitude and processing"""
        base_amplitude = self.amplitude.value()
        try:
            if self.enable_attenuation.isChecked():
                attn_db = self.attenuation.value()
                final_amplitude = base_amplitude * (10 ** (-attn_db / 20))
                final_amplitude = max(final_amplitude, 1e-10)  # Ensure non-zero
            else:
                final_amplitude = base_amplitude
        except Exception as e:
            logger.error(f"Error calculating amplitude: {e}")
            final_amplitude = 1e-10  # Safe fallback

        return {
            'duration': self.duration.value() / 1000.0,  # Convert ms to seconds
            'sample_rate': 44100,
            'base_amplitude': base_amplitude,
            'amplitude': final_amplitude,
            'attenuation': self.attenuation.value(),
            'enable_attenuation': self.enable_attenuation.isChecked(),
            'export_wav': self.export_wav.isChecked(),
            'export_cpp': self.export_cpp.isChecked(),
            'enable_fade_in': self.enable_fade_in.isChecked(),
            'enable_fade_out': self.enable_fade_out.isChecked(),
            'enable_fade': self.enable_fade_in.isChecked() or self.enable_fade_out.isChecked(),  # Backward compatibility
            'fade_in_duration': self.fade_in.value() / 1000.0 if self.enable_fade_in.isChecked() else 0,
            'fade_out_duration': self.fade_out.value() / 1000.0 if self.enable_fade_out.isChecked() else 0,
            'fade_in_power': self.fade_in_power.value(),
            'fade_out_power': self.fade_out_power.value(),
            'enable_normalization': self.enable_normalization.isChecked(),
            'normalize_value': self.normalize_value.value(),
            'fade_before_norm': self.process_order.currentText() == "Fade then Normalize",
            'use_random_seed': self.use_random_seed.isChecked(),
            'seed': self.seed_input.value() if not self.use_random_seed.isChecked() else None,
            'rng_type': self.parent().rng_type.currentText().lower().replace(' ', '_')
        }

    def _get_file_settings(self) -> dict:
        """Get file-related settings including paths and templates"""
        cpp_name = self.cpp_filename.text()
        base_name = os.path.splitext(cpp_name)[0]
        return {
            'folder_path': self.folder_path,
            'wav_filename': self.wav_filename.text(),
            'cpp_filename': self.cpp_filename.text(),
            'header_filename': base_name + '.h',
            'cpp_template': self.cpp_template,
            'carousel_template': self.carousel_template
        }

    def _get_filter_settings(self) -> list:
        """Get current filter settings from parent's filter panel"""
        if hasattr(self.parent(), 'filter_panel'):
            self.filter_settings = self.parent().filter_panel.get_current_settings().get('filters', [])
        return self.filter_settings

    def _get_carousel_settings(self) -> dict:
        """Get carousel-specific settings"""
        is_carousel_tab = self.mode_tabs.currentIndex() == 0
        return {
            'carousel_enabled': is_carousel_tab,
            'carousel_samples': (
                self.num_samples.value() if hasattr(self, 'num_samples') 
                else self.saved_settings.get('carousel_samples', self.DEFAULT_CAROUSEL_SETTINGS['carousel_samples'])
            ),
            'carousel_noise_duration_ms': self.duration.value(),
            'silence_duration_ms': (
                self.silence_duration.value() if hasattr(self, 'silence_duration') 
                else self.saved_settings.get('silence_duration_ms', self.DEFAULT_CAROUSEL_SETTINGS['silence_duration_ms'])
            ),
            'export_combined': (
                self.export_combined.isChecked() if hasattr(self, 'export_combined') 
                else self.saved_settings.get('export_combined', self.DEFAULT_CAROUSEL_SETTINGS['export_combined'])
            ),
            'export_individual': (
                self.export_individual.isChecked() if hasattr(self, 'export_individual') 
                else self.saved_settings.get('export_individual', self.DEFAULT_CAROUSEL_SETTINGS['export_individual'])
            ),
            'global_normalization': (
                self.global_norm.isChecked() if hasattr(self, 'global_norm') 
                else self.saved_settings.get('global_normalization', self.DEFAULT_CAROUSEL_SETTINGS['global_normalization'])
            )
        }

    def edit_template(self, is_carousel: bool):
        """Open appropriate template editor based on mode"""
        try:
            if is_carousel:
                dialog = CarouselTemplateDialog(self.carousel_template, self)
                dialog.template_changed.connect(self.update_carousel_template)
            else:
                dialog = CppTemplateDialog(self.cpp_template, self)
                dialog.template_changed.connect(self.update_cpp_template)
            dialog.exec()
        except Exception as e:
            logger.error(f"Template dialog error: {e}")

    def update_cpp_template(self, template: dict):
        """Handle regular cpp template changes"""
        logger.debug("Updating cpp template:")
        logger.debug("Old template: %s", self.cpp_template)
        logger.debug("New template: %s", template)
        self.cpp_template = template
        self.export_settings['cpp_template'] = template  # Update export settings
        if hasattr(self.parent(), 'cpp_template'):
            self.parent().cpp_template = template  # Update parent's template too
            logger.debug("Updated parent's cpp_template: %s", self.parent().cpp_template)

    def update_carousel_template(self, template: dict):
        """Handle carousel template changes"""
        logger.debug("Updating carousel template:")
        logger.debug("Old template: %s", self.carousel_template)
        logger.debug("New template: %s", template)
        self.carousel_template = template
        self.export_settings['carousel_template'] = template  # Update export settings
        if hasattr(self.parent(), 'carousel_template'):
            self.parent().carousel_template = template
            logger.debug("Updated parent's carousel_template: %s", self.parent().carousel_template)

    def apply_saved_settings(self, settings: Dict[str, Any]):
        """Apply previously saved settings to dialog controls"""
        try:
            if not settings:
                logger.debug("No settings provided to apply_saved_settings")
                return

            logger.debug("Applying saved settings: %s", settings)

            # Duration
            if 'duration' in settings:
                logger.debug("Setting duration: %s", settings['duration'])
                self.duration.setValue(settings['duration'] * 1000.0)  # Convert seconds back to milliseconds
            
            # Amplitude/Attenuation
            if 'base_amplitude' in settings:
                logger.debug("Setting base_amplitude: %s", settings['base_amplitude'])
                self.amplitude.setValue(settings['base_amplitude'])
            elif 'amplitude' in settings:
                logger.debug("Setting amplitude: %s", settings['amplitude'])
                self.amplitude.setValue(settings['amplitude'])

            if 'enable_attenuation' in settings:
                logger.debug("Setting enable_attenuation: %s", settings['enable_attenuation'])
                self.enable_attenuation.setChecked(settings['enable_attenuation'])
            if 'attenuation' in settings:
                logger.debug("Setting attenuation: %s", settings['attenuation'])
                self.attenuation.setValue(settings['attenuation'])
                self.attenuation.setEnabled(settings.get('enable_attenuation', False))
                    
            # Process order
            if 'fade_before_norm' in settings:
                logger.debug("Setting fade_before_norm: %s", settings['fade_before_norm'])
                self.process_order.setCurrentText("Fade then Normalize" if settings['fade_before_norm'] else "Normalize then Fade")
                    
            # Fade settings    
            if 'fade_in_duration' in settings:
                logger.debug("Setting fade_in_duration: %s", settings['fade_in_duration'])
                self.fade_in.setValue(settings['fade_in_duration'] * 1000.0)
            if 'fade_out_duration' in settings:
                logger.debug("Setting fade_out_duration: %s", settings['fade_out_duration'])
                self.fade_out.setValue(settings['fade_out_duration'] * 1000.0)
            if 'fade_in_power' in settings:
                logger.debug("Setting fade_in_power: %s", settings['fade_in_power'])
                self.fade_in_power.setValue(settings['fade_in_power'])
            if 'fade_out_power' in settings:
                logger.debug("Setting fade_out_power: %s", settings['fade_out_power'])
                self.fade_out_power.setValue(settings['fade_out_power'])
            if 'enable_fade_in' in settings:
                logger.debug("Setting enable_fade_in: %s", settings['enable_fade_in'])
                self.enable_fade_in.setChecked(settings['enable_fade_in'])
            if 'enable_fade_out' in settings:
                logger.debug("Setting enable_fade_out: %s", settings['enable_fade_out'])
                self.enable_fade_out.setChecked(settings['enable_fade_out'])
            elif 'enable_fade' in settings:  # Backward compatibility
                logger.debug("Setting enable_fade (backward compatibility): %s", settings['enable_fade'])
                self.enable_fade_in.setChecked(settings['enable_fade'])
                self.enable_fade_out.setChecked(settings['enable_fade'])
                
            # Normalization
            if 'enable_normalization' in settings:
                logger.debug("Setting enable_normalization: %s", settings['enable_normalization'])
                self.enable_normalization.setChecked(settings['enable_normalization'])
            if 'normalize_value' in settings:
                logger.debug("Setting normalize_value: %s", settings['normalize_value'])
                self.normalize_value.setValue(settings['normalize_value'])
                
            # File settings
            if 'folder_path' in settings:
                logger.debug("Setting folder_path: %s", settings['folder_path'])
                self.folder_path = settings['folder_path']
                self.folder_path_label.setText(settings['folder_path'])
            if 'wav_filename' in settings:
                logger.debug("Setting wav_filename: %s", settings['wav_filename'])
                self.wav_filename.setText(settings['wav_filename'])
            if 'cpp_filename' in settings:
                logger.debug("Setting cpp_filename: %s", settings['cpp_filename'])
                self.cpp_filename.setText(settings['cpp_filename'])

            # Export options
            if 'export_wav' in settings:
                logger.debug("Setting export_wav: %s", settings['export_wav'])
                self.export_wav.setChecked(settings['export_wav'])
            if 'export_cpp' in settings:
                logger.debug("Setting export_cpp: %s", settings['export_cpp'])
                self.export_cpp.setChecked(settings['export_cpp'])
                
            # Seed settings
            if 'use_random_seed' in settings:
                logger.debug("Setting use_random_seed: %s", settings['use_random_seed'])
                self.use_random_seed.setChecked(settings['use_random_seed'])
            if 'seed' in settings and settings['seed'] is not None:
                logger.debug("Setting seed: %s", settings['seed'])
                self.seed_input.setValue(settings['seed'])

            # Always apply carousel settings regardless of current tab
            if hasattr(self, 'num_samples'):
                logger.debug("Setting carousel_samples: %s", settings.get('carousel_samples', 20))
                self.num_samples.setValue(settings.get('carousel_samples', 20))
            if hasattr(self, 'silence_duration'):
                logger.debug("Setting silence_duration_ms: %s", settings.get('silence_duration_ms', 190.0))
                self.silence_duration.setValue(settings.get('silence_duration_ms', 190.0))
            if hasattr(self, 'export_combined'):
                logger.debug("Setting export_combined: %s", settings.get('export_combined', True))
                self.export_combined.setChecked(settings.get('export_combined', True))
            if hasattr(self, 'export_individual'):
                logger.debug("Setting export_individual: %s", settings.get('export_individual', False))
                self.export_individual.setChecked(settings.get('export_individual', False))
            if hasattr(self, 'global_norm'):
                logger.debug("Setting global_normalization: %s", settings.get('global_normalization', True))
                self.global_norm.setChecked(settings.get('global_normalization', True))
                self.global_norm.setEnabled(settings.get('enable_normalization', True))

            # Set the tab after applying all settings
            logger.debug("Setting tab index: %s", 0 if settings.get('carousel_enabled', False) else 1)
            self.mode_tabs.setCurrentIndex(0 if settings.get('carousel_enabled', False) else 1)

        except Exception as e:
            logger.error("Error applying saved settings: %s", e, exc_info=True)  # Added exc_info for full traceback

    def _update_normalization_controls(self, enabled: bool):
        """Update controls that depend on normalization being enabled"""
        self.normalize_value.setEnabled(enabled)
        if hasattr(self, 'global_norm'):
            self.global_norm.setEnabled(enabled)

    def _update_carousel_normalization(self, state: int):
        """Update carousel normalization option based on global normalization state"""
        enabled = state == Qt.CheckState.Checked.value
        if hasattr(self, 'global_norm'):
            self.global_norm.setEnabled(enabled)
            if not enabled:
                self.global_norm.setChecked(False)

    def _update_wav_export_options(self, state: int):
        """Update carousel WAV export options based on global WAV export state"""
        enabled = bool(state)  # Convert Qt.CheckState to boolean
        if hasattr(self, 'export_combined'):
            self.export_combined.setEnabled(enabled)
        if hasattr(self, 'export_individual'):
            self.export_individual.setEnabled(enabled)

# In other panel classes (AnalyzerPanel, SourcePanel, FilterPanel)
# Add change notification to parameter changes:

def on_parameter_changed(self):
    """Call when any parameter changes"""
    if hasattr(self.parent(), 'mark_unsaved_changes'):
        self.parent().mark_unsaved_changes()

class OverlayTemplate:
    # Symbol mapping for display - shared by all overlay-related classes
    SYMBOLS = {
        'x': '×',  # X (filled)
        'o': '○',  # Hollow circle
        's': '□',  # Hollow square
        't': '▽',  # Hollow triangle
        '+': '+'   # Plus (filled)
    }

    def __init__(self, name: str, color: str, points: List[Tuple[float, float]], 
                 interpolation: str = "linear", symbol: str = 'o'):
        self.name = name
        self.color = color
        self.points = points
        self.enabled = True
        self.offset = 0
        self.interpolation = interpolation  # Add interpolation type
        self.symbol = symbol  # Add symbol type

class OverlayEditDialog(QDialog):
    template_changed = pyqtSignal(object)  # Add signal for real-time updates
    
    def __init__(self, parent=None, template=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.template = template
        self.has_changes = False
        self.init_ui()
        if self.template:
            self.load_template(self.template)  # Load existing template data
            
        # Connect signals for real-time updates
        self.name_edit.textChanged.connect(self.on_change)
        self.color_combo.currentIndexChanged.connect(self.on_change)
        self.symbol_combo.currentIndexChanged.connect(self.on_change)
        self.interp_combo.currentIndexChanged.connect(self.on_change)

    def on_change(self, *args):
        """Handle any change in the dialog"""
        self.has_changes = True
        self.emit_template_change()

    def reject(self):
        """Handle dialog cancellation"""
        if self.has_changes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to discard them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        # Clear preview template and emit change to update display
        if hasattr(self.parent(), 'preview_template'):
            self.parent().preview_template = None
            self.parent().overlay_changed.emit()
        super().reject()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Template name
        name_layout = QHBoxLayout()
        self.name_edit = QLineEdit()
        name_layout.addWidget(QLabel("Name:"))
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Color selection
        color_layout = QHBoxLayout()
        self.color_combo = QComboBox()
        colors = [('#ff0000', 'Red'), ('#00ff00', 'Green'), 
                 ('#0000ff', 'Blue'), ('#ff00ff', 'Magenta'), 
                 ('#00ffff', 'Cyan')]
        for code, name in colors:
            self.color_combo.addItem(name, code)
        color_layout.addWidget(QLabel("Color:"))
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        # Add symbol selector with just the symbols shown
        symbol_layout = QHBoxLayout()
        self.symbol_combo = QComboBox()
        # Use symbols from OverlayTemplate
        for code, display in OverlayTemplate.SYMBOLS.items():
            self.symbol_combo.addItem(display, code)
        symbol_layout.addWidget(QLabel("Symbol:"))
        symbol_layout.addWidget(self.symbol_combo)
        layout.addLayout(symbol_layout)

        # Add interpolation selector
        interp_layout = QHBoxLayout()
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(['Linear', 'Cubic', 'Akima'])
        interp_layout.addWidget(QLabel("Interpolation:"))
        interp_layout.addWidget(self.interp_combo)
        layout.addLayout(interp_layout)

        # Points editor
        points_group = QGroupBox("Points")
        points_layout = QVBoxLayout(points_group)
        
        # Points list
        self.points_list = QListWidget()
        self.points_list.itemDoubleClicked.connect(self.edit_point)
        points_layout.addWidget(self.points_list)
        
        # Point control buttons
        point_buttons = QHBoxLayout()
        add_point = QPushButton("Add")
        edit_point = QPushButton("Edit")
        remove_point = QPushButton("Remove")
        
        add_point.clicked.connect(self.add_new_point)
        edit_point.clicked.connect(lambda: self.edit_point(self.points_list.currentItem()))
        remove_point.clicked.connect(self.remove_point)
        
        point_buttons.addWidget(add_point)
        point_buttons.addWidget(edit_point)
        point_buttons.addWidget(remove_point)
        points_layout.addLayout(point_buttons)
        
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # Different button configurations for new vs edit
        button_layout = QHBoxLayout()
        if self.template:  # Editing existing
            save_btn = QPushButton("Save Changes")
            save_as_btn = QPushButton("Save as New")
            cancel_btn = QPushButton("Cancel")
            
            save_btn.clicked.connect(self.accept)
            save_as_btn.clicked.connect(lambda: self.done(2))  # Custom code for save as new
            cancel_btn.clicked.connect(self.reject)
            
            button_layout.addWidget(save_btn)
            button_layout.addWidget(save_as_btn)
        else:  # New template
            ok_btn = QPushButton("Create")
            cancel_btn = QPushButton("Cancel")
            ok_btn.clicked.connect(self.accept)
            cancel_btn.clicked.connect(self.reject)
            
            button_layout.addWidget(ok_btn)
            button_layout.addWidget(cancel_btn)  # Add cancel button to layout
        
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

    def emit_template_change(self):
        """Emit signal with current template state"""
        temp = self.get_template()
        if self.template:  # Editing existing template
            temp.enabled = self.template.enabled  # Preserve enabled state
            temp.offset = self.template.offset    # Preserve offset
        self.template_changed.emit(temp)

    def add_new_point(self):
        """Open edit dialog for new point"""
        while True:  # Keep dialog open until valid point or cancel
            dialog = PointEditDialog(parent=self, is_add=True)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                freq, level = dialog.get_values()
                
                # Check for duplicate frequency
                duplicate = False
                for i in range(self.points_list.count()):
                    other_freq = self.points_list.item(i).data(Qt.ItemDataRole.UserRole)[0]
                    if abs(other_freq - freq) < 0.1:
                        QMessageBox.warning(self, "Error", 
                            "A point at this frequency already exists!")
                        duplicate = True
                        break
                
                if not duplicate:
                    # Add new point and sort
                    item = QListWidgetItem(f"{freq} Hz, {level} dB")
                    item.setData(Qt.ItemDataRole.UserRole, (freq, level))
                    self.points_list.addItem(item)
                    self.sort_points()
                    self.has_changes = True  # Mark as changed when adding point
                    self.emit_template_change()  # Emit after adding point
                    break  # Exit loop on successful add
            else:
                break  # User cancelled

    def remove_point(self):
        current = self.points_list.currentRow()
        if current >= 0:
            self.points_list.takeItem(current)
            self.has_changes = True  # Mark as changed when removing point
            self.emit_template_change()  # Emit after removing point

    def edit_point(self, item):
        """Edit an existing point"""
        if not item:
            return
            
        freq, level = item.data(Qt.ItemDataRole.UserRole)
        dialog = PointEditDialog(parent=self, freq=freq, level=level)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_freq, new_level = dialog.get_values()
            
            # Check for duplicate frequency (excluding current point)
            for i in range(self.points_list.count()):
                other_item = self.points_list.item(i)
                if other_item != item:  # Skip current item
                    other_freq = other_item.data(Qt.ItemDataRole.UserRole)[0]
                    if abs(other_freq - new_freq) < 0.1:
                        QMessageBox.warning(self, "Error", 
                            "A point at this frequency already exists!")
                        return
            
            # Update point
            item.setText(f"{new_freq} Hz, {new_level} dB")
            item.setData(Qt.ItemDataRole.UserRole, (new_freq, new_level))
            self.sort_points()
            self.has_changes = True  # Mark as changed when editing point
            self.emit_template_change()  # Emit after editing point

    def get_template(self) -> OverlayTemplate:
        return OverlayTemplate(
            name=self.name_edit.text(),
            color=self.color_combo.currentData(),
            points=self.get_points(),
            interpolation=self.interp_combo.currentText().lower(),
            symbol=self.symbol_combo.currentData()
        )

    def load_template(self, template: OverlayTemplate):
        self.name_edit.setText(template.name)
        index = self.color_combo.findData(template.color)
        if index >= 0:
            self.color_combo.setCurrentIndex(index)
        
        # Set symbol
        symbol_index = self.symbol_combo.findData(template.symbol)
        if symbol_index >= 0:
            self.symbol_combo.setCurrentIndex(symbol_index)
        
        self.points_list.clear()
        for freq, level in template.points:
            item = QListWidgetItem(f"{freq} Hz, {level} dB")
            item.setData(Qt.ItemDataRole.UserRole, (freq, level))
            self.points_list.addItem(item)
        
        # Set interpolation
        index = self.interp_combo.findText(template.interpolation.title())
        if index >= 0:
            self.interp_combo.setCurrentIndex(index)

    def sort_points(self):
        """Sort points by frequency"""
        points = []
        for i in range(self.points_list.count()):
            item = self.points_list.item(i)
            points.append((item.data(Qt.ItemDataRole.UserRole), item.text()))
        
        points.sort(key=lambda x: x[0][0])  # Sort by frequency
        
        self.points_list.clear()
        for (data, text) in points:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, data)
            self.points_list.addItem(item)

    def get_points(self) -> List[Tuple[float, float]]:
        points = []
        for i in range(self.points_list.count()):
            item = self.points_list.item(i)
            points.append(item.data(Qt.ItemDataRole.UserRole))
        return sorted(points, key=lambda x: x[0])  # Sort by frequency

class OverlayManager(QGroupBox):
    overlay_changed = pyqtSignal()  # For real-time preview updates
    overlay_confirmed = pyqtSignal()  # For confirmed changes that should mark profile as dirty
    
    def __init__(self, parent=None):
        super().__init__("Overlay Templates", parent)
        self.templates = []
        self.max_overlays = 5
        self.colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff']
        self.preview_template = None  # Add storage for preview template
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create scroll area for template list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # Create container widget for templates
        template_container = QWidget()
        self.template_layout = QVBoxLayout(template_container)
        scroll.setWidget(template_container)
        layout.addWidget(scroll)
        
        # Add button
        add_btn = QPushButton("Add Overlay")
        add_btn.clicked.connect(self.add_template)
        layout.addWidget(add_btn)

    def update_list(self):
        """Update the list of templates with controls"""
        # Clear existing items
        while self.template_layout.count():
            item = self.template_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add items with controls
        for i, template in enumerate(self.templates):
            item = QWidget()
            item_layout = QHBoxLayout(item)
            item_layout.setContentsMargins(2, 2, 2, 2)
            item_layout.setSpacing(4)
            
            # Enable checkbox
            enable_check = QCheckBox()
            enable_check.setChecked(template.enabled)
            enable_check.toggled.connect(lambda checked, t=template: self.toggle_template(t, checked))
            item_layout.addWidget(enable_check)
            
            # Add symbol display
            symbol_label = QLabel(OverlayTemplate.SYMBOLS.get(template.symbol, '○'))  # Default to circle if symbol not found
            symbol_label.setFixedWidth(20)
            symbol_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            item_layout.addWidget(symbol_label)
            
            # Template name
            name_label = QLabel(template.name)
            item_layout.addWidget(name_label, stretch=1)
            
            # Offset control with smaller width
            offset_spin = QSpinBox()
            offset_spin.setRange(-150, 150)  # Changed from ±50 to ±150
            offset_spin.setValue(template.offset)
            offset_spin.setSuffix(" dB")
            offset_spin.setFixedWidth(70)  # Make spinner more compact
            offset_spin.valueChanged.connect(lambda value, t=template: self.update_offset(t, value))
            item_layout.addWidget(offset_spin)

            # Invert values button
            invert_btn = QPushButton("±")
            invert_btn.setFixedSize(24, 24)
            invert_btn.setToolTip("Invert Values")
            invert_btn.clicked.connect(lambda _, t=template: self.invert_values(t))
            item_layout.addWidget(invert_btn)

            # Control buttons with symbols
            edit_btn = QPushButton("✎")
            edit_btn.setFixedSize(24, 24)
            edit_btn.setToolTip("Edit")
            edit_btn.clicked.connect(lambda _, idx=i: self.edit_template(idx))
            item_layout.addWidget(edit_btn)

            del_btn = QPushButton("×")
            del_btn.setFixedSize(24, 24)
            del_btn.setToolTip("Delete")
            del_btn.clicked.connect(lambda _, idx=i: self.confirm_remove(idx))
            item_layout.addWidget(del_btn)
            
            self.template_layout.addWidget(item)
        
        self.template_layout.addStretch()

    def invert_values(self, template):
        """Invert all dB values in the template"""
        template.points = [(freq, -level) for freq, level in template.points]
        self.overlay_changed.emit()
        self.overlay_confirmed.emit()  # This is a direct user action, so confirm it

    def toggle_template(self, template, enabled):
        template.enabled = enabled
        self.overlay_changed.emit()
        self.overlay_confirmed.emit()  # This is a direct user action, so confirm it

    def update_offset(self, template, offset):
        template.offset = offset
        self.overlay_changed.emit()
        self.overlay_confirmed.emit()  # This is a direct user action, so confirm it

    def add_template(self):
        """Add a new overlay template"""
        # Create default points with standard frequencies
        default_points = [
            (250, 0), (500, 0), (1000, 0), (2000, 0), (3000, 0),
            (4000, 0), (6000, 0), (8000, 0), (9000, 0), (10000, 0),
            (11200, 0), (12500, 0), (14000, 0), (16000, 0)
        ]
        
        # Create template with default points
        template = OverlayTemplate(
            name=f"Overlay {len(self.templates) + 1}",
            color=f"#{hash(time.time()) % 0xFFFFFF:06x}",  # Random color
            points=default_points,
            interpolation="linear",
            symbol="o"
        )
        
        dialog = OverlayEditDialog(self, template)
        # Connect template_changed signal for preview
        dialog.template_changed.connect(self._preview_new_template)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            template = dialog.get_template()
            self.templates.append(template)
            self.preview_template = None  # Clear preview
            self.update_list()
            self.overlay_changed.emit()
            self.overlay_confirmed.emit()  # Template was added, confirm the change
        else:
            self.preview_template = None  # Clear preview on cancel
            self.overlay_changed.emit()

    def _preview_new_template(self, template: OverlayTemplate):
        """Handle preview of new template being created"""
        self.preview_template = template
        self.overlay_changed.emit()

    def edit_template(self, index):
        """Edit an existing overlay template"""
        if 0 <= index < len(self.templates):
            # Store the original template for restoration on cancel
            original_template = self.templates[index]
            
            # Create a copy of the template for editing
            temp_template = OverlayTemplate(
                name=original_template.name,
                color=original_template.color,
                points=original_template.points.copy(),
                interpolation=original_template.interpolation,
                symbol=original_template.symbol
            )
            temp_template.enabled = original_template.enabled
            temp_template.offset = original_template.offset
            
            dialog = OverlayEditDialog(self, temp_template)
            # Connect template_changed signal to update graph in real-time
            dialog.template_changed.connect(lambda temp: self._handle_template_change(index, temp))
            result = dialog.exec()
            
            if result == QDialog.DialogCode.Accepted:
                new_template = dialog.get_template()
                new_template.offset = original_template.offset  # Preserve offset
                self.templates[index] = new_template
                self.overlay_confirmed.emit()  # Template was edited, confirm the change
            elif result == 2:  # Save as new
                if len(self.templates) < self.max_overlays:
                    # Create new template from dialog
                    new_template = dialog.get_template()
                    # Restore original template
                    self.templates[index] = original_template
                    # Add new template to list
                    self.templates.append(new_template)
                    self.overlay_confirmed.emit()  # New template was added, confirm the change
                else:
                    QMessageBox.warning(self, "Error", "Maximum number of overlays reached")
            else:  # Cancelled - restore original template
                self.templates[index] = original_template
            
            self.update_list()
            self.overlay_changed.emit()

    def duplicate_template(self, index):
        if 0 <= index < len(self.templates) and len(self.templates) < self.max_overlays:
            template = self.templates[index]
            new_template = OverlayTemplate(
                name=f"{template.name} (copy)",
                color=template.color,
                points=template.points.copy()
            )
            self.templates.append(new_template)
            self.update_list()
            self.overlay_changed.emit()

    def confirm_remove(self, index):
        """Confirm before removing template"""
        reply = QMessageBox.question(
            self, "Confirm Delete",
            "Are you sure you want to delete this overlay?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.remove_template(index)

    def remove_template(self, index):
        if 0 <= index < len(self.templates):
            self.templates.pop(index)
            self.update_list()
            self.overlay_changed.emit()
            self.overlay_confirmed.emit()  # Template was removed, confirm the change

    def get_templates(self):
        templates = self.templates.copy()
        if self.preview_template:
            templates.append(self.preview_template)
        return templates

    def _handle_template_change(self, index: int, template: OverlayTemplate):
        """Handle real-time template updates during editing"""
        if 0 <= index < len(self.templates):
            self.templates[index] = template
            self.overlay_changed.emit()  # Only emit changed for preview, not confirmed yet

class CppTemplateDialog(QDialog):
    template_changed = pyqtSignal(dict)  # Add signal

    def __init__(self, template_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("C++ Template Editor")
        self.template_data = template_data
        # Set larger default size instead of minimum
        self.resize(400, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Set minimum size for better visibility
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        # Help text
        help_text = QLabel(
            "Define the template for generating C++ code.\n"
            "Available placeholders: @{var_name}, @{length_name}, @{length}, @{array_data}"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Template text editor
        self.template_edit = QPlainTextEdit()
        layout.addWidget(QLabel("Template:"))
        layout.addWidget(self.template_edit)

        # Basic variables section
        var_group = QGroupBox("Standard Variables")
        var_layout = QFormLayout(var_group)

        # Variable name
        self.var_name = QLineEdit(self.template_data.get('var_name', 'audioData'))
        var_layout.addRow("Array Name:", self.var_name)

        # Length name
        self.length_name = QLineEdit(self.template_data.get('length_name', 'AUDIO_LENGTH'))
        var_layout.addRow("Length Name:", self.length_name)

        layout.addWidget(var_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Load current template
        self.template_edit.setPlainText(self.template_data.get('template_text', ''))

    def accept(self):
        """Override accept to emit template change"""
        template = self.get_template()
        self.template_changed.emit(template)
        super().accept()

    def get_template(self) -> dict:
        return {
            'template_text': self.template_edit.toPlainText(),
            'var_name': self.var_name.text(),
            'length_name': self.length_name.text()
        }

class CarouselTemplateDialog(QDialog):
    """Dialog for editing C++ carousel template"""
    template_changed = pyqtSignal(dict)

    def __init__(self, template_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Carousel Template Editor")
        self.template_data = template_data.copy()
        
        self.resize(500, 600)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)  # Increased spacing
        layout.setContentsMargins(8, 10, 8, 10)  # Increased margins
        
        # Set minimum size for better visibility
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

        # Help text
        help_text = QLabel(
            "Define the template for generating carousel C++ code.\n"
            "Available placeholders:\n"
            "- @{num_buffers}: Number of noise buffers\n"
            "- @{samples_per_buffer}: Samples per noise buffer\n"
            "- @{silence_samples}: Number of silence samples\n"
            "- @{generator_type}: Type of noise generator used\n"
            "- @{buffer_declarations}: Individual buffer declarations\n"
            "- @{buffer_list}: List of buffer pointers\n"
            "- @{buffer_array_name}: Name of buffer array\n"
            "- @{silence_buffer_name}: Name of silence buffer\n"
            "- @{silence_data}: Silence buffer data"
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Template text editor
        self.template_edit = QPlainTextEdit()
        self.template_edit.setPlainText(self.template_data['template_text'])
        layout.addWidget(QLabel("Template:"))
        layout.addWidget(self.template_edit)

        # Buffer naming options
        name_group = QGroupBox("Buffer Naming")
        name_layout = QFormLayout()

        self.buffer_format = QLineEdit(self.template_data['buffer_name_format'])
        name_layout.addRow("Buffer Name Format:", self.buffer_format)
        
        self.array_name = QLineEdit(self.template_data['buffer_array_name'])
        name_layout.addRow("Buffer Array Name:", self.array_name)
        
        self.silence_name = QLineEdit(self.template_data['silence_buffer_name'])
        name_layout.addRow("Silence Buffer Name:", self.silence_name)

        name_group.setLayout(name_layout)
        layout.addWidget(name_group)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        """Save template changes and emit signal"""
        template = {
            'template_text': self.template_edit.toPlainText(),
            'buffer_name_format': self.buffer_format.text(),
            'buffer_array_name': self.array_name.text(),
            'silence_buffer_name': self.silence_name.text()
        }
        self.template_changed.emit(template)
        super().accept()

def create_menu_bar(parent: QMainWindow) -> QMenuBar:
    menubar = QMenuBar()
    
    # File menu  
    file_menu = QMenu("&File", parent)
    file_menu.addAction("&Save Settings", parent.save_settings)
    file_menu.addAction("&Load Settings", parent.load_settings)
    file_menu.addSeparator()
    file_menu.addAction("Export &White Noise...", parent.export_white_noise)
    file_menu.addSeparator()
    file_menu.addAction("&Exit", parent.close)
    menubar.addMenu(file_menu)
    
    return menubar

class CarouselSettingsDialog(QDialog):
    """Dialog for configuring multi-noise carousel settings"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Carousel Settings")
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)
        layout.setSpacing(6)  # Increased spacing
        layout.setContentsMargins(8, 10, 8, 10)  # Increased margins
        layout.setHorizontalSpacing(8)  # Space between labels and fields
        layout.setVerticalSpacing(6)  # Space between rows
        
        # Number of samples
        self.num_samples = QSpinBox()
        self.num_samples.setRange(1, 100)
        self.num_samples.setValue(20)
        layout.addRow("Number of Samples:", self.num_samples)
        
        # Noise duration
        self.noise_duration = QDoubleSpinBox()
        self.noise_duration.setRange(1.0, 1000.0)
        self.noise_duration.setValue(10.0)
        self.noise_duration.setSuffix(" ms")
        layout.addRow("Noise Duration:", self.noise_duration)
        
        # Silence duration
        self.silence_duration = QDoubleSpinBox()
        self.silence_duration.setRange(0.0, 1000.0)
        self.silence_duration.setValue(190.0)
        self.silence_duration.setSuffix(" ms")
        layout.addRow("Silence Duration:", self.silence_duration)
        
        # Export options
        self.export_group = QGroupBox("Export Format")
        export_layout = QVBoxLayout()
        
        self.export_combined = QCheckBox("Combined Sequence")
        self.export_individual = QCheckBox("Individual Files")
        self.export_combined.setChecked(True)
        
        export_layout.addWidget(self.export_combined)
        export_layout.addWidget(self.export_individual)
        self.export_group.setLayout(export_layout)
        layout.addRow(self.export_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_settings(self) -> dict:
        return {
            'num_samples': self.num_samples.value(),
            'noise_duration_ms': self.noise_duration.value(),
            'silence_duration_ms': self.silence_duration.value(),
            'export_combined': self.export_combined.isChecked(),
            'export_individual': self.export_individual.isChecked()
        }

class PointEditDialog(QDialog):
    """Dialog for editing a single frequency/level point"""
    def __init__(self, parent=None, freq=1000.0, level=0.0, is_add=False):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle("Add Point" if is_add else "Edit Point")
        self.freq = freq
        self.level = level
        self.init_ui()
        
    def init_ui(self):
        layout = QFormLayout(self)
        
        # Frequency input
        self.freq_input = QDoubleSpinBox()
        self.freq_input.setRange(20.0, 20000.0)
        self.freq_input.setValue(self.freq)
        self.freq_input.setSuffix(" Hz")
        layout.addRow("Frequency:", self.freq_input)
        
        # Level input
        self.level_input = QDoubleSpinBox()
        self.level_input.setRange(-100.0, 100.0)
        self.level_input.setValue(self.level)
        self.level_input.setSuffix(" dB")
        layout.addRow("Level:", self.level_input)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
        
    def get_values(self) -> Tuple[float, float]:
        return (self.freq_input.value(), self.level_input.value())

class DeviceComboBox(QComboBox):
    """Custom QComboBox for audio device selection with automatic refresh"""
    
    device_list_updated = pyqtSignal()  # Signal emitted when device list is updated
    
    def __init__(self, input_devices: bool = False):
        super().__init__()
        self._is_input = input_devices
        self._current_devices = None
        self._refresh_thread = None
        self._lock = threading.Lock()
        self._popup_visible = False
        
        # Create timer for periodic refresh
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(1000)  # 1 second interval
        self._refresh_timer.timeout.connect(self._start_refresh)
        
        # Initial device list update
        self._force_update_devices()
        
        # Connect signal to handle device list updates
        self.device_list_updated.connect(self._apply_device_list_update)
        
    def showPopup(self):
        """Override showPopup to refresh device list before showing"""
        # Don't show popup if disabled
        if not self.isEnabled():
            return
            
        logger.debug("Dropdown about to show - starting refresh")
        self._popup_visible = True
        
        # Show current list immediately
        super().showPopup()
        
        # Start initial refresh
        self._start_refresh()
        
        # Start periodic refresh timer
        self._refresh_timer.start()
    
    def hidePopup(self):
        """Override hidePopup to track popup state"""
        self._popup_visible = False
        # Stop the refresh timer
        self._refresh_timer.stop()
        super().hidePopup()
    
    def _start_refresh(self):
        """Start a new refresh thread if one isn't already running"""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._refresh_thread = threading.Thread(target=self._background_refresh)
            self._refresh_thread.daemon = True
            self._refresh_thread.start()
    
    def _background_refresh(self):
        """Background thread to refresh device list"""
        try:
            with self._lock:
                # Force sounddevice to rescan hardware
                if self.isEnabled():  # Only do this when enabled (not playing)
                    sd._terminate()
                    sd._initialize()
                
                # Get current device list
                devices = sd.query_devices()
                device_list = []
                
                # Build list of available devices
                for i in range(len(devices)):
                    try:
                        device = sd.query_devices(i)
                        channels = device['max_input_channels'] if self._is_input else device['max_output_channels']
                        if channels > 0:
                            name = f"{device['name']} ({('In' if self._is_input else 'Out')}: {channels})"
                            device_list.append((name, i))
                    except Exception as e:
                        logger.error(f"Error querying device {i}: {e}")
                        continue
                
                # Only update if device list has changed
                if self._current_devices != device_list:
                    logger.debug("Device list changed, will update combo box")
                    self._current_devices = device_list
                    # Emit signal to update UI in main thread
                    self.device_list_updated.emit()
                    
        except Exception as e:
            logger.error(f"Error refreshing device list: {e}")
    
    def _apply_device_list_update(self):
        """Apply device list update in the main thread"""
        current_data = self.currentData()
        
        self.clear()
        self.addItem("No Audio Device", None)
        
        # Add devices
        for name, idx in self._current_devices:
            self.addItem(name, idx)
            
        # Restore previous selection if it still exists
        if current_data is not None:
            index = self.findData(current_data)
            if index >= 0:
                self.setCurrentIndex(index)
        
        # If popup is visible, hide and show it again to force size update
        if self._popup_visible:
            self.hidePopup()
            self.showPopup()
    
    def _force_update_devices(self):
        """Force a synchronous device list update while preserving selection"""
        with self._lock:
            current_data = self.currentData()
            
            try:
                # Force sounddevice to rescan hardware
                if self.isEnabled():  # Only do this when enabled (not playing)
                    sd._terminate()
                    sd._initialize()
                
                # Get current device list
                devices = sd.query_devices()
                device_list = []
                
                # Build list of available devices
                for i in range(len(devices)):
                    try:
                        device = sd.query_devices(i)
                        channels = device['max_input_channels'] if self._is_input else device['max_output_channels']
                        if channels > 0:
                            name = f"{device['name']} ({('In' if self._is_input else 'Out')}: {channels})"
                            device_list.append((name, i))
                    except Exception as e:
                        logger.error(f"Error querying device {i}: {e}")
                        continue
                
                # Update device list
                self._current_devices = device_list
                self.clear()
                self.addItem("No Audio Device", None)
                
                # Add devices
                for name, idx in device_list:
                    self.addItem(name, idx)
                    
                # Restore previous selection if it still exists
                if current_data is not None:
                    index = self.findData(current_data)
                    if index >= 0:
                        self.setCurrentIndex(index)
                        
            except Exception as e:
                logger.error(f"Error updating device list: {e}")
    
    def currentDeviceInfo(self):
        """Get current device info or None if no device selected"""
        device_idx = self.currentData()
        if device_idx is not None:
            try:
                return sd.query_devices(device_idx)
            except Exception as e:
                logger.error(f"Error getting device info: {e}")
        return None

    def get_device_info(self) -> Optional[dict]:
        """Get current device info as a serializable dict"""
        device_idx = self.currentData()
        if device_idx is not None:
            try:
                device = sd.query_devices(device_idx)
                return {
                    'name': device['name'],
                    'hostapi': device['hostapi'],
                    'max_input_channels': device['max_input_channels'],
                    'max_output_channels': device['max_output_channels']
                }
            except:
                return None
        return None

    def set_device_from_info(self, device_info: Optional[dict]) -> bool:
        """Try to find and set device matching the saved info"""
        if not device_info:
            self.setCurrentIndex(0)  # Set to "No Audio Device"
            return True

        # Force refresh device list
        self._force_update_devices()

        # Try to find matching device
        devices = sd.query_devices()
        for i in range(len(devices)):
            try:
                device = sd.query_devices(i)
                if (device['name'] == device_info['name'] and
                    device['hostapi'] == device_info['hostapi'] and
                    device['max_input_channels'] == device_info['max_input_channels'] and
                    device['max_output_channels'] == device_info['max_output_channels']):
                    
                    # Found matching device, check if it has required channels
                    channels = device['max_input_channels'] if self._is_input else device['max_output_channels']
                    if channels > 0:
                        index = self.findData(i)
                        if index >= 0:
                            self.setCurrentIndex(index)
                            return True
            except:
                continue

        # No matching device found
        self.setCurrentIndex(0)  # Set to "No Audio Device"
        return False
