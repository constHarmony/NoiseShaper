# ui_components.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QSlider, QMenuBar, QMenu, QStatusBar, QMainWindow,
    QMessageBox, QDialog, QFileDialog, QDialogButtonBox, QScrollArea, 
    QFrame, QSizePolicy, QLineEdit, QGridLayout, QListWidget, QListWidgetItem,
    QPlainTextEdit, QInputDialog  # Add QInputDialog here
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from typing import Dict, Any, List, Tuple, Optional  # Add these imports
import sounddevice as sd
from config import AudioConfig
from audio_sources import MonitoredInputSource
from PyQt6.QtGui import QDoubleValidator
import numpy as np
import os  # Add this import

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
        self.config = config
        self.setWindowTitle("Audio Buffer Settings")
        self.setModal(True)
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

        # Device selector
        self.device_combo = QComboBox()
        self.update_device_list()
        layout.addRow("Output Device:", self.device_combo)

        # Monitoring checkbox
        self.monitor_checkbox = QCheckBox("Enable Monitoring")
        self.monitor_checkbox.toggled.connect(self.on_monitor_toggled)
        layout.addRow(self.monitor_checkbox)

        # Volume slider with marks
        volume_layout = QHBoxLayout()
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(20)  # Default to 20%
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        
        volume_label = QLabel("Volume:")
        volume_value = QLabel("20%")  # Initial value
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
        return {
            'monitoring_enabled': self.monitor_checkbox.isChecked(),
            'monitoring_volume': self.volume_slider.value(),
            'output_device_index': self.device_combo.currentData()
        }

    def apply_settings(self, settings: Dict[str, Any]):
        if 'monitoring_enabled' in settings:
            self.monitor_checkbox.setChecked(settings['monitoring_enabled'])
        if 'monitoring_volume' in settings:
            self.volume_slider.setValue(settings['monitoring_volume'])
        if 'output_device_index' in settings:
            index = self.device_combo.findData(settings['output_device_index'])
            if (index >= 0):
                self.device_combo.setCurrentIndex(index)

class InputDevicePanel(QGroupBox):
    device_changed = pyqtSignal(int)

    def __init__(self, config: AudioConfig):
        super().__init__("Input Device")
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # Device selector
        self.device_combo = QComboBox()
        self.update_device_list()
        self.device_combo.currentIndexChanged.connect(self.on_device_changed)
        layout.addRow("Input Device:", self.device_combo)

        # Input channel selector
        self.channel_combo = QComboBox()
        self.channel_combo.setVisible(False)
        layout.addRow("Input Channel:", self.channel_combo)

    def update_device_list(self):
        update_device_list(self.device_combo, input_devices=True)

    def on_device_changed(self):
        device_idx = self.device_combo.currentData()
        if device_idx is not None:
            self.device_changed.emit(device_idx)
            self.config.device_input_index = device_idx
            
            # Update channel selector
            device_info = sd.query_devices(device_idx)
            self.channel_combo.clear()
            for i in range(device_info['max_input_channels']):
                self.channel_combo.addItem(f"Channel {i+1}", i)
            self.channel_combo.setVisible(device_info['max_input_channels'] > 1)

class SourcePanel(QGroupBox):
    source_changed = pyqtSignal()
    export_requested = pyqtSignal(dict)  # Add new signal

    def __init__(self, config: AudioConfig):
        super().__init__("Audio Source")
        self.config = config
        self.is_playing = False
        self.current_source = None
        self.export_settings = {}  # Initialize empty export settings
        
        # Create MonitoringPanel and InputDevicePanel
        self.monitoring_panel = MonitoringPanel(config)
        self.input_device_panel = InputDevicePanel(config)
        
        # Hide input device panel initially
        self.input_device_panel.hide()
        
        # Initialize cpp_template with default values
        self.cpp_template = {
            'template_text': (
                "// Auto-generated audio data header\n"
                "#pragma once\n\n"
                "#define {length_name} {length}  // Array length\n\n"
                "// Audio samples normalized to int16 (-32768 to 32767)\n"
                "const int16_t {var_name}[{length_name}] = {{\n"
                "{array_data}\n"
                "}};\n"
            ),
            'var_name': 'audioData',
            'length_name': 'AUDIO_LENGTH'
        }
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Mode selector and export group
        mode_group = QHBoxLayout()
        
        # Source type selector
        self.source_type = QComboBox()
        self.source_type.addItems(["White Noise", "Spectral Synthesis", "Test Mode"])
        mode_group.addWidget(QLabel("Mode:"))
        mode_group.addWidget(self.source_type)
        
        # Export button - only for noise modes
        self.export_button = QPushButton("Export...")
        self.export_button.clicked.connect(self.export_noise)
        mode_group.addWidget(self.export_button)
        
        layout.addLayout(mode_group)

        # Add RNG type selector between mode group and amplitude control
        rng_layout = QHBoxLayout()
        rng_layout.addWidget(QLabel("RNG Type:"))
        self.rng_type = QComboBox()
        self.rng_type.addItems(["Standard Normal", "Uniform"])
        self.rng_type.currentTextChanged.connect(self.on_rng_type_changed)
        rng_layout.addWidget(self.rng_type)
        layout.addLayout(rng_layout)

        # Add amplitude control between mode group and monitoring panel
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Amplitude:"))
        self.amplitude_control = QDoubleSpinBox()
        self.amplitude_control.setRange(0.0, 1.0)
        self.amplitude_control.setValue(0.5)  # Default to 0.5 like DIY examples
        self.amplitude_control.setSingleStep(0.1)
        self.amplitude_control.valueChanged.connect(self.on_amplitude_changed)
        amp_layout.addWidget(self.amplitude_control)
        layout.addLayout(amp_layout)

        # Add input device panel (for test mode)
        layout.addWidget(self.input_device_panel)
        
        # Add monitoring panel
        layout.addWidget(self.monitoring_panel)

        # Play/Stop button
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        layout.addWidget(self.play_button)

        # Connect signals
        self.source_type.currentTextChanged.connect(self.on_source_type_changed)
        self.monitoring_panel.monitoring_changed.connect(self.on_monitoring_changed)
        self.monitoring_panel.volume_changed.connect(self.on_volume_changed)
        self.input_device_panel.device_changed.connect(lambda x: self.update_controls())
        
        # Initial control state
        self.update_controls()

    def export_noise(self):
        """Trigger appropriate export dialog based on source type"""
        try:
            source_type = self.get_source_type()
            dialog = ExportDialog(self, mode=source_type)
            
            # Apply stored settings and current mode's amplitude
            if self.export_settings:
                dialog.apply_saved_settings(self.export_settings)
            else:
                # Initialize with current amplitude
                current_amp = (self.config.amp_whitenoise 
                             if source_type == "White Noise" 
                             else self.config.amp_spectral)
                dialog.amplitude.setValue(current_amp)
            
            # Connect amplitude control to live amplitude
            dialog.amplitude.valueChanged.connect(self.amplitude_control.setValue)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                settings = dialog.get_settings()
                # Store settings for reuse
                self.export_settings = settings
                self.export_requested.emit(settings)
                
        except Exception as e:
            print(f"Export dialog error: {e}")

    def on_source_type_changed(self, new_type: str):
        """Handle source type changes with mode-specific amplitude"""
        self.export_button.setVisible(new_type != "Test Mode")
        
        # Update amplitude based on mode
        if new_type == "White Noise":
            self.amplitude_control.setValue(self.config.amp_whitenoise)
        elif new_type == "Spectral Synthesis":
            self.amplitude_control.setValue(self.config.amp_spectral)
        
        # Stop playback if running
        if self.is_playing:
            self.toggle_playback()
            
        # Update control states
        self.update_controls()

    def update_controls(self):
        """Update control states based on current settings"""
        source_type = self.source_type.currentText()
        is_test_mode = source_type == "Test Mode"
        
        # Show/hide panels based on mode
        self.input_device_panel.setVisible(is_test_mode)
        self.export_button.setVisible(not is_test_mode)
        
        # Update play button state
        if (is_test_mode):
            # Enable play button only if an input device is selected
            device_idx = self.input_device_panel.device_combo.currentData()
            self.play_button.setEnabled(device_idx is not None)
        else:
            # For noise modes, always enable play button
            self.play_button.setEnabled(True)

    def get_current_settings(self) -> Dict[str, Any]:
        """Returns current source settings with both amplitudes"""
        settings = {
            'source_type': self.source_type.currentText(),
            'monitoring': self.monitoring_panel.get_current_settings(),
            'input_device': self.input_device_panel.device_combo.currentData(),
            'amp_whitenoise': self.config.amp_whitenoise,
            'amp_spectral': self.config.amp_spectral,
            'rng_type': self.rng_type.currentText().lower().replace(' ', '_'),
            'cpp_template': self.cpp_template  # Add template settings
        }
        return settings

    def apply_settings(self, settings: Dict[str, Any]):
        """Applies loaded settings with both amplitudes"""
        if 'source_type' in settings:
            index = self.source_type.findText(settings['source_type'])
            if index >= 0:
                self.source_type.setCurrentIndex(index)
        
        if 'monitoring' in settings:
            self.monitoring_panel.apply_settings(settings['monitoring'])
            
        # Update control states
        self.update_controls()
        
        # Store both amplitudes
        if 'amp_whitenoise' in settings:
            self.config.amp_whitenoise = settings['amp_whitenoise']
        if 'amp_spectral' in settings:
            self.config.amp_spectral = settings['amp_spectral']
            
        # Set the right amplitude for current mode
        source_type = self.source_type.currentText()
        if source_type == "White Noise":
            self.amplitude_control.setValue(self.config.amp_whitenoise)
        elif source_type == "Spectral Synthesis":
            self.amplitude_control.setValue(self.config.amp_spectral)
            
        if 'rng_type' in settings:
            index = self.rng_type.findText(settings['rng_type'].replace('_', ' ').title())
            if index >= 0:
                self.rng_type.setCurrentIndex(index)
                
        # Apply export settings if present
        if 'export' in settings:
            self.export_settings = settings['export'].copy()
            
        # Apply cpp template settings if present
        if 'cpp_template' in settings:
            self.cpp_template = settings['cpp_template']

    def on_device_changed(self):
        """Handle device selection changes"""
        device_idx = self.input_device_panel.device_combo.currentData()
        if device_idx is not None:
            device_info = sd.query_devices(device_idx)
            
            # Update input channel selector
            if self.source_type.currentText() == "Microphone/Line In":
                self.input_device_panel.channel_combo.clear()
                for i in range(device_info['max_input_channels']):
                    self.input_device_panel.channel_combo.addItem(f"Channel {i+1}", i)
                self.input_device_panel.channel_combo.setVisible(device_info['max_input_channels'] > 1)

            # Update config and current source if running
            self.config.device_output_index = device_idx
            self.config.output_device_enabled = True
            if self.current_source and hasattr(self.current_source, 'update_output_device'):
                self.current_source.update_output_device()
        
        # Update control states
        self.update_controls()

    def toggle_playback(self):
        try:
            if self.is_playing:
                # First set button state
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
                
            else:
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
                self.config.device_input_index = input_device_idx
                self.config.device_output_index = output_device_idx
                self.config.input_device_enabled = (input_device_idx is not None)
                self.config.output_device_enabled = (output_device_idx is not None)
                self.config.monitoring_enabled = self.monitoring_panel.monitor_checkbox.isChecked()
                self.config.monitoring_volume = self.monitoring_panel.volume_slider.value() / 100.0
                
                # Update state before emitting signal
                self.play_button.setText("Stop")
                self.play_button.setChecked(True)
                self.is_playing = True
                
                # Finally emit signal to start processing
                self.source_changed.emit()

        except Exception as e:
            print(f"Playback toggle error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Playback error: {str(e)}")
            self.is_playing = False
            self.play_button.setText("Play")
            self.play_button.setChecked(False)
            if self.current_source:
                try:
                    self.current_source.close()
                finally:
                    self.current_source = None

    def handle_source_reference(self, source):
        """Store reference to current source"""
        self.current_source = source

    def on_volume_changed(self, value):
        """Handle volume slider changes"""
        self.config.monitoring_volume = value / 100.0

    def on_monitoring_changed(self, enabled: bool):
        """Handle monitoring toggle"""
        try:
            # Update config
            self.config.monitoring_enabled = enabled
            self.config.output_device_enabled = True  # Enable output device
            
            # Get the current output device
            device_idx = self.monitoring_panel.device_combo.currentData()
            self.config.device_output_index = device_idx
            
            # If we have an active source with monitoring capability, update it
            if self.current_source:
                if hasattr(self.current_source, 'update_output_device'):
                    self.current_source.update_output_device()
                if hasattr(self.current_source, 'update_monitoring'):
                    self.current_source.update_monitoring()
                
        except Exception as e:
            print(f"Monitoring toggle error: {e}")
            # Revert checkbox state if there was an error
            self.monitoring_panel.monitor_checkbox.blockSignals(True)
            self.monitoring_panel.monitor_checkbox.setChecked(not enabled)
            self.monitoring_panel.monitor_checkbox.blockSignals(False)
            raise

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
        self.init_ui()

    def init_ui(self):
        layout = QFormLayout(self)

        # Analysis Settings
        analysis_group = QGroupBox("FFT Analysis")
        analysis_layout = QFormLayout(analysis_group)
        
        # FFT Size
        self.fft_size = QComboBox()
        self.fft_size.addItems(['128', '256', '512', '1024', '2048', '4096', '8192'])  # Added smaller sizes
        self.fft_size.setCurrentText(str(self.config.fft_size))
        self.fft_size.currentTextChanged.connect(self.on_settings_changed)
        analysis_layout.addRow("FFT Size:", self.fft_size)

        # Window Type
        self.window_type = QComboBox()
        self.window_type.addItems(['hanning', 'hamming', 'blackman', 'flattop', 'rectangular'])
        self.window_type.setCurrentText(self.config.window_type)
        self.window_type.currentTextChanged.connect(self.on_window_changed)
        analysis_layout.addRow("Window:", self.window_type)

        # Scale Type
        self.scale_type = QComboBox()
        self.scale_type.addItems(['Linear', 'Logarithmic'])
        self.scale_type.currentTextChanged.connect(self.on_scale_changed)
        analysis_layout.addRow("Scale:", self.scale_type)

        # Averaging
        self.averaging = QSpinBox()
        self.averaging.setRange(1, 16)
        self.averaging.setValue(self.config.averaging_count)
        self.averaging.valueChanged.connect(self.on_settings_changed)
        analysis_layout.addRow("Averaging:", self.averaging)
        
        layout.addRow(analysis_group)
        
    def on_scale_changed(self, new_scale: str):
        """Explicitly handle scale changes"""
        try:
            # Update config
            self.config.scale_type = new_scale.lower()
            # Emit signal for parent to handle
            self.settings_changed.emit()
        except Exception as e:
            print(f"Scale change error: {e}")

    def on_window_changed(self, window_type: str):
        """Explicitly handle window type changes"""
        try:
            # Update config
            self.config.window_type = window_type
            # Emit signal for parent to handle
            self.settings_changed.emit()
        except Exception as e:
            print(f"Window change error: {e}")

    def on_settings_changed(self):
        """Signal that settings have changed"""
        self.settings_changed.emit()
        # Notify parent of changes
        if hasattr(self.parent(), 'mark_unsaved_changes'):
            self.parent().mark_unsaved_changes()

    def get_current_settings(self) -> Dict[str, Any]:
        settings = {
            'fft_size': int(self.fft_size.currentText()),
            'window_type': self.window_type.currentText(),
            'scale_type': self.scale_type.currentText().lower(),
            'averaging_count': self.averaging.value()
        }
        return settings

    def apply_settings(self, settings: Dict[str, Any]):
        if 'fft_size' in settings:
            self.fft_size.setCurrentText(str(settings['fft_size']))
        if 'window_type' in settings:
            index = self.window_type.findText(settings['window_type'])
            if index >= 0:
                self.window_type.setCurrentIndex(index)
        if 'scale_type' in settings:
            index = self.scale_type.findText(settings['scale_type'].title())
            if index >= 0:
                self.scale_type.setCurrentIndex(index)
        if 'averaging_count' in settings:
            self.averaging.setValue(settings['averaging_count'])

class FilterParamDialog(QDialog):
    def __init__(self, filter_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Filter Parameters")
        self.setModal(True)
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
            self.param_widgets['amplitude'] = ParameterControl(0.0, 1.0, self.params.get('amplitude', 1.0), 2, "", 0.1)
            params_layout.addRow("Cutoff:", self.param_widgets['cutoff'])
            params_layout.addRow("Order:", self.param_widgets['order'])
            params_layout.addRow("Amplitude:", self.param_widgets['amplitude'])

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
            self.param_widgets['amplitude'] = ParameterControl(0.0, 1.0, self.params.get('amplitude', 1.0), 2, "", 0.1)
            params_layout.addRow("Low Cut:", self.param_widgets['lowcut'])
            params_layout.addRow("High Cut:", self.param_widgets['highcut'])
            params_layout.addRow("Order:", self.param_widgets['order'])
            params_layout.addRow("Amplitude:", self.param_widgets['amplitude'])

        elif self.filter_type == 'notch':
            self.param_widgets['freq'] = ParameterControl(20.0, 20000.0, self.params.get('freq', 1000.0), 0, " Hz")
            self.param_widgets['q'] = ParameterControl(0.1, 100.0, self.params.get('q', 30.0), 1)
            self.param_widgets['amplitude'] = ParameterControl(0.0, 1.0, self.params.get('amplitude', 1.0), 2, "", 0.1)
            params_layout.addRow("Frequency:", self.param_widgets['freq'])
            params_layout.addRow("Q Factor:", self.param_widgets['q'])
            params_layout.addRow("Amplitude:", self.param_widgets['amplitude'])

        elif self.filter_type in ['gaussian', 'parabolic']:
            self.param_widgets['center_freq'] = ParameterControl(20.0, 20000.0, self.params.get('center_freq', 1000.0), 0, " Hz")
            self.param_widgets['width'] = ParameterControl(1.0, 5000.0, self.params.get('width', 100.0), 0, " Hz")
            self.param_widgets['amplitude'] = ParameterControl(0.0, 1.0, self.params.get('amplitude', 0.5), 2, "", 0.1)
            self.param_widgets['skew'] = ParameterControl(-5.0, 5.0, self.params.get('skew', 0.0), 2)
            if self.filter_type == 'gaussian':
                self.param_widgets['kurtosis'] = ParameterControl(0.2, 5.0, self.params.get('kurtosis', 1.0), 2)
            else:  # parabolic
                self.param_widgets['flatness'] = ParameterControl(0.2, 5.0, self.params.get('flatness', 1.0), 2)
            
            params_layout.addRow("Center:", self.param_widgets['center_freq'])
            params_layout.addRow("Width:", self.param_widgets['width'])
            params_layout.addRow("Amplitude:", self.param_widgets['amplitude'])
            params_layout.addRow("Skew:", self.param_widgets['skew'])
            if self.filter_type == 'gaussian':
                params_layout.addRow("Kurtosis:", self.param_widgets['kurtosis'])
            else:  # parabolic
                params_layout.addRow("Flatness:", self.param_widgets['flatness'])

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

    def __init__(self, config: AudioConfig, processor=None):  # Add processor parameter
        super().__init__("Filters")
        self.config = config
        self.processor = processor  # Store processor reference
        self.filters = []
        self.init_ui()

    def init_ui(self):
        # Use QVBoxLayout for the main panel
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # Replace button layout with combo box and single add button
        add_layout = QHBoxLayout()
        
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Lowpass", "Highpass", "Bandpass", "Notch", "Gaussian", "Parabolic"])  # Added Parabolic
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
        self.filter_layout.setSpacing(4)  # Reduce space between filters
        self.filter_layout.setContentsMargins(4, 4, 4, 4)  # Reduce margins
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
            'parabolic': {'type': 'parabolic', 'center_freq': 1000, 'width': 100, 'amplitude': 1.0}
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
        # Clear existing filters
        while self.filters:
            self.remove_filter(0)
            
        # Add filters from settings
        for filter_params in settings.get('filters', []):
            filter_type = filter_params['type']
            widget = FilterWidget(filter_type, filter_params)
            widget.parameterChanged.connect(
                lambda p, idx=len(self.filters): self.filter_updated.emit(idx, p))
            widget.removeRequested.connect(
                lambda idx=len(self.filters): self.remove_filter(idx))
            self.filter_layout.insertWidget(len(self.filters), widget)
            self.filters.append(widget)

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

class SpectralComponentsPanel(QGroupBox):  # Renamed from ParabolaPanel
    parabola_updated = pyqtSignal(int, dict)
    parabola_removed = pyqtSignal(int)
    parabola_added = pyqtSignal(dict)

    def __init__(self, processor=None):  # Add processor parameter
        super().__init__("Spectral Components")
        self.processor = processor  # Store processor reference
        self.parabolas = []
        self.init_ui()

    def init_ui(self):
        # Use QVBoxLayout for the main panel
        main_layout = QVBoxLayout(self)
        self.setLayout(main_layout)

        # Add button
        add_btn = QPushButton("Add Component")
        main_layout.addWidget(add_btn)
        add_btn.clicked.connect(self.add_parabola)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Create container widget for components
        self.parabola_container = QWidget()
        self.parabola_layout = QVBoxLayout(self.parabola_container)
        self.parabola_layout.setSpacing(4)
        self.parabola_layout.setContentsMargins(4, 4, 4, 4)
        self.parabola_layout.addStretch()
        
        # Add container to scroll area
        scroll.setWidget(self.parabola_container)
        
        # Add scroll area to main layout, with stretch
        main_layout.addWidget(scroll, stretch=1)

    def add_parabola(self):
        params = {
            'id': len(self.parabolas) + 1,
            'center_freq': 1000.0,
            'width': 100.0,
            'amplitude': 0.5,
            'slope': 1.0,
            'phase': 0.0
        }
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
            before="""
// Auto-generated audio data header
#pragma once

#define {length_name} {length}  // Array length

// Audio samples normalized to int16 (-32768 to 32767)
extern const int16_t {var_name}[{length_name}];
""",
            after="",
            var_name="audioData",
            length_name="AUDIO_LENGTH"
        )

        # Source file template with Arduino example (.cpp)
        source_template = cls(
            name="Arduino Source",
            before="""
// Auto-generated audio source file
#include "{filename}.h"

/* Example usage:
class AudioPlayer {
    int currentIndex = 0;
public:
    int16_t getNextSample() {
        if (currentIndex >= {length_name}) currentIndex = 0;
        return {var_name}[currentIndex++];
    }
};
*/

// Audio samples array
const int16_t {var_name}[{length_name}] = {""",
            after="""};
""",
            var_name="audioData",
            length_name="AUDIO_LENGTH"
        )

        return [header_template, source_template]

class ExportDialog(QDialog):
    def __init__(self, parent=None, mode="White Noise"):  # Add mode parameter
        super().__init__(parent)
        self.setWindowTitle(f"Export {mode}")  # Update title with mode
        self.setModal(True)
        self.folder_path = None
        # Initialize cpp_template with default values
        self.cpp_template = {
            'template_text': (
                "// Auto-generated audio data header\n"
                "#pragma once\n\n"
                "#define {length_name} {length}  // Array length\n\n"
                "// Audio samples normalized to int16 (-32768 to 32767)\n"
                "const int16_t {var_name}[{length_name}] = {{\n"
                "{array_data}\n"
                "}};\n"
            ),
            'var_name': 'audioData',
            'length_name': 'AUDIO_LENGTH'
        }
        self.mode = mode  # Store mode
        
        # Get saved settings from parent if available
        if hasattr(parent, 'export_settings'):
            self.saved_settings = parent.export_settings
        else:
            self.saved_settings = {}
            
        # Get appropriate amplitude based on mode
        parent_amplitude = 0.5  # Default fallback
        if hasattr(parent, 'amplitude_control'):
            if mode == "White Noise":
                parent_amplitude = parent.config.amp_whitenoise
            elif mode == "Spectral Synthesis":
                parent_amplitude = parent.config.amp_spectral
            
        # Initialize export settings dict if not already populated
        if not self.saved_settings:
            self.saved_settings = {
                'amplitude': parent_amplitude,
                'attenuation': 0,
                'enable_attenuation': False
            }

        # Initialize UI elements
        self.init_ui()

        # Initialize cpp_template from parent if available
        if hasattr(parent, 'cpp_template'):
            self.cpp_template = parent.cpp_template.copy()
        else:
            self.cpp_template = {
                'template_text': "// Default template\n",
                'var_name': 'audioData',
                'length_name': 'AUDIO_LENGTH'
            }

    def apply_saved_settings(self, settings):
        """Apply previously saved settings to dialog controls"""
        try:
            if not settings:
                return
                
            # Duration
            if 'duration' in settings:
                self.duration.setValue(settings['duration'])
                
            # Amplitude/Attenuation - handle zero amplitude case
            if 'amplitude' in settings:
                amplitude = settings['amplitude']
                if amplitude > 0:  # Only calculate attenuation for non-zero amplitude
                    attn_db = -20 * np.log10(amplitude)
                    self.enable_attenuation.setChecked(attn_db > 0)
                    if attn_db > 0:
                        self.attenuation.setValue(int(min(attn_db, 96)))  # Clamp to max 96 dB
                else:
                    # For zero amplitude, set max attenuation
                    self.enable_attenuation.setChecked(True)
                    self.attenuation.setValue(96)  # Set to maximum allowed attenuation
                    
            # Fade settings
            if 'fade_in_duration' in settings:
                self.fade_in.setValue(settings['fade_in_duration'] * 1000.0)  # Convert to ms
            if 'fade_out_duration' in settings:
                self.fade_out.setValue(settings['fade_out_duration'] * 1000.0)  # Convert to ms
            if 'fade_in_power' in settings:
                self.fade_in_power.setValue(settings['fade_in_power'])
            if 'fade_out_power' in settings:
                self.fade_out_power.setValue(settings['fade_out_power'])
            if 'enable_fade' in settings:
                self.enable_fade_in.setChecked(settings['enable_fade'])
                self.enable_fade_out.setChecked(settings['enable_fade'])
                
            # Normalization
            if 'enable_normalization' in settings:
                self.enable_normalization.setChecked(settings['enable_normalization'])
            if 'normalize_value' in settings:
                self.normalize_value.setValue(settings['normalize_value'])
                
            # File settings
            if 'folder_path' in settings:
                self.folder_path = settings['folder_path']
                self.folder_path_label.setText(settings['folder_path'])
            if 'wav_filename' in settings:
                self.wav_filename.setText(settings['wav_filename'])
            if 'cpp_filename' in settings:
                self.cpp_filename.setText(settings['cpp_filename'])
            
            # Export options
            if 'export_wav' in settings:
                self.export_wav.setChecked(settings['export_wav'])
            if 'export_cpp' in settings:
                self.export_cpp.setChecked(settings['export_cpp'])
                
            # Seed settings
            if 'use_random_seed' in settings:
                self.use_random_seed.setChecked(settings['use_random_seed'])
            if 'seed' in settings and settings['seed'] is not None:
                self.seed_input.setValue(settings['seed'])
                
        except Exception as e:
            print(f"Error applying saved settings: {e}")

    def init_ui(self):
        layout = QFormLayout(self)

        # Export options group
        options_group = QGroupBox(f"{self.mode} Export Options")  # Update group title
        options_layout = QFormLayout(options_group)

        # Duration control
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.1, 3600.0)
        self.duration.setValue(10.0)
        self.duration.setSuffix(" milliseconds")
        options_layout.addRow("Duration:", self.duration)

        # Add base amplitude control - Updated to load from parent
        self.amplitude = QDoubleSpinBox()
        self.amplitude.setRange(0.0, 1.0)
        self.amplitude.setValue(self.parent().amplitude_control.value() if hasattr(self.parent(), 'amplitude_control') else 0.5)  # Get from parent
        self.amplitude.setSingleStep(0.1)
        self.amplitude.setDecimals(2)
        self.amplitude.setSuffix("x")
        options_layout.addRow("Base Amplitude:", self.amplitude)

        # Add seed control group after duration
        seed_group = QGroupBox("Random Seed")
        seed_layout = QHBoxLayout(seed_group)
        
        self.use_random_seed = QCheckBox("Random Seed")
        self.use_random_seed.setChecked(True)
        self.use_random_seed.toggled.connect(self.toggle_seed_input)
        
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 999999999)
        self.seed_input.setValue(12345)  # Default seed
        self.seed_input.setEnabled(False)
        
        # Regenerate button to get a new random seed
        self.regen_seed = QPushButton("New Seed")
        self.regen_seed.clicked.connect(self.generate_random_seed)
        self.regen_seed.setEnabled(False)
        
        seed_layout.addWidget(self.use_random_seed)
        seed_layout.addWidget(self.seed_input)
        seed_layout.addWidget(self.regen_seed)
        options_layout.addRow(seed_group)

        # Fade controls group moved after seed
        fade_group = QGroupBox("Fade Settings")
        fade_layout = QHBoxLayout(fade_group)
        
        # Fade in controls (left side)
        fade_in_group = QGroupBox("Fade In")
        fade_in_layout = QFormLayout()
        
        self.enable_fade_in = QCheckBox("Enable")
        self.enable_fade_in.setChecked(True)
        fade_in_layout.addRow(self.enable_fade_in)
        
        self.fade_in = QDoubleSpinBox()
        self.fade_in.setRange(0.1, 1000.0)
        self.fade_in.setValue(1.0)
        self.fade_in.setSuffix(" ms")
        self.fade_in.setEnabled(True)
        fade_in_layout.addRow("Duration:", self.fade_in)
        
        self.fade_in_power = QDoubleSpinBox()
        self.fade_in_power.setRange(0.1, 5.0)  # Increased max value
        self.fade_in_power.setValue(2.0)
        self.fade_in_power.setSingleStep(0.1)
        self.fade_in_power.setEnabled(True)
        fade_in_layout.addRow("Power:", self.fade_in_power)
        
        fade_in_group.setLayout(fade_in_layout)
        fade_layout.addWidget(fade_in_group)
        
        # Fade out controls (right side)
        fade_out_group = QGroupBox("Fade Out")
        fade_out_layout = QFormLayout()
        
        self.enable_fade_out = QCheckBox("Enable")
        self.enable_fade_out.setChecked(True)
        fade_out_layout.addRow(self.enable_fade_out)
        
        self.fade_out = QDoubleSpinBox()
        self.fade_out.setRange(0.1, 1000.0)
        self.fade_out.setValue(1.0)
        self.fade_out.setSuffix(" ms")
        self.fade_out.setEnabled(True)
        fade_out_layout.addRow("Duration:", self.fade_out)
        
        self.fade_out_power = QDoubleSpinBox()
        self.fade_out_power.setRange(0.1, 5.0)  # Increased max value
        self.fade_out_power.setValue(2.0)
        self.fade_out_power.setSingleStep(0.1)
        self.fade_out_power.setEnabled(True)
        fade_out_layout.addRow("Power:", self.fade_out_power)
        
        fade_out_group.setLayout(fade_out_layout)
        fade_layout.addWidget(fade_out_group)
        
        # Connect enable checkboxes
        self.enable_fade_in.toggled.connect(lambda e: self._update_fade_controls('in', e))
        self.enable_fade_out.toggled.connect(lambda e: self._update_fade_controls('out', e))
        
        options_layout.addRow(fade_group)

        # Normalization and attenuation
        norm_layout = QHBoxLayout()
        
        # Normalization checkbox and value
        self.enable_normalization = QCheckBox("Enable Normalization")
        self.enable_normalization.setChecked(True)
        norm_layout.addWidget(self.enable_normalization)
        
        self.normalize_value = QDoubleSpinBox()
        self.normalize_value.setRange(0.0, 2.0)
        self.normalize_value.setValue(1.0)
        self.normalize_value.setSingleStep(0.1)
        self.normalize_value.setDecimals(2)
        self.normalize_value.setEnabled(True)
        norm_layout.addWidget(self.normalize_value)
        
        options_layout.addRow(norm_layout)

        attn_layout = QHBoxLayout()
        self.enable_attenuation = QCheckBox("Additional Attenuation")
        self.enable_attenuation.setChecked(False)
        attn_layout.addWidget(self.enable_attenuation)
        
        self.attenuation = QSpinBox()
        self.attenuation.setRange(0, 96)
        self.attenuation.setValue(12)
        self.attenuation.setSuffix(" dB")
        self.attenuation.setEnabled(False)
        self.enable_attenuation.toggled.connect(self.attenuation.setEnabled)
        attn_layout.addWidget(self.attenuation)
        options_layout.addRow(attn_layout)
        
        layout.addRow(options_group)

        # File settings group
        file_group = QGroupBox("Output Files")
        file_layout = QFormLayout(file_group)

        # Folder path with browse button
        folder_layout = QHBoxLayout()
        self.folder_path_label = QLabel("None")
        folder_layout.addWidget(self.folder_path_label, stretch=1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_folder)
        folder_layout.addWidget(browse_btn)
        file_layout.addRow("Output Folder:", folder_layout)

        # WAV file settings
        wav_layout = QHBoxLayout()
        self.wav_filename = QLineEdit("noise.wav")
        wav_layout.addWidget(self.wav_filename, stretch=1)
        self.export_wav = QCheckBox("Export WAV")
        self.export_wav.setChecked(True)
        self.export_wav.toggled.connect(self.validate_export_options)
        wav_layout.addWidget(self.export_wav)
        file_layout.addRow("WAV Filename:", wav_layout)

        # CPP file settings
        cpp_layout = QHBoxLayout()
        self.cpp_filename = QLineEdit("noise.h")  # Changed default to .h
        cpp_layout.addWidget(self.cpp_filename, stretch=1)
        
        self.export_cpp = QCheckBox("Export C++/H")  # Updated label
        self.export_cpp.setChecked(True)
        self.export_cpp.toggled.connect(self.validate_export_options)
        
        template_btn = QPushButton("Template...")
        template_btn.clicked.connect(self.edit_template)
        
        cpp_layout.addWidget(self.cpp_filename)
        cpp_layout.addWidget(template_btn)
        cpp_layout.addWidget(self.export_cpp)
        file_layout.addRow("C++ Filename:", cpp_layout)

        layout.addRow(file_group)

        # Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addRow(self.button_box)

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
        # Ensure proper file extensions before getting settings
        if self.export_wav.isChecked() and not self.wav_filename.text().lower().endswith('.wav'):
            self.wav_filename.setText(self.wav_filename.text() + '.wav')

        if self.export_cpp.isChecked() and not self.cpp_filename.text().lower().endswith(('.cpp', '.h')):
            self.cpp_filename.setText(self.cpp_filename.text() + '.h')

        # Convert attenuation in dB to amplitude multiplier only if enabled
        base_amplitude = self.amplitude.value()  # Get base amplitude
        try:
            if self.enable_attenuation.isChecked():
                attn_db = self.attenuation.value()
                final_amplitude = base_amplitude * (10 ** (-attn_db / 20))
                # Ensure we don't get values too close to zero
                final_amplitude = max(final_amplitude, 1e-10)
            else:
                final_amplitude = base_amplitude
        except Exception as e:
            print(f"Error calculating amplitude: {e}")
            final_amplitude = 1e-10  # Provide safe fallback value

        # Use cpp_filename for both header and source if cpp export is enabled
        cpp_name = self.cpp_filename.text()
        base_name = os.path.splitext(cpp_name)[0]  # Remove extension
        
        settings = {
            'duration': self.duration.value(),
            'sample_rate': 44100,
            'base_amplitude': base_amplitude,  # Add base amplitude to settings
            'amplitude': final_amplitude,  # Final amplitude after attenuation
            'export_wav': self.export_wav.isChecked(),
            'export_cpp': self.export_cpp.isChecked(),
            'enable_fade': self.enable_fade_in.isChecked() or self.enable_fade_out.isChecked(),
            'fade_in_duration': self.fade_in.value() / 1000.0 if self.enable_fade_in.isChecked() else 0,
            'fade_out_duration': self.fade_out.value() / 1000.0 if self.enable_fade_out.isChecked() else 0,
            'fade_in_power': self.fade_in_power.value(),
            'fade_out_power': self.fade_out_power.value(),
            'enable_normalization': self.enable_normalization.isChecked(),
            'normalize_value': self.normalize_value.value(),
            'folder_path': self.folder_path,
            'wav_filename': self.wav_filename.text(),
            'header_filename': base_name + '.h',  # Force .h extension for header
            'source_filename': base_name + '.cpp', # Force .cpp extension for source
            'cpp_template': self.cpp_template,
            'use_random_seed': self.use_random_seed.isChecked(),
            'seed': None if self.use_random_seed.isChecked() else self.seed_input.value()
        }
        return settings

    def edit_template(self):
        """Open template editor dialog"""
        dialog = CppTemplateDialog(self.cpp_template, self)
        # Connect template change signal
        dialog.template_changed.connect(self.on_template_changed)
        dialog.exec()

    def on_template_changed(self, template: dict):
        """Handle template changes"""
        self.cpp_template = template
        # Also update parent's template
        if hasattr(self.parent(), 'cpp_template'):
            self.parent().cpp_template = self.cpp_template.copy()
            # Mark changes if needed
            if hasattr(self.parent(), 'mark_unsaved_changes'):
                self.parent().mark_unsaved_changes()

# In other panel classes (AnalyzerPanel, SourcePanel, FilterPanel)
# Add change notification to parameter changes:

def on_parameter_changed(self):
    """Call when any parameter changes"""
    if hasattr(self.parent(), 'mark_unsaved_changes'):
        self.parent().mark_unsaved_changes()

class HearingTestDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extended Audiometry Data")
        self.setModal(True)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Data entry grid
        grid = QGridLayout()
        self.data_entries = {}
        
        # Extended audiogram frequencies
        frequencies = [20, 31.5, 63, 125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10000, 12500, 16000]
        
        # Headers
        grid.addWidget(QLabel("Frequency (Hz)"), 0, 0)
        grid.addWidget(QLabel("Threshold (dB)"), 0, 1)
        
        # Create entry rows
        for i, freq in enumerate(frequencies):
            freq_label = QLabel(f"{freq}")
            threshold = QSpinBox()
            threshold.setRange(-60, 120)  # Extended range with more negative values
            threshold.setValue(-10)  # Default to slightly better than average
            threshold.setSuffix(" dB")
            
            grid.addWidget(freq_label, i+1, 0)
            grid.addWidget(threshold, i+1, 1)
            
            self.data_entries[freq] = threshold

        # Add scroll area for many frequencies
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container.setLayout(grid)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> dict:
        return {freq: spin.value() for freq, spin in self.data_entries.items()}

    def set_data(self, data: dict):
        for freq, value in data.items():
            if freq in self.data_entries:
                self.data_entries[freq].setValue(value)

class OverlayTemplate:
    def __init__(self, name: str, color: str, points: List[Tuple[float, float]]):
        self.name = name
        self.color = color
        self.points = points
        self.enabled = True
        self.offset = 0  # Change to int since we're using QSpinBox

class PointEditDialog(QDialog):
    """Dialog for editing individual points"""
    def __init__(self, freq: float = None, level: float = None, is_add: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Point" if is_add else "Edit Point")
        self.setModal(True)
        
        # Use last values if no values provided
        if freq is None:
            freq = PointEditDialog.last_freq if hasattr(PointEditDialog, 'last_freq') else 1000
        if level is None:
            level = PointEditDialog.last_level if hasattr(PointEditDialog, 'last_level') else 0
            
        self.init_ui(freq, level)

    def init_ui(self, freq: float, level: float):
        layout = QFormLayout(self)
        
        # Frequency input
        self.freq_edit = QDoubleSpinBox()
        self.freq_edit.setRange(20, 20000)
        self.freq_edit.setValue(freq)
        self.freq_edit.setSuffix(" Hz")
        layout.addRow("Frequency:", self.freq_edit)
        
        # Level input
        self.level_edit = QDoubleSpinBox()
        self.level_edit.setRange(-120, 20)
        self.level_edit.setValue(level)
        self.level_edit.setSuffix(" dB")
        layout.addRow("Level:", self.level_edit)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self) -> Tuple[float, float]:
        # Store values for next time
        PointEditDialog.last_freq = self.freq_edit.value()
        PointEditDialog.last_level = self.level_edit.value()
        return (self.freq_edit.value(), self.level_edit.value())

class OverlayEditDialog(QDialog):
    def __init__(self, parent=None, template=None):
        super().__init__(parent)
        self.template = template
        self.setModal(True)
        self.init_ui()
        if self.template:
            self.load_template(self.template)  # Load existing template data

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
            
            button_layout.addWidget(ok_btn)
            
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

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
                    break  # Exit loop on successful add
            else:
                break  # User cancelled

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

    def remove_point(self):
        current = self.points_list.currentRow()
        if current >= 0:
            self.points_list.takeItem(current)
            self.points = self.get_points()

    def get_points(self) -> List[Tuple[float, float]]:
        points = []
        for i in range(self.points_list.count()):
            item = self.points_list.item(i)
            points.append(item.data(Qt.ItemDataRole.UserRole))
        return sorted(points, key=lambda x: x[0])  # Sort by frequency

    def get_template(self) -> OverlayTemplate:
        return OverlayTemplate(
            name=self.name_edit.text(),
            color=self.color_combo.currentData(),
            points=self.get_points()
        )

    def load_template(self, template: OverlayTemplate):
        self.name_edit.setText(template.name)
        index = self.color_combo.findData(template.color)
        if index >= 0:
            self.color_combo.setCurrentIndex(index)
        
        self.points_list.clear()
        for freq, level in template.points:
            item = QListWidgetItem(f"{freq} Hz, {level} dB")
            item.setData(Qt.ItemDataRole.UserRole, (freq, level))
            self.points_list.addItem(item)

    def edit_point(self, item):
        """Handle editing point via dialog"""
        if not item:
            return
            
        freq, level = item.data(Qt.ItemDataRole.UserRole)
        dialog = PointEditDialog(freq, level, False, self)
        
        while True:  # Keep dialog open until valid edit or cancel
            if dialog.exec() == QDialog.DialogCode.Accepted:
                new_freq, new_level = dialog.get_values()
                
                # Check for duplicate frequency
                duplicate = False
                for i in range(self.points_list.count()):
                    if i != self.points_list.row(item):
                        other_freq = self.points_list.item(i).data(Qt.ItemDataRole.UserRole)[0]
                        if abs(other_freq - new_freq) < 0.1:
                            QMessageBox.warning(self, "Error", 
                                "A point at this frequency already exists!")
                            duplicate = True
                            break
                
                if not duplicate:
                    # Update point
                    item.setText(f"{new_freq} Hz, {new_level} dB")
                    item.setData(Qt.ItemDataRole.UserRole, (new_freq, new_level))
                    self.sort_points()
                    break  # Exit loop on successful edit
            else:
                break

    def add_point(self):
        """Add new point via dialog"""
        dialog = PointEditDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            freq, level = dialog.get_values()
            
            # Check for duplicate frequency
            for i in range(self.points_list.count()):
                other_freq = self.points_list.item(i).data(Qt.ItemDataRole.UserRole)[0]
                if abs(other_freq - freq) < 0.1:  # Small threshold for float comparison
                    QMessageBox.warning(self, "Error", 
                        "A point at this frequency already exists!")
                    return
            
            # Add new point
            item = QListWidgetItem(f"{freq} Hz, {level} dB")
            item.setData(Qt.ItemDataRole.UserRole, (freq, level))
            self.points_list.addItem(item)
            self.points = self.get_points()

class OverlayManager(QGroupBox):
    overlay_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Overlay Templates", parent)
        self.templates = []
        self.max_overlays = 5
        self.colors = ['#ff0000', '#00ff00', '#0000ff', '#ff00ff', '#00ffff']
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
            
            # Template name
            name_label = QLabel(template.name)
            item_layout.addWidget(name_label, stretch=1)
            
            # Offset control with smaller width
            offset_spin = QSpinBox()
            offset_spin.setRange(-50, 50)
            offset_spin.setValue(template.offset)
            offset_spin.setSuffix(" dB")
            offset_spin.setFixedWidth(70)  # Make spinner more compact
            offset_spin.valueChanged.connect(lambda value, t=template: self.update_offset(t, value))
            item_layout.addWidget(offset_spin)

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

    def toggle_template(self, template, enabled):
        template.enabled = enabled
        self.overlay_changed.emit()

    def update_offset(self, template, offset):
        template.offset = offset
        self.overlay_changed.emit()

    def add_template(self):
        if len(self.templates) >= self.max_overlays:
            QMessageBox.warning(self, "Error", "Maximum number of overlays reached")
            return
            
        dialog = OverlayEditDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            template = dialog.get_template()
            self.templates.append(template)
            self.update_list()
            self.overlay_changed.emit()

    def edit_template(self, index):
        """Edit an existing overlay template"""
        if 0 <= index < len(self.templates):
            dialog = OverlayEditDialog(self, self.templates[index])  # Pass the template
            result = dialog.exec()
            if result == QDialog.DialogCode.Accepted:
                new_template = dialog.get_template()
                new_template.offset = self.templates[index].offset  # Preserve offset
                self.templates[index] = new_template
                self.update_list()
                self.overlay_changed.emit()
            elif result == 2:  # Save as new
                template = dialog.get_template()
                if len(self.templates) < self.max_overlays:
                    self.templates.append(template)
                else:
                    QMessageBox.warning(self, "Error", "Maximum number of overlays reached")
            
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

    def get_templates(self):
        return self.templates

class CppTemplateDialog(QDialog):
    template_changed = pyqtSignal(dict)  # Add signal

    def __init__(self, template_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("C++ Template Editor")
        self.setModal(True)
        self.template_data = template_data
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Help text
        help_text = QLabel(
            "Define the template for generating C++ code.\n"
            "Available placeholders: {var_name}, {length_name}, {array_data}"
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
    file_menu.addSeparator()
    file_menu.addAction("Export &White Noise...", parent.export_white_noise)
    file_menu.addAction("&Exit", parent.close)
    menubar.addMenu(file_menu)
    
    return menubar
    file_menu.addSeparator()
    file_menu.addAction("&Load Settings", parent.load_settings)
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
    file_menu.addSeparator()
    file_menu.addAction("Export &White Noise...", parent.export_white_noise)
    file_menu.addSeparator()
    file_menu.addAction("&Exit", parent.close)
    menubar.addMenu(file_menu)
    
    return menubar