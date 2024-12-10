# spectrum_analyzer_ui.py
import sys
import os  # Make sure this is imported
import numpy as np
import json
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import *  # For UI components
from PyQt6.QtCore import QTimer, Qt, QSettings
from PyQt6.QtGui import QAction  # Add this import
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf  # Add this import at the top
from scipy.interpolate import splrep, splev  # Add this import
import time  # Add this import

# Local imports
from config import AudioConfig
from processor import AudioProcessor
from audio_sources import NoiseSource, MonitoredInputSource, AudioExporter
from filters import BandpassFilter, LowpassFilter, HighpassFilter, NotchFilter, GaussianFilter, ParabolicFilter  # Add import
from ui_components import (
    SourcePanel, AnalyzerPanel, FilterPanel, 
    create_menu_bar, StatusBar, ExportDialog, SpectralComponentsPanel,
    HearingTestDialog, OverlayManager, BufferSettingsDialog, OverlayTemplate  # Add OverlayTemplate
)


class SpectrumAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Add before other init code
        self.current_file = None
        self.has_unsaved_changes = False
        self.recent_files = []
        self.max_recent_files = 5
        self.export_settings = {}  # Store last used export settings
        
        # Load recent files list from settings
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        self.recent_files = settings.value('recent_files', [], str)

        self.title = "Noise Shaper"
        
        # Continue with existing init code
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 700)

        # Initialize configuration and processor
        self.config = AudioConfig()
        self.processor = AudioProcessor(self.config)
        
        # Initialize overlay manager first
        self.overlay_manager = OverlayManager()
        self.overlay_manager.overlay_changed.connect(self.update_overlays)
        
        # Setup UI components
        self.init_ui()
        
        # Setup update timer
        self.timer = QTimer()
        self.timer.setInterval(30)  # ~33 fps
        self.timer.timeout.connect(self.update_plot)

        self.filter_panel.filter_updated.connect(self.update_filter)
        self.source_panel.export_requested.connect(self.export_noise)

        # Connect status callbacks based on mode
        self.config.on_underflow = self.source_panel.monitoring_panel.set_underflow
        self.config.on_overflow = self.source_panel.monitoring_panel.set_overflow
        
        # Connect buffer settings
        self.source_panel.monitoring_panel.settings_clicked.connect(self.show_buffer_settings)

        # Remove status indicators from statusbar - they're now in monitoring panel for test mode only

        # Call handle_mode_change with the initial mode
        initial_mode = self.source_panel.source_type.currentText()
        self.handle_mode_change(initial_mode)

        # Connect settings change signals
        self.analyzer_panel.settings_changed.connect(self.on_settings_changed)
        self.source_panel.source_changed.connect(self.on_settings_changed)
        self.filter_panel.filter_updated.connect(lambda *_: self.on_settings_changed())
        self.filter_panel.filter_removed.connect(lambda *_: self.on_settings_changed())
        self.filter_panel.filter_parameters.connect(lambda *_: self.on_settings_changed())

        # Add before calling init_ui()
        self.load_stored_settings()

        # Apply any saved export settings
        self.export_settings = QSettings('YourOrg', 'SpectrumAnalyzer').value('export_settings', {}, dict)

    def load_stored_settings(self):
        """Load persistent settings from QSettings"""
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        self.recent_files = settings.value('recent_files', [], str)
        self.export_settings = settings.value('export_settings', {}, dict)

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create control panel
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # Create graph panel
        graph_panel = self.create_graph_panel()
        main_layout.addWidget(graph_panel, stretch=2)

        # Create menu bar
        self.menubar = self.create_menu_bar()
        self.setMenuBar(self.menubar)  # This line was missing

        # Create status bar
        self.statusbar = StatusBar()
        self.setStatusBar(self.statusbar)

    def create_menu_bar(self) -> QMenuBar:
        menubar = QMenuBar()
        
        # File menu
        file_menu = QMenu("&File", self)
        
        # Create actions with shortcuts
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_session)
        
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_settings)
        
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_settings)
        
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_settings_as)
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Add actions to menu
        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        
        # Add Recent Files submenu
        self.recent_menu = QMenu("Recent Files", self)
        self.update_recent_menu()
        file_menu.addMenu(self.recent_menu)
        
        file_menu.addSeparator()
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        menubar.addMenu(file_menu)
        return menubar

    def new_session(self):
        if self.check_unsaved_changes():
            self.reset_to_defaults()
            self.current_file = None
            self.has_unsaved_changes = False
            self.update_window_title()

    def reset_to_defaults(self):
        # Reset all panels to default values
        self.analyzer_panel.apply_settings({})
        self.source_panel.apply_settings({})
        self.filter_panel.apply_settings({})
        
        # Clear overlays
        self.overlay_manager.templates.clear()
        self.overlay_manager.update_list()
        self.update_overlays()
        
        self.update_analyzer_settings()
        
        # Clear export settings
        self.export_settings = {}
        
        # Reset export settings in source panel
        if hasattr(self.source_panel, 'export_settings'):
            self.source_panel.export_settings = {}

        # Reset amplitudes to defaults
        self.config.amp_whitenoise = 0.5
        self.config.amp_spectral = 1.0
        
        # Update amplitude control based on current mode
        source_type = self.source_panel.source_type.currentText()
        if source_type == "White Noise":
            self.source_panel.amplitude_control.setValue(0.5)
        elif source_type == "Spectral Synthesis":
            self.source_panel.amplitude_control.setValue(1.0)

    def load_settings(self):
        if not self.check_unsaved_changes():
            return
            
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON Files (*.json);;All Files (*)")
        
        if filename:
            self.load_settings_file(filename)

    def load_settings_file(self, filename):
        try:
            with open(filename, 'r') as f:
                settings = json.load(f)
            
            # First apply filter settings to processor
            if 'filters' in settings:
                # Clear existing filters first
                self.processor.filters.clear()
                if self.processor.source:
                    self.processor.source.filters.clear()
                
                # Add each filter from settings, with type preserved
                for filter_settings in settings['filters'].get('filters', []):
                    # Create a copy to avoid modifying original
                    params = filter_settings.copy()
                    # Ensure type exists before calling add_filter
                    if 'type' in params:
                        self.add_filter(params)  # add_filter will handle type removal
                    
            # Then apply UI panel settings
            self.analyzer_panel.apply_settings(settings.get('analyzer', {}))
            self.source_panel.apply_settings(settings.get('source', {}))
            self.filter_panel.apply_settings(settings.get('filters', {}))
            
            # Load overlay templates
            self.overlay_manager.templates.clear()
            for t in settings.get('overlays', []):
                template = OverlayTemplate(
                    name=t['name'],
                    color=t['color'],
                    points=t['points']
                )
                template.enabled = t.get('enabled', True)
                template.offset = t.get('offset', 0)
                self.overlay_manager.templates.append(template)
            self.overlay_manager.update_list()
            
            # Load export settings and update source panel
            if 'export' in settings:
                self.export_settings = settings['export']
                if hasattr(self.source_panel, 'export_settings'):
                    self.source_panel.export_settings = settings['export'].copy()
            
            # Load cpp template settings
            if 'cpp_template' in settings:
                self.source_panel.cpp_template = settings['cpp_template']
            
            # Update UI
            self.update_analyzer_settings()
            self.update_overlays()
            self.current_file = filename
            self.has_unsaved_changes = False
            self.add_recent_file(filename)
            self.update_window_title()
            self.statusbar.showMessage(f"Settings loaded from {filename}")
            
        except Exception as e:
            self.show_error("Load Error", f"Error loading settings: {str(e)}")

    def save_settings(self):
        if not self.current_file:
            return self.save_settings_as()
        return self.save_settings_to_file(self.current_file)

    def save_settings_as(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json);;All Files (*)")
        
        if filename:
            # Add .json extension if not present
            if not filename.lower().endswith('.json'):
                filename += '.json'
            return self.save_settings_to_file(filename)
        return False

    def save_settings_to_file(self, filename):
        try:
            # Get overlay settings
            overlay_settings = [
                {
                    'name': t.name,
                    'color': t.color,
                    'points': t.points,
                    'enabled': t.enabled,
                    'offset': t.offset
                }
                for t in self.overlay_manager.templates
            ]
            
            # Get the latest export settings from source panel
            if hasattr(self.source_panel, 'export_settings'):
                self.export_settings = self.source_panel.export_settings.copy()
            
            settings = {
                'version': '1.0.0a',  # Add version info
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),  # Add timestamp
                'analyzer': self.analyzer_panel.get_current_settings(),
                'source': self.source_panel.get_current_settings(),
                'filters': self.filter_panel.get_current_settings(),
                'overlays': overlay_settings,
                'export': self.export_settings,
                'cpp_template': self.source_panel.cpp_template  # Add this line
            }
            
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.current_file = filename
            self.has_unsaved_changes = False
            self.add_recent_file(filename)
            self.update_window_title()
            self.statusbar.showMessage(f"Settings saved to {filename}")
            return True
            
        except Exception as e:
            self.show_error("Save Error", f"Error saving settings: {str(e)}")
            return False

    def add_recent_file(self, filename):
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        self.recent_files.insert(0, filename)
        while len(self.recent_files) > self.max_recent_files:
            self.recent_files.pop()
        self.update_recent_menu()
        
        # Save to QSettings
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        settings.setValue('recent_files', self.recent_files)

    def update_recent_menu(self):
        self.recent_menu.clear()
        for i, filename in enumerate(self.recent_files):
            action = self.recent_menu.addAction(os.path.basename(filename))
            action.setData(filename)
            action.triggered.connect(lambda checked, f=filename: self.load_settings_file(f))

    def update_window_title(self):
        title = self.title  # Use self.title instead of hardcoded string
        if self.current_file:
            title = f"{os.path.basename(self.current_file)} - {title}"
        if self.has_unsaved_changes:
            title = f"*{title}"
        self.setWindowTitle(title)

    def check_unsaved_changes(self):
        if not self.has_unsaved_changes:
            return True
            
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.StandardButton.Save | 
            QMessageBox.StandardButton.Discard | 
            QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Save:
            return self.save_settings()
        elif reply == QMessageBox.StandardButton.Cancel:
            return False
        return True

    def mark_unsaved_changes(self):
        self.has_unsaved_changes = True
        self.update_window_title()

    def show_error(self, title: str, message: str):
        """Shows an error dialog"""
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        """Handle window close events (X button or Alt+F4)"""
        if not self.check_unsaved_changes():
            event.ignore()
            return
            
        # Save settings before closing
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        settings.setValue('recent_files', self.recent_files)
        settings.setValue('export_settings', self.export_settings)
        
        try:
            # Stop any active audio/processing
            if self.timer.isActive():
                self.timer.stop()
            if self.source_panel.is_playing:
                self.source_panel.toggle_playback()
            self.stop_processing()
            
            # Close processor
            if hasattr(self, 'processor'):
                self.processor.close()

            # Clean up plot
            if hasattr(self, 'plot_curve'):
                self.graph_widget.removeItem(self.plot_curve)
                self.plot_curve = None
                
        except Exception as e:
            print(f"Shutdown error: {e}")
        finally:
            event.accept()

    def setup_graph(self):
        """Initializes the PyQtGraph plot widget"""
        self.graph_widget.setBackground('w')
        self.graph_widget.showGrid(x=True, y=True)
        self.graph_widget.setLabel('left', 'Magnitude (dB)')
        self.graph_widget.setLabel('bottom', 'Frequency (Hz)')
        
        # Create filled curve style
        pen = pg.mkPen(color=(80, 40, 200), width=2)
        brush = pg.mkBrush(color=(100, 50, 255, 50))
        
        # Create the plot curve with fill
        self.plot_curve = pg.PlotDataItem(
            fillLevel=-90,
            brush=brush,
            pen=pen
        )
        self.graph_widget.addItem(self.plot_curve)
        
        # Initial setup
        self.update_graph_scale()

    def update_plot(self):
        """Updates the spectrum plot with proper scaling"""
        try:
            freq, spec_db = self.processor.process()
            if len(freq) == 0 or len(spec_db) == 0:  # Check for empty arrays
                return

            # Apply averaging if enabled
            if self.analyzer_panel.averaging.value() > 1:
                if not hasattr(self, '_prev_spec') or self._prev_spec.shape != spec_db.shape:
                    self._prev_spec = spec_db
                else:
                    alpha = 1.0 / self.analyzer_panel.averaging.value()
                    spec_db = alpha * spec_db + (1 - alpha) * self._prev_spec
                self._prev_spec = spec_db.copy()

            # For log mode, ensure we don't have zero frequencies
            if self.analyzer_panel.scale_type.currentText().lower() == 'logarithmic':
                mask = freq > 0
                if np.any(mask):
                    freq = freq[mask]
                    spec_db = spec_db[mask]
            
            # Update the plot - pyqtgraph handles the scaling
            self.plot_curve.setData(freq, spec_db)
            
        except Exception as e:
            print(f"Plot error details: {str(e)}")
            # Don't update plot if there's an error

    def update_analyzer_settings(self):
        """Updates analyzer settings when changed in the UI"""
        try:
            # Update configuration
            new_settings = self.analyzer_panel.get_current_settings()
            for key, value in new_settings.items():
                setattr(self.config, key, value)

            # Update processor
            self.processor.update_window()
            
            # Force graph update
            self.update_graph_scale()
            
            # Clear previous spectrum data to avoid averaging issues across scale changes
            if hasattr(self, '_prev_spec'):
                delattr(self, '_prev_spec')
            
        except Exception as e:
            print(f"Settings update error: {str(e)}")
            self.show_error("Settings Error", f"Error updating analyzer settings: {str(e)}")

    def update_graph_scale(self):
        """Updates graph scaling and ticks based on scale type"""
        try:
            scale_type = self.analyzer_panel.scale_type.currentText().lower()
            is_log = scale_type == 'logarithmic'
            
            # Configure X axis
            ax = self.graph_widget.getAxis('bottom')
            
            if is_log:
                # Set log mode first
                self.graph_widget.setLogMode(x=True, y=False)
                
                # Set range for log scale (after setting log mode)
                self.graph_widget.setXRange(np.log10(20), np.log10(20000))
                
                # Log scale ticks - use log values for positions
                major_ticks = [
                    (np.log10(freq), label) for freq, label in [
                        (20, '20'), (100, '100'), (1000, '1k'), 
                        (10000, '10k'), (20000, '20k')
                    ]
                ]
                
                minor_ticks = [
                    (np.log10(freq), str(freq)) for freq in [
                    30, 40, 50, 60, 70, 80, 90,
                    200, 300, 400, 500, 600, 700, 800, 900,
                    2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000
                    ]
                ]
                
                ax.setTicks([major_ticks, minor_ticks])
            else:
                # Set log mode first for linear scale
                self.graph_widget.setLogMode(x=False, y=False)
                
                # Set range for linear scale
                self.graph_widget.setXRange(0, 20000)
                
                # Linear scale ticks
                major_ticks = [
                    (0, '0'), (5000, '5k'), (10000, '10k'),
                    (15000, '15k'), (20000, '20k')
                ]
                minor_ticks = [
                    (i * 1000, str(i)) for i in range(1, 20) 
                    if i % 5 != 0
                ]
                
                ax.setTicks([major_ticks, minor_ticks])
            
            # Update Y axis
            self.graph_widget.setYRange(self.config.min_db, self.config.max_db)
            ay = self.graph_widget.getAxis('left')
            ay.setTicks([[(x, f"{x}") for x in range(self.config.min_db, self.config.max_db + 1, 10)]])
            
        except Exception as e:
            print(f"Scale update error: {e}")

        # Update overlays after scale change
        self.update_overlays()

    def create_control_panel(self) -> QWidget:
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)
        
        # Source settings panel
        self.source_panel = SourcePanel(self.config)
        self.source_panel.source_changed.connect(self.handle_source_change)
        layout.addWidget(self.source_panel)

        # Analyzer settings panel (always visible)
        self.analyzer_panel = AnalyzerPanel(self.config)
        self.analyzer_panel.settings_changed.connect(self.update_analyzer_settings)
        layout.addWidget(self.analyzer_panel)

        # Filter panel (only for white noise)
        self.filter_panel = FilterPanel(self.config, self.processor)
        self.filter_panel.filter_parameters.connect(self.add_filter)
        self.filter_panel.filter_removed.connect(self.remove_filter)
        self.filter_panel.filter_updated.connect(self.update_filter)
        
        # Parabola panel (only for parabolic noise)
        self.parabola_panel = SpectralComponentsPanel(self.processor)  # Updated name
        self.parabola_panel.parabola_added.connect(self.add_parabola)
        self.parabola_panel.parabola_removed.connect(self.remove_parabola)
        self.parabola_panel.parabola_updated.connect(self.update_parabola)
        
        # Add both panels but hide them initially
        layout.addWidget(self.filter_panel, stretch=1)  # Add stretch=1
        layout.addWidget(self.parabola_panel, stretch=1)  # Add stretch=1
        self.parabola_panel.hide()

        # Show/hide appropriate panels based on source type
        self.source_panel.source_type.currentTextChanged.connect(self.handle_mode_change)
        
        # Add overlay manager after other panels
        layout.addWidget(self.overlay_manager)
        
        # Set size policy for control panel
        control_panel.setFixedWidth(300)
        control_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        return control_panel

    def update_hearing_test_offset(self, value):
        """Update the overlay offset when the slider is changed"""
        self.hearing_test_offset = value
        self.update_hearing_test_overlay()

    def handle_mode_change(self, mode: str):
        """Show/hide panels based on selected mode"""
        self.filter_panel.setVisible(mode == "White Noise")
        self.parabola_panel.setVisible(mode == "Spectral Synthesis")
        
        # Update status indicators based on mode
        is_test_mode = mode == "Test Mode"
        self.source_panel.monitoring_panel.overflow_indicator.setVisible(is_test_mode)
        self.source_panel.monitoring_panel.underflow_indicator.setVisible(True)  # Always show UF
        
        # Stop playback if running
        if self.source_panel.is_playing:
            self.source_panel.toggle_playback()

    def create_graph_panel(self):
        """Creates the graph panel with the spectrum display"""
        graph_panel = QWidget()
        layout = QVBoxLayout(graph_panel)
        
        # Create plot widget
        self.graph_widget = pg.PlotWidget()
        self.setup_graph()
        layout.addWidget(self.graph_widget)
        
        return graph_panel


    def update_graph_ranges(self):
        """Updates the graph axis ranges based on current settings"""
        self.graph_widget.setYRange(self.config.min_db, self.config.max_db)
        self.graph_widget.setXRange(self.config.min_frequency, self.config.max_frequency)


    def handle_source_change(self):
        """Handles changes in the audio source selection"""
        try:
            # Stop any current processing first
            self.stop_processing()

            # Only create new source if we're supposed to be playing
            if self.source_panel.is_playing:
                # Get current amplitude from UI
                current_amplitude = self.source_panel.amplitude_control.value()
                
                source_type = self.source_panel.get_source_type()
                if source_type == "Test Mode":
                    source = MonitoredInputSource(self.config)
                else:
                    noise_type = source_type.lower().replace(" synthesis", "").replace(" noise", "")
                    source = NoiseSource(self.config, noise_type)
                    # Set initial amplitude
                    source.generator.amplitude = current_amplitude
                    
                    # If it's spectral synthesis, add any existing components
                    if noise_type == 'spectral':
                        for widget in self.parabola_panel.parabolas:
                            source.add_parabola(widget.get_parameters())
                
                # Store source reference in source panel
                self.source_panel.handle_source_reference(source)
                self.processor.set_source(source)
                self.start_processing()
                
        except Exception as e:
            self.show_error("Source Error", f"Error changing audio source: {str(e)}")
            print(f"Error changing audio source: {str(e)}")
            self.source_panel.is_playing = False
            self.source_panel.play_button.setText("Play")
            self.source_panel.play_button.setChecked(False)

    def add_filter(self, filter_params: Dict[str, Any]):
        """Adds a new filter"""
        try:
            filter_type = filter_params.pop('type')
            if filter_type == 'bandpass':
                filter_ = BandpassFilter(self.config, **filter_params)
            elif filter_type == 'lowpass':
                filter_ = LowpassFilter(self.config, **filter_params)
            elif filter_type == 'highpass':
                filter_ = HighpassFilter(self.config, **filter_params)
            elif filter_type == 'notch':
                filter_ = NotchFilter(self.config, **filter_params)
            elif filter_type == 'gaussian':
                filter_ = GaussianFilter(self.config, **filter_params)
            elif filter_type == 'parabolic':  # Add parabolic filter support
                filter_ = ParabolicFilter(self.config, **filter_params)
            
            self.processor.add_filter(filter_)
            if (self.processor.source and 
                isinstance(self.processor.source, NoiseSource)):
                self.processor.source.add_filter(filter_)
                
        except Exception as e:
            self.show_error("Filter Error", f"Error adding filter: {str(e)}")

    def remove_filter(self, index: int):
        """Removes the filter at the specified index"""
        try:
            self.processor.remove_filter(index)
            
            # Remove from source if it's a noise source
            if (self.processor.source is not None and 
                self.processor.source.__class__.__name__ == 'NoiseSource'):
                self.processor.source.remove_filter(index)
                
        except Exception as e:
            self.show_error("Filter Error", f"Error removing filter: {str(e)}")

    def update_filter(self, index: int, params: dict):
        """Update filter parameters"""
        try:
            # Update processor filter
            self.processor.update_filter(index, params)
            
            # If using a noise source, update its filters too
            if (self.processor.source and 
                isinstance(self.processor.source, NoiseSource)):
                self.processor.source.update_filter(index, params)
                
        except Exception as e:
            self.show_error("Filter Error", f"Error updating filter: {str(e)}")

    def add_parabola(self, params: Dict[str, Any]):
        """Handle adding a new parabola"""
        try:
            if self.processor.source and isinstance(self.processor.source, NoiseSource):
                self.processor.source.add_parabola(params)
        except Exception as e:
            self.show_error("Parabola Error", f"Error adding parabola: {str(e)}")

    def remove_parabola(self, index: int):
        """Handle removing a parabola"""
        try:
            if self.processor.source and isinstance(self.processor.source, NoiseSource):
                self.processor.source.remove_parabola(index)
        except Exception as e:
            self.show_error("Parabola Error", f"Error removing parabola: {str(e)}")

    def update_parabola(self, index: int, params: dict):
        """Handle updating parabola parameters"""
        try:
            if self.processor.source and isinstance(self.processor.source, NoiseSource):
                self.processor.source.update_parabola(index, params)
        except Exception as e:
            self.show_error("Parabola Error", f"Error updating parabola: {str(e)}")

    def update_spectral_normalization(self, enabled: bool):
        """Handle spectral normalization changes"""
        if isinstance(self.processor.source, NoiseSource):
            self.processor.source.set_spectral_normalization(enabled)

    def update_filter_normalization(self, enabled: bool):
        """Handle white noise filter normalization changes"""
        if isinstance(self.processor.source, NoiseSource):
            self.processor.source.set_filter_normalization(enabled)

    def start_processing(self):
        """Starts the audio processing"""
        if not self.timer.isActive():
            self.timer.start()
            self.statusbar.showMessage("Processing started")

    def stop_processing(self):
        """Stops the audio processing"""
        if self.timer.isActive():
            self.timer.stop()
            self.processor.close()
            self.statusbar.showMessage("Processing stopped")

    def save_settings(self):
        if not self.current_file:
            return self.save_settings_as()
        return self.save_settings_to_file(self.current_file)

    def save_settings_as(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", "", "JSON Files (*.json);;All Files (*)")
        
        if filename:
            # Add .json extension if not present
            if not filename.lower().endswith('.json'):
                filename += '.json'
            return self.save_settings_to_file(filename)
        return False

    def save_settings_to_file(self, filename):
        try:
            # Get overlay settings
            overlay_settings = [
                {
                    'name': t.name,
                    'color': t.color,
                    'points': t.points,
                    'enabled': t.enabled,
                    'offset': t.offset
                }
                for t in self.overlay_manager.templates
            ]
            
            # Get the latest export settings from source panel
            if hasattr(self.source_panel, 'export_settings'):
                self.export_settings = self.source_panel.export_settings.copy()
            
            settings = {
                'version': '1.0.0a',  # Add version info
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),  # Add timestamp
                'analyzer': self.analyzer_panel.get_current_settings(),
                'source': self.source_panel.get_current_settings(),
                'filters': self.filter_panel.get_current_settings(),
                'overlays': overlay_settings,
                'export': self.export_settings,
                'cpp_template': self.source_panel.cpp_template  # Add this line
            }
            
            with open(filename, 'w') as f:
                json.dump(settings, f, indent=4)
            
            self.current_file = filename
            self.has_unsaved_changes = False
            self.add_recent_file(filename)
            self.update_window_title()
            self.statusbar.showMessage(f"Settings saved to {filename}")
            return True
            
        except Exception as e:
            self.show_error("Save Error", f"Error saving settings: {str(e)}")
            return False

    def add_recent_file(self, filename):
        if filename in self.recent_files:
            self.recent_files.remove(filename)
        self.recent_files.insert(0, filename)
        while len(self.recent_files) > self.max_recent_files:
            self.recent_files.pop()
        self.update_recent_menu()
        
        # Save to QSettings
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        settings.setValue('recent_files', self.recent_files)

    def update_recent_menu(self):
        self.recent_menu.clear()
        for i, filename in enumerate(self.recent_files):
            action = self.recent_menu.addAction(os.path.basename(filename))
            action.setData(filename)
            action.triggered.connect(lambda checked, f=filename: self.load_settings_file(f))

    def update_window_title(self):
        title = self.title  # Use self.title instead of hardcoded string
        if self.current_file:
            title = f"{os.path.basename(self.current_file)} - {title}"
        if self.has_unsaved_changes:
            title = f"*{title}"
        self.setWindowTitle(title)

    def check_unsaved_changes(self):
        if not self.has_unsaved_changes:
            return True
            
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to save them?",
            QMessageBox.StandardButton.Save | 
            QMessageBox.StandardButton.Discard | 
            QMessageBox.StandardButton.Cancel
        )
        
        if reply == QMessageBox.StandardButton.Save:
            return self.save_settings()
        elif reply == QMessageBox.StandardButton.Cancel:
            return False
        return True

    def mark_unsaved_changes(self):
        self.has_unsaved_changes = True
        self.update_window_title()

    def closeEvent(self, event):
        if not self.check_unsaved_changes():
            event.ignore()
            return
        
        # Continue with existing cleanup
        try:
            if self.timer.isActive():
                self.timer.stop()
            # Stop playback if running
            if hasattr(self, 'source_panel') and self.source_panel.is_playing:
                self.source_panel.toggle_playback()

            # Stop processing
            self.stop_processing()

            # Close processor
            if hasattr(self, 'processor'):
                self.processor.close()

            # Delete plot curve to prevent Qt warnings
            if hasattr(self, 'plot_curve'):
                self.graph_widget.removeItem(self.plot_curve)
                self.plot_curve = None

        except Exception as e:
            print(f"Shutdown error: {str(e)}")
        finally:
            event.accept()  # Always accept close event

    def export_white_noise(self):
        """Menu callback that triggers export dialog in white noise mode"""
        self.source_panel.source_type.setCurrentText("White Noise")
        self.source_panel.export_noise()

    # Remove export_spectral_noise method since it's handled by source_type in export_noise

    def export_noise(self, settings: dict):
        try:
            # Store settings for reuse
            self.export_settings = settings.copy()
            
            # Save to persistent storage
            app_settings = QSettings('YourOrg', 'SpectrumAnalyzer')
            app_settings.setValue('export_settings', self.export_settings)
            
            # Mark as changed to ensure it gets saved with project
            self.mark_unsaved_changes()
            
            base_path = settings['folder_path']
            wav_path = os.path.join(base_path, settings['wav_filename'])
            header_path = os.path.join(base_path, settings['header_filename'])

            # Generate samples based on source type
            source_type = self.source_panel.get_source_type()
            noise_type = source_type.lower().replace(" synthesis", "").replace(" noise", "")
            temp_source = NoiseSource(self.config, noise_type)
            
            # Set RNG type from source panel
            rng_type = self.source_panel.rng_type.currentText().lower().replace(' ', '_')
            temp_source.set_rng_type(rng_type)

            try:
                # Ensure normalization is properly set for the temporary source
                if noise_type == 'spectral':
                    temp_source.generator.normalize = settings['enable_normalization']
                    temp_source.generator.normalize_value = settings['normalize_value']
                else:
                    temp_source.generator.normalize = settings['enable_normalization']
                    temp_source.generator.normalize_value = settings['normalize_value']

                # Set the seed before generating
                if settings.get('seed') is not None:
                    temp_source.generator.set_seed(settings['seed'])
                else:
                    temp_source.generator.set_seed(None)  # Will use random seed

                # Copy current settings
                if source_type == "White Noise":
                    for filter_ in self.processor.filters:
                        temp_source.add_filter(filter_)
                elif source_type == "Spectral Synthesis":
                    for widget in self.parabola_panel.parabolas:
                        temp_source.add_parabola(widget.get_parameters())

                # Generate the signal - set normalize to use export dialog setting
                signal = temp_source.export_signal(
                    duration=settings['duration']/1000.0,  # Convert ms to seconds
                    sample_rate=settings['sample_rate'],
                    amplitude=settings['amplitude'],
                    enable_fade=settings['enable_fade'],
                    fade_in_duration=settings['fade_in_duration'],
                    fade_out_duration=settings['fade_out_duration'],
                    fade_in_power=settings['fade_in_power'],
                    fade_out_power=settings['fade_out_power'],
                    enable_normalization=settings['enable_normalization'],  # Use dialog setting
                    normalize_value=settings['normalize_value']  # Add this line
                )

                # Save WAV file if requested
                if settings['export_wav']:
                    sf.write(wav_path, signal, settings['sample_rate'])

                # Generate and save C++ code if requested
                if settings['export_cpp']:
                    cpp_code = AudioExporter.generate_cpp_code(signal, settings)
                    with open(header_path, 'w') as f:
                        f.write(cpp_code)
                    
                self.statusbar.showMessage(f"Export complete: {wav_path}")

            finally:
                temp_source.close()

        except ValueError as e:
            self.show_error("Export Error", str(e))  # Changed from _handle_error to show_error
        except Exception as e:
            self.show_error("Export Error", f"Error exporting noise: {str(e)}")  # Changed here too

    def update_overlays(self):
        """Update all overlay curves"""
        # Remove existing overlay curves
        for item in self.graph_widget.items():
            if isinstance(item, pg.PlotDataItem) and item != self.plot_curve:
                self.graph_widget.removeItem(item)
        
        # Add curves for each template
        for template in self.overlay_manager.get_templates():
            if not template.enabled or not template.points:
                continue
                
            # Get points and apply offset
            freqs, levels = zip(*template.points)
            freqs = np.array(freqs)
            levels = np.array(levels) + template.offset

            # Apply log transform if needed
            if self.analyzer_panel.scale_type.currentText().lower() == 'logarithmic':
                freqs = np.maximum(freqs, 1)
            
            try:
                # Create line curve with points
                curve = pg.PlotDataItem(
                    freqs, levels,
                    pen=pg.mkPen(color=template.color, width=2),
                    symbol='o',
                    symbolSize=6,
                    symbolBrush=template.color,
                    symbolPen=template.color
                )
                
                self.graph_widget.addItem(curve)
                
            except Exception as e:
                print(f"Error creating overlay curve: {e}")
                continue

    def show_buffer_settings(self):
        """Show the buffer settings dialog"""
        dialog = BufferSettingsDialog(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            settings = dialog.get_settings()
            # Update config
            self.config.input_buffer_size = settings['input_buffer_size']
            self.config.output_buffer_size = settings['output_buffer_size']
            self.config.chunk_size = settings['chunk_size']
            # Restart audio if needed
            if self.source_panel.is_playing:
                self.source_panel.toggle_playback()
                self.source_panel.toggle_playback()

    def on_settings_changed(self):
        """Called when any settings change"""
        self.mark_unsaved_changes()

def main():
    app = QApplication(sys.argv)
    window = SpectrumAnalyzerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()