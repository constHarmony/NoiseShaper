# spectrum_analyzer_ui.py
import sys
import os  # Make sure this is imported
import numpy as np
import json
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import *  # For UI components
from PyQt6.QtCore import QTimer, Qt, QSettings, QEvent  # Add QEvent
from PyQt6.QtGui import QAction  # Add this import
import pyqtgraph as pg
import sounddevice as sd
import soundfile as sf  # Add this import at the top
from scipy.interpolate import splrep, splev  # Add this import
import time  # Add this import
import logging
from PyQt6.QtWidgets import QStyleFactory

# Local imports
from config import AudioConfig, VERSION  # Add VERSION import
from processor import AudioProcessor
from audio_sources import NoiseSource, MonitoredInputSource, AudioExporter  # Add this import
from filters import (BandpassFilter, LowpassFilter, HighpassFilter, 
                    NotchFilter, GaussianFilter, ParabolicFilter, PlateauFilter)  # Add PlateauFilter
from ui_components import (
    SourcePanel, AnalyzerPanel, FilterPanel, 
    create_menu_bar, StatusBar, ExportDialog, SpectralComponentsPanel,
    BufferSettingsDialog, OverlayTemplate, OverlayManager, CppTemplate  # Added CppTemplate
)

# Get logger but don't set level - it's controlled by AudioConfig
logger = logging.getLogger(__name__)

class SpectrumAnalyzerUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Add before other init code
        self.current_file = None
        self.has_unsaved_changes = False
        self.recent_files = []
        self.max_recent_files = 5
        self.export_settings = {}  # Store last used export settings
        
        # Set minimum window size that works for all modes
        self.setMinimumWidth(800)
        self.setMinimumHeight(942)
        
        # Apply platform-specific styles and window attributes
        if sys.platform == 'darwin':  # macOS specific styles
            self.setStyle(QStyleFactory.create('Fusion'))  # Use Fusion style on macOS
            
            # Enable window resizing on macOS
            self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint)
            self.setUnifiedTitleAndToolBarOnMac(True)
            
            # Set size policy to allow resizing
            sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setSizePolicy(sizePolicy)
            
            # Detect dark mode on macOS
            dark_mode = False
            try:
                import subprocess
                cmd = 'defaults read -g AppleInterfaceStyle'
                dark_mode = subprocess.check_output(cmd.split()).decode().strip() == 'Dark'
            except:
                pass
            
            # Colors for dark/light mode
            if dark_mode:
                bg_color = "#333333"
                text_color = "#FFFFFF"
                border_color = "#555555"
                slider_bg = "#444444"
                button_gradient_start = "#555555"
                button_gradient_end = "#444444"
                group_box_color = "#CCCCCC"
            else:
                bg_color = "#F5F5F5"  # Softer background
                text_color = "#2C2C2C"  # Softer black
                border_color = "#CCCCCC"  # Lighter borders
                slider_bg = "#E8E8E8"  # Softer slider background
                button_gradient_start = "#F0F0F0"  # Softer button gradient
                button_gradient_end = "#E0E0E0"
                group_box_color = "#2C2C2C"
            
            # Set stylesheet for better macOS appearance with dark mode support
            self.setStyleSheet(f"""
                QMainWindow, QWidget {{
                    background-color: {bg_color};
                    color: {text_color};
                }}
                QSlider::groove:horizontal {{
                    height: 4px;
                    background: {slider_bg};
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    margin: 1px 0;
                }}
                QSlider::handle:horizontal {{
                    width: 14px;
                    height: 14px;
                    margin: -5px 0;
                    border-radius: 7px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 {button_gradient_start}, 
                                              stop:1 {button_gradient_end});
                    border: 1px solid {border_color};
                }}
                QComboBox {{
                    padding: 1px 20px 1px 6px;  /* Increased padding */
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    min-height: 20px;  /* Slightly taller */
                    background-color: {bg_color};
                    color: {text_color};
                    font-size: 12px;  /* Increased for better readability */
                    selection-background-color: {button_gradient_end};
                }}
                QPushButton {{
                    padding: 2px 8px;  /* Increased padding */
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                              stop:0 {button_gradient_start}, 
                                              stop:1 {button_gradient_end});
                    min-height: 20px;  /* Slightly taller */
                    min-width: 50px;  /* Slightly wider */
                    font-size: 12px;  /* Increased for better readability */
                    color: {text_color};
                }}
                QSpinBox, QDoubleSpinBox {{
                    padding: 1px 4px;  /* Increased padding */
                    border: 1px solid {border_color};
                    border-radius: 2px;
                    min-height: 20px;  /* Slightly taller */
                    min-width: 60px;  /* Slightly wider */
                    background-color: {bg_color};
                    color: {text_color};
                    font-size: 12px;  /* Increased for better readability */
                }}
                QSpinBox::up-button, QDoubleSpinBox::up-button,
                QSpinBox::down-button, QDoubleSpinBox::down-button {{
                    width: 16px;  /* Slightly wider */
                    border-radius: 1px;
                    background-color: {bg_color};
                }}
                QGroupBox {{
                    margin-top: 12px;  /* Keep as is */
                    padding-top: 16px;  /* Keep as is */
                    padding: 6px;  /* Increased from 4px for better internal spacing */
                    margin-bottom: 2px;  /* Keep as is */
                    border: 1px solid {border_color};
                }}
                QGroupBox::title {{
                    margin-top: -14px;  /* Keep as is */
                    margin-bottom: 0px;
                    subcontrol-position: top left;
                    padding: 0 4px;  /* Keep as is */
                    font-weight: bold;
                    background-color: palette(window);
                }}
                QVBoxLayout {{
                    spacing: 2px;  /* Increased from 1px for better spacing between elements */
                }}
                QHBoxLayout {{
                    spacing: 2px;  /* Keep as is */
                }}
                QLabel {{
                    margin: 1px 0;  /* Added small vertical margin */
                    padding: 0;
                    min-height: 14px;
                }}
                QSlider {{
                    margin: 1px 0;  /* Keep as is */
                }}
                QPushButton {{
                    margin: 1px;  /* Added small margin */
                    padding: 1px 3px;  /* Keep as is */
                    min-height: 16px;  /* Keep as is */
                }}
                QComboBox {{
                    margin: 1px;  /* Added small margin */
                    padding: 1px 12px 1px 3px;  /* Keep as is */
                    min-height: 16px;  /* Keep as is */
                }}
                QSpinBox, QDoubleSpinBox {{
                    margin: 1px;  /* Added small margin */
                    padding: 1px 3px;  /* Keep as is */
                    min-height: 16px;  /* Keep as is */
                }}
                QCheckBox {{
                    margin: 1px 0;  /* Keep as is */
                    padding: 0;
                }}
            """)
        
        # Load recent files list from settings - Fix the loading code here
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        recent = settings.value('recent_files', [])
        # Convert to list if it's not already
        self.recent_files = recent if isinstance(recent, list) else []

        self.title = f"Noise Shaper v{VERSION}"
        
        # Continue with existing init code
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1200, 700)

        # Initialize configuration and processor
        self.config = AudioConfig()
        self.processor = AudioProcessor(self.config)
        
        # Initialize overlay manager first
        self.overlay_manager = OverlayManager()
        self.overlay_manager.overlay_changed.connect(self.update_overlays)
        self.overlay_manager.overlay_confirmed.connect(self.on_settings_changed)
        
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
        
        # Connect overlay manager signals for UI updates only
        self.overlay_manager.overlay_changed.connect(self.update_overlays)
        self.overlay_manager.overlay_confirmed.connect(self.on_settings_changed)

        # Add before calling init_ui()
        self.load_stored_settings()

        # Apply any saved export settings
        self.export_settings = QSettings('YourOrg', 'SpectrumAnalyzer').value('export_settings', {}, dict)

        # Load window geometry after UI setup
        self.load_window_geometry()

        # Add event filter to track window state changes
        self.installEventFilter(self)

    # Add these new methods
    def load_window_geometry(self):
        """Load and apply saved window geometry"""
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        
        geometry = settings.value('window_geometry')
        state = settings.value('window_state')
        
        if geometry:
            # Verify the geometry is visible on current screen setup
            available = QApplication.primaryScreen().availableGeometry()
            for screen in QApplication.screens():
                available = available.united(screen.availableGeometry())
            
            # Create rect from saved geometry
            saved = self.geometry()
            if isinstance(geometry, str):
                parts = list(map(int, geometry.split(',')))
                saved.setRect(parts[0], parts[1], parts[2], parts[3])
            else:
                saved = geometry
            
            # Check if saved geometry intersects any current screen
            if available.intersects(saved):
                self.setGeometry(saved)
            else:
                # Center on primary screen if saved geometry is invalid
                center = QApplication.primaryScreen().availableGeometry().center()
                saved.moveCenter(center)
                self.setGeometry(saved)
        
        # Restore window state (maximized/fullscreen)
        if state:
            self.restoreState(state)

    def load_stored_settings(self):
        """Load persistent settings from QSettings"""
        settings = QSettings('YourOrg', 'SpectrumAnalyzer')
        # Fix the loading code here too
        recent = settings.value('recent_files', [])
        self.recent_files = recent if isinstance(recent, list) else []
        self.export_settings = settings.value('export_settings', {}, dict)

    def eventFilter(self, obj, event) -> bool:
        """Track window state changes and mouse leave events"""
        if obj is self:
            if event.type() in [QEvent.Type.WindowStateChange]:
                # Only log window dimensions on actual state changes
                geometry = self.geometry()
                logger.debug(f"Window state changed - Width: {geometry.width()}, Height: {geometry.height()}")
                
                # Save whenever window state changes
                settings = QSettings('YourOrg', 'SpectrumAnalyzer')
                settings.setValue('window_geometry', self.geometry())
                settings.setValue('window_state', self.saveState())
            elif event.type() in [QEvent.Type.Move, QEvent.Type.Resize]:
                # Just save settings without logging for move/resize events
                settings = QSettings('YourOrg', 'SpectrumAnalyzer')
                settings.setValue('window_geometry', self.geometry())
                settings.setValue('window_state', self.saveState())
        elif obj is self.graph_widget:
            if event.type() == QEvent.Type.Leave:
                # Clear coordinates when mouse leaves the widget
                self.coord_label.setText("")
        return super().eventFilter(obj, event)

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
        self.setMenuBar(self.menubar)

        # Create status bar with coordinate label
        self.statusbar = StatusBar()
        self.coord_label = QLabel("")
        self.statusbar.addPermanentWidget(self.coord_label)  # Permanent widgets appear on the right
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
        # Create a fresh config with default values
        fresh_config = AudioConfig()
        
        # Copy default values to current config
        self.config.fft_size = fresh_config.fft_size
        self.config.window_type = fresh_config.window_type
        self.config.scale_type = fresh_config.scale_type
        self.config.averaging_count = fresh_config.averaging_count
        self.config.amp_whitenoise = fresh_config.amp_whitenoise
        self.config.amp_spectral = fresh_config.amp_spectral
        
        # Reset all panels to default values
        self.analyzer_panel.apply_settings({
            'fft_size': fresh_config.fft_size,
            'window_type': fresh_config.window_type,
            'scale_type': fresh_config.scale_type,
            'averaging_count': fresh_config.averaging_count,
            'decay_enabled': False,
            'decay_rate': 0.1,
            'trigger_enabled': False,
            'trigger_level': -45,
            'hold_time': 0.2,
            'trigger_reset_mode': "Hold Time",
            'trigger_edge_mode': "Rising"
        })
        
        # Reset source panel with default amplitudes
        self.source_panel.apply_settings({
            'amp_whitenoise': fresh_config.amp_whitenoise,
            'amp_spectral': fresh_config.amp_spectral,
            'cpp_template': None,  # This will trigger using default template in SourcePanel
            'carousel_template': None  # This will trigger using default template in SourcePanel
        })
        
        # Clear filters from both UI and processor
        self.processor.filters.clear()
        self.filter_panel.apply_settings({})
        
        # Clear spectral components from both UI and processor
        if isinstance(self.processor.source, NoiseSource):
            self.processor.source._parabolas.clear()
            self.processor.source.generator.parabolas.clear()
        # Clear UI components
        for i in range(len(self.parabola_panel.parabolas)-1, -1, -1):
            self.parabola_panel.remove_parabola(i)
        
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
            
        # Update amplitude control based on current mode
        source_type = self.source_panel.source_type.currentText()
        if source_type == "White Noise":
            self.source_panel.amplitude_control.setValue(fresh_config.amp_whitenoise)
        elif source_type == "Spectral Synthesis":
            self.source_panel.amplitude_control.setValue(fresh_config.amp_spectral)

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
            
            # Apply UI panel settings first (except filters)
            self.analyzer_panel.apply_settings(settings.get('analyzer', {}))
            self.source_panel.apply_settings(settings.get('source', {}))
            
            # Handle filters after source is configured
            if 'filters' in settings:
                # Clear existing filters once - they're shared between processor and source
                self.processor.filters.clear()
                logger.debug("load_settings_file - cleared filters")
                
                # First apply to filter panel UI
                self.filter_panel.apply_settings(settings.get('filters', {}))
                
                # Then create and add filters to processor
                for filter_settings in settings['filters'].get('filters', []):
                    # Create a copy to avoid modifying original
                    params = filter_settings.copy()
                    # Ensure type exists before calling add_filter
                    if 'type' in params:
                        self.add_filter(params)  # add_filter will handle type removal

            # Handle spectral components
            if isinstance(self.processor.source, NoiseSource):
                # Clear existing components from both UI and processor
                self.processor.source._parabolas.clear()
                self.processor.source.generator.parabolas.clear()
            # Clear UI components
            for i in range(len(self.parabola_panel.parabolas)-1, -1, -1):
                self.parabola_panel.remove_parabola(i)
            
            # Add new components if present in settings
            if 'spectral_components' in settings:
                # Add new components - this will update both UI and processor through signals
                for component in settings['spectral_components']:
                    self.parabola_panel.add_parabola(component)
            
            # Load overlay templates with interpolation
            self.overlay_manager.templates.clear()
            for t in settings.get('overlays', []):
                template = OverlayTemplate(
                    name=t['name'],
                    color=t['color'],
                    points=t['points'],
                    interpolation=t.get('interpolation', 'linear'),  # Default to linear
                    symbol=t.get('symbol', 'o')  # Default to circle if not specified
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

    @staticmethod
    def _numpy_to_list(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: SpectrumAnalyzerUI._numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [SpectrumAnalyzerUI._numpy_to_list(item) for item in obj]
        return obj

    def save_settings_to_file(self, filename):
        try:
            # Get overlay settings with interpolation
            overlay_settings = [
                {
                    'name': t.name,
                    'color': t.color,
                    'points': self._numpy_to_list(t.points),  # Convert points
                    'enabled': t.enabled,
                    'offset': t.offset,
                    'interpolation': t.interpolation,
                    'symbol': t.symbol  # Add symbol to saved settings
                }
                for t in self.overlay_manager.templates
            ]
            
            # Get the latest export settings from source panel
            if hasattr(self.source_panel, 'export_settings'):
                self.export_settings = self._numpy_to_list(self.source_panel.export_settings)  # Convert settings
            
            settings = {
                'version': VERSION,  # Use VERSION from config
                'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analyzer': self._numpy_to_list(self.analyzer_panel.get_current_settings()),
                'source': self._numpy_to_list(self.source_panel.get_current_settings()),
                'filters': self._numpy_to_list(self.filter_panel.get_current_settings()),
                'overlays': self._numpy_to_list(overlay_settings),
                'export': self._numpy_to_list(self.export_settings),
                'cpp_template': self._numpy_to_list(self.source_panel.cpp_template),
                'spectral_components': [widget.get_parameters() for widget in self.parabola_panel.parabolas]
            }
            
            # Convert all settings to ensure no numpy arrays remain
            settings = self._numpy_to_list(settings)
            
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
            # Create action with full path as text
            action = self.recent_menu.addAction(filename)
            action.setData(filename)
            action.triggered.connect(lambda checked, f=filename: self.load_recent_file(f))

    def load_recent_file(self, filename):
        """Load a recent file with unsaved changes check"""
        if self.check_unsaved_changes():
            self.load_settings_file(filename)

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
            
        # Save window state is now handled by eventFilter
        # Just save other settings
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
            logger.error(f"Shutdown error: {e}")
        finally:
            event.accept()

    def setup_graph(self):
        """Initializes the PyQtGraph plot widget"""
        # Detect dark mode on macOS
        dark_mode = False
        if sys.platform == 'darwin':
            try:
                import subprocess
                cmd = 'defaults read -g AppleInterfaceStyle'
                dark_mode = subprocess.check_output(cmd.split()).decode().strip() == 'Dark'
            except:
                pass

        # Set colors based on dark mode
        if dark_mode:
            self.graph_widget.setBackground('#333333')
            grid_color = (80, 80, 80)
            text_color = '#FFFFFF'
            curve_color = (100, 100, 255)
            fill_color = (80, 80, 200, 50)
        else:
            self.graph_widget.setBackground('w')
            grid_color = (200, 200, 200)
            text_color = '#000000'
            curve_color = (80, 40, 200)
            fill_color = (100, 50, 255, 50)

        # Configure grid and axes
        plot_item = self.graph_widget.getPlotItem()
        
        # Style axes
        for axis in ['left', 'bottom']:
            ax = plot_item.getAxis(axis)
            ax.setTextPen(text_color)
            ax.setPen(pg.mkPen(color=text_color, width=1))
        
        # Set grid properties
        plot_item.getAxis('left').setGrid(True)
        plot_item.getAxis('bottom').setGrid(True)
        plot_item.showGrid(x=True, y=True, alpha=0.5)
        
        # Set grid pens
        plot_item.getAxis('left').setStyle(tickTextOffset=5)
        plot_item.getAxis('bottom').setStyle(tickTextOffset=5)
        plot_item.getAxis('left').setPen(pg.mkPen(color=grid_color, width=1, style=Qt.PenStyle.DotLine))
        plot_item.getAxis('bottom').setPen(pg.mkPen(color=grid_color, width=1, style=Qt.PenStyle.DotLine))
        
        # Set axis labels with proper color
        self.graph_widget.setLabel('left', 'Magnitude (dB)', color=text_color)
        self.graph_widget.setLabel('bottom', 'Frequency (Hz)', color=text_color)
        
        # Create filled curve style
        pen = pg.mkPen(color=curve_color, width=2)
        brush = pg.mkBrush(color=fill_color)
        
        # Create the plot curve with fill
        self.plot_curve = pg.PlotDataItem(
            fillLevel=-90,
            brush=brush,
            pen=pen
        )
        self.graph_widget.addItem(self.plot_curve)

        # Add trigger threshold line (initially hidden)
        trigger_color = 'r' if not dark_mode else '#FF5555'
        self.trigger_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(color=trigger_color, width=1, style=Qt.PenStyle.DashLine),
            movable=False,
            label='Trigger {value:.1f} dB',
            labelOpts={'position': 0.95, 'color': trigger_color}
        )
        self.graph_widget.addItem(self.trigger_line)
        self.trigger_line.hide()  # Hide initially

        # Connect mouse move event
        self.graph_widget.scene().sigMouseMoved.connect(self.mouse_moved)
        
        # Install event filter for mouse leave events
        self.graph_widget.installEventFilter(self)
        
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

            # For log mode, ensure we handle very low frequencies properly
            if self.analyzer_panel.scale_type.currentText().lower() == 'logarithmic':
                # Instead of filtering, replace zeros/negatives with small positive value
                min_freq = 0.1  # 0.1 Hz minimum
                freq = np.maximum(freq, min_freq)
            
            # Update the plot - pyqtgraph handles the scaling
            self.plot_curve.setData(freq, spec_db)
            
        except Exception as e:
            logger.error(f"Plot error details: {str(e)}")
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
            
            # Update trigger line if it exists
            if hasattr(self, 'trigger_line'):
                trigger_enabled = self.analyzer_panel.trigger_enabled.isChecked()
                trigger_level = self.analyzer_panel.trigger_level.value()
                self.trigger_line.setValue(trigger_level)
                self.trigger_line.setVisible(trigger_enabled and 
                    self.source_panel.get_source_type() == "Test Mode")
            
            # Force graph update
            self.update_graph_scale()
            
            # Clear previous spectrum data to avoid averaging issues across scale changes
            if hasattr(self, '_prev_spec'):
                delattr(self, '_prev_spec')
            
        except Exception as e:
            logger.error(f"Settings update error: {str(e)}")
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
                # Start from 0.1 Hz instead of 20 Hz
                self.graph_widget.setXRange(np.log10(0.1), np.log10(20000))
                
                # Log scale ticks - use log values for positions
                major_ticks = [
                    (np.log10(freq), label) for freq, label in [
                        (0.1, '0.1'), (1, '1'), (10, '10'), (100, '100'), 
                        (1000, '1k'), (10000, '10k'), (20000, '20k')
                    ]
                ]
                
                minor_ticks = [
                    (np.log10(freq), str(freq)) for freq in [
                    0.2, 0.5, 2, 5, 20, 50,
                    200, 500, 2000, 5000
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
            logger.error(f"Scale update error: {e}")

        # Update overlays after scale change
        self.update_overlays()

    def create_control_panel(self) -> QWidget:
        control_panel = QWidget()
        layout = QVBoxLayout(control_panel)
        
        # Make layout extremely compact vertically
        layout.setSpacing(2)  # Reduced from 6
        layout.setContentsMargins(2, 2, 2, 2)  # Reduced from 4,4,4,4
        
        # Source settings panel
        self.source_panel = SourcePanel(self.config)
        self.source_panel.source_changed.connect(self.handle_source_change)
        layout.addWidget(self.source_panel)

        # Analyzer settings panel (always visible)
        self.analyzer_panel = AnalyzerPanel(self.config)
        self.analyzer_panel.settings_changed.connect(self.update_analyzer_settings)
        layout.addWidget(self.analyzer_panel)

        # Create a widget to hold filter and parabola panels
        filter_container = QWidget()
        filter_layout = QVBoxLayout(filter_container)
        filter_layout.setSpacing(0)  # Reduced from 1
        filter_layout.setContentsMargins(0, 0, 0, 0)  # No margins

        # Filter panel (only for white noise)
        self.filter_panel = FilterPanel(self.config, self.processor)
        self.filter_panel.filter_parameters.connect(self.add_filter)
        self.filter_panel.filter_removed.connect(self.remove_filter)
        self.filter_panel.filter_updated.connect(self.update_filter)
        filter_layout.addWidget(self.filter_panel)
        
        # Parabola panel (only for parabolic noise)
        self.parabola_panel = SpectralComponentsPanel(self.processor)
        self.parabola_panel.parabola_added.connect(self.add_parabola)
        self.parabola_panel.parabola_removed.connect(self.remove_parabola)
        self.parabola_panel.parabola_updated.connect(self.update_parabola)
        filter_layout.addWidget(self.parabola_panel)
        self.parabola_panel.hide()

        # Add filter container to main layout with stretch
        layout.addWidget(filter_container, 1)  # Add stretch=1

        # Add overlay manager after other panels
        layout.addWidget(self.overlay_manager)
        
        # Set size policy for control panel - adjusted for macOS
        if sys.platform == 'darwin':
            control_panel.setFixedWidth(300)  # Slightly narrower
        else:
            control_panel.setFixedWidth(300)
        control_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        # Show/hide appropriate panels based on source type
        self.source_panel.source_type.currentTextChanged.connect(self.handle_mode_change)
        
        return control_panel

    def handle_mode_change(self, mode: str):
        """Show/hide panels based on selected mode"""
        self.filter_panel.setVisible(mode == "White Noise")
        self.parabola_panel.setVisible(mode == "Spectral Synthesis")
        
        # Update status indicators based on mode
        is_test_mode = mode == "Test Mode"
        self.source_panel.monitoring_panel.overflow_indicator.setVisible(is_test_mode)
        self.source_panel.monitoring_panel.underflow_indicator.setVisible(True)  # Always show UF
        
        # Show/hide test mode specific controls and trigger line
        self.analyzer_panel.show_test_mode_controls(is_test_mode)
        if hasattr(self, 'trigger_line'):
            self.trigger_line.setVisible(is_test_mode and self.analyzer_panel.trigger_enabled.isChecked())
        
        # Stop playback if running and restart with new source type
        if self.source_panel.is_playing:
            self.source_panel.toggle_playback()  # Stop
            self.source_panel.toggle_playback()  # Start with new source type

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
            logger.error(f"Error changing audio source: {e}")
            self.source_panel.is_playing = False
            self.source_panel.play_button.setText("Play")
            self.source_panel.play_button.setChecked(False)
            # Re-enable device selection on error
            self.source_panel.monitoring_panel.device_combo.setEnabled(True)
            self.source_panel.input_device_panel.device_combo.setEnabled(True)
            # Show error to user
            self.show_error("Source Error", f"Error changing audio source: {str(e)}")

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
            elif filter_type == 'plateau':  # Add this case
                filter_ = PlateauFilter(self.config, **filter_params)
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            # Only add to processor - it will handle adding to source
            self.processor.add_filter(filter_)
                
        except Exception as e:
            self.show_error("Filter Error", f"Error adding filter: {str(e)}")

    def remove_filter(self, index: int):
        """Removes the filter at the specified index"""
        try:
            # Only remove from processor - it will handle removing from source
            self.processor.remove_filter(index)
                
        except Exception as e:
            self.show_error("Filter Error", f"Error removing filter: {str(e)}")

    def update_filter(self, index: int, params: dict):
        """Update filter parameters"""
        try:
            # Only update processor filter - it will handle updating source
            self.processor.update_filter(index, params)
                
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

    def export_noise(self, settings: dict):
        """Export noise to WAV and/or C++ file"""
        try:
            logger.debug("\nDEBUG: export_noise called with settings:")
            for key, value in settings.items():
                logger.debug(f"- {key}: {value}")

            # Store original states
            original_monitoring = self.config.monitoring_enabled
            was_playing = self.source_panel.is_playing

            # Stop playback if it's running
            if was_playing:
                self.source_panel.toggle_playback()

            # Disable monitoring during export
            self.config.monitoring_enabled = False

            # Create noise source with appropriate type
            source_type = settings.get('source_type', 'White Noise').lower()
            noise_type = 'white' if source_type == 'white noise' else 'spectral'
            noise_source = NoiseSource(self.config, noise_type=noise_type)

            try:
                # Set RNG type
                noise_source.set_rng_type(settings.get('rng_type', 'standard_normal'))

                # Add all active filters from the filter panel
                for filter_widget in self.filter_panel.filters:
                    # Create new filter based on widget parameters
                    filter_params = filter_widget.get_parameters()
                    filter_type = filter_params.pop('type')
                    
                    if filter_type == 'bandpass':
                        filter_obj = BandpassFilter(self.config, **filter_params)
                    elif filter_type == 'lowpass':
                        filter_obj = LowpassFilter(self.config, **filter_params)
                    elif filter_type == 'highpass':
                        filter_obj = HighpassFilter(self.config, **filter_params)
                    elif filter_type == 'notch':
                        filter_obj = NotchFilter(self.config, **filter_params)
                    elif filter_type == 'gaussian':
                        filter_obj = GaussianFilter(self.config, **filter_params)
                    elif filter_type == 'parabolic':
                        filter_obj = ParabolicFilter(self.config, **filter_params)
                    elif filter_type == 'plateau':
                        filter_obj = PlateauFilter(self.config, **filter_params)
                        
                    noise_source.add_filter(filter_obj)

                # Copy parabolas if in spectral mode
                if noise_type == 'spectral':
                    for widget in self.parabola_panel.parabolas:
                        noise_source.add_parabola(widget.get_parameters())

                # Then add any additional filters from export settings
                for filter_settings in settings.get('filters', []):
                    filter_type = filter_settings.get('type', '').lower()
                    if filter_type == 'bandpass':
                        filter_obj = BandpassFilter(
                            self.config,
                            filter_settings.get('lowcut', 20),
                            filter_settings.get('highcut', 20000),
                            filter_settings.get('order', 4),
                            filter_settings.get('amplitude', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'lowpass':
                        filter_obj = LowpassFilter(
                            self.config,
                            filter_settings.get('cutoff', 20000),
                            filter_settings.get('order', 4),
                            filter_settings.get('amplitude', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'highpass':
                        filter_obj = HighpassFilter(
                            self.config,
                            filter_settings.get('cutoff', 20),
                            filter_settings.get('order', 4),
                            filter_settings.get('amplitude', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'notch':
                        filter_obj = NotchFilter(
                            self.config,
                            filter_settings.get('frequency', 1000),
                            filter_settings.get('q', 30.0),
                            filter_settings.get('amplitude', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'gaussian':
                        filter_obj = GaussianFilter(
                            self.config,
                            filter_settings.get('center_freq', 1000),
                            filter_settings.get('width', 100),
                            filter_settings.get('amplitude', 1.0),
                            filter_settings.get('skew', 0.0),
                            filter_settings.get('kurtosis', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'parabolic':
                        filter_obj = ParabolicFilter(
                            self.config,
                            filter_settings.get('center_freq', 1000),
                            filter_settings.get('width', 100),
                            filter_settings.get('amplitude', 1.0),
                            filter_settings.get('skew', 0.0),
                            filter_settings.get('flatness', 1.0)
                        )
                        noise_source.add_filter(filter_obj)
                    elif filter_type == 'plateau':
                        filter_obj = PlateauFilter(
                            self.config,
                            filter_settings.get('center_freq', 1000),
                            filter_settings.get('width', 100),
                            filter_settings.get('flat_width', 50),
                            filter_settings.get('amplitude', 1.0)
                        )
                        noise_source.add_filter(filter_obj)

                # Generate audio data
                if settings.get('carousel_enabled', False):
                    # Export sequence directly from noise source
                    silence, samples = noise_source.export_sequence(settings)
                    
                    # Export individual samples if requested and WAV export is enabled
                    if settings.get('export_individual', False) and settings.get('export_wav', True):
                        folder_path = settings.get('folder_path', '.')
                        for i, sample in enumerate(samples):
                            wav_path = os.path.join(folder_path, f'noise_{i+1}.wav')
                            sf.write(wav_path, sample, settings.get('sample_rate', 44100))
                    
                    # Export combined sequence if requested
                    if settings.get('export_combined', True):
                        # Create combined sequence
                        sequence = []
                        for sample in samples:
                            sequence.extend([sample, silence])
                        audio_data = np.concatenate(sequence)
                        
                        # Export WAV if enabled
                        if settings.get('export_wav', True):
                            wav_path = os.path.join(settings.get('folder_path', '.'), 
                                                  settings.get('wav_filename', 'noise.wav'))
                            sf.write(wav_path, audio_data, settings.get('sample_rate', 44100))
                        
                        # Export C++ if enabled
                        if settings.get('export_cpp', True):
                            cpp_path = os.path.join(settings.get('folder_path', '.'),
                                                  settings.get('cpp_filename', 'noise.h'))
                            cpp_code = AudioExporter.generate_cpp_code(audio_data, settings)
                            with open(cpp_path, 'w') as f:
                                f.write(cpp_code)
                else:
                    # Generate single noise sample
                    audio_data = noise_source.export_signal(
                        duration=settings.get('duration', 10.0),
                        sample_rate=settings.get('sample_rate', 44100),
                        amplitude=settings.get('amplitude', 0.5),
                        enable_fade=settings.get('enable_fade', True),
                        fade_in_duration=settings.get('fade_in_duration', 0.001),
                        fade_out_duration=settings.get('fade_out_duration', 0.001),
                        fade_in_power=settings.get('fade_in_power', 2.0),
                        fade_out_power=settings.get('fade_out_power', 2.0),
                        enable_normalization=settings.get('enable_normalization', True),
                        normalize_value=settings.get('normalize_value', 1.0),
                        fade_before_norm=settings.get('fade_before_norm', False),
                        rng_type=settings.get('rng_type', 'standard_normal'),
                        use_random_seed=settings.get('use_random_seed', True),
                        seed=settings.get('seed', None)
                    )
                    
                    # Export WAV if enabled
                    if settings.get('export_wav', True):
                        wav_path = os.path.join(settings.get('folder_path', '.'), 
                                              settings.get('wav_filename', 'noise.wav'))
                        sf.write(wav_path, audio_data, settings.get('sample_rate', 44100))
                    
                    # Export C++ if enabled
                    if settings.get('export_cpp', True):
                        cpp_path = os.path.join(settings.get('folder_path', '.'),
                                              settings.get('cpp_filename', 'noise.h'))
                        cpp_code = AudioExporter.generate_cpp_code(audio_data, settings)
                        with open(cpp_path, 'w') as f:
                            f.write(cpp_code)

            finally:
                # Restore original states
                self.config.monitoring_enabled = original_monitoring
                # Clean up noise source
                noise_source.close()
                # Restore playback if it was active
                if was_playing:
                    self.source_panel.toggle_playback()
                        
        except Exception as e:
            logger.error(f"Error exporting noise: {e}")
            raise

    def update_overlays(self):
        """Update all overlay curves with interpolation support"""
        # Remove existing overlay curves and points
        for item in self.graph_widget.items():
            if (isinstance(item, pg.PlotDataItem) or isinstance(item, pg.ScatterPlotItem)) and item != self.plot_curve:
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
                # Generate interpolated points if needed
                if len(freqs) > 1:  # Only need to check if we have enough points
                    # Create dense x points for interpolation
                    x_dense = np.linspace(min(freqs), max(freqs), 500)
                    
                    if template.interpolation == 'linear':
                        y_dense = np.interp(x_dense, freqs, levels)
                    elif template.interpolation == 'cubic':
                        from scipy.interpolate import CubicSpline
                        cs = CubicSpline(freqs, levels)
                        y_dense = cs(x_dense)
                    elif template.interpolation == 'akima':
                        from scipy.interpolate import Akima1DInterpolator
                        ak = Akima1DInterpolator(freqs, levels)
                        y_dense = ak(x_dense)
                    
                    # Create interpolated curve
                    curve = pg.PlotDataItem(
                        x_dense, y_dense,
                        pen=pg.mkPen(color=template.color, width=2)
                    )
                    # Add points on top with custom symbol
                    points = pg.ScatterPlotItem(
                        freqs, levels,
                        symbol=template.symbol,
                        size=10,  # Slightly larger to make symbols more visible
                        pen=pg.mkPen(color=template.color, width=1.5),
                        brush=template.color if template.symbol in ['x', '+'] else None  # Fill for x and +, hollow for others
                    )
                    self.graph_widget.addItem(curve)
                    self.graph_widget.addItem(points)
                else:
                    # Just points with lines
                    curve = pg.PlotDataItem(
                        freqs, levels,
                        pen=pg.mkPen(color=template.color, width=2),
                        symbol=template.symbol,
                        symbolSize=10,  # Slightly larger to make symbols more visible
                        symbolPen=pg.mkPen(color=template.color, width=1.5),
                        symbolBrush=template.color if template.symbol in ['x', '+'] else None  # Fill for x and +, hollow for others
                    )
                    self.graph_widget.addItem(curve)
                
            except Exception as e:
                logger.error(f"Error creating overlay curve: {e}")
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

    def mouse_moved(self, evt):
        """Handle mouse move event to show coordinates in status bar"""
        plot_rect = self.graph_widget.plotItem.vb.sceneBoundingRect()
        if plot_rect.contains(evt):
            mouse_point = self.graph_widget.plotItem.vb.mapSceneToView(evt)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Handle logarithmic scale
            if self.analyzer_panel.scale_type.currentText().lower() == 'logarithmic':
                # Convert from log scale back to linear frequency
                x = 10 ** x
                
            # Ensure frequency is not negative or zero
            x = max(0.1, x)  # Minimum 0.1 Hz
            
            # Format frequency with appropriate units
            if x >= 1000:
                freq_str = f"{x/1000:.1f} kHz"
            else:
                freq_str = f"{x:.0f} Hz"
                
            # Update coordinate label
            self.coord_label.setText(f"Frequency: {freq_str}, Magnitude: {y:.1f} dB")
        else:
            # Clear coordinates when mouse leaves plot area
            self.coord_label.setText("")

def main():
    app = QApplication(sys.argv)
    window = SpectrumAnalyzerUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()