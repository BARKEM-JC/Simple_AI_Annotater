import sys
import os
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QSlider, QFileDialog, QTextEdit,
                            QListWidget, QSplitter, QFrame, QLineEdit, QComboBox,
                            QMessageBox, QSpinBox, QCheckBox, QGroupBox, QGridLayout,
                            QListWidgetItem, QInputDialog, QProgressBar, QStatusBar,
                            QTabWidget, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QKeySequence
from PyQt5.QtWidgets import QShortcut
import time
from pathlib import Path
from video_player import VideoPlayer
from region_selector import RegionSelector
from label_manager import LabelManager
from clip_manager import ClipManager
from data_exporter import DataExporter
from auto_annotation_manager import AutoAnnotationManager
from synthetic_data_widget import SyntheticDataWidget

class AILabelerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.images_path = None  # For image directory loading
        self.source_type = None  # 'video' or 'images'
        self.source_name = None  # Name to use for export prefixes
        self.current_frame = None
        self.frame_count = 0
        self.current_frame_index = 0
        self.fps = 30
        
        # Create output directory
        self.output_dir = Path("labeled_data")
        self.output_dir.mkdir(exist_ok=True)

        # Persistent annotations state
        self.persistent_annotations = {}  # Map of annotation IDs to their data
        self.deleted_annotations = set()  # Set of deleted annotation IDs
        
        # Auto-annotation state
        self.auto_annotation_enabled = True
        self.auto_annotation_timer = QTimer()
        self.auto_annotation_timer.setSingleShot(True)
        self.auto_annotation_timer.timeout.connect(self.run_auto_annotation)
        self.auto_annotation_delay = 100
        
        # Frame change tracking for persistence
        self._last_frame_index = None
        
        self.init_ui()
        self.setup_shortcuts()
        self.connect_signals()
        
    def reset_persistent_annotation_state(self):
        """Reset all persistent annotation tracking state"""
        self.persistent_annotations = {}
        self.deleted_annotations = set()
        self._last_frame_index = None
        
    def init_ui(self):
        self.setWindowTitle("AI Dataset Labeler")
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create scroll area to make the whole page scrollable
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create scrollable content widget
        scrollable_widget = QWidget()
        scrollable_layout = QHBoxLayout(scrollable_widget)
        
        # Create splitter for main sections
        splitter = QSplitter(Qt.Horizontal)
        scrollable_layout.addWidget(splitter)
        
        # Left panel - Video and controls
        left_panel = self.create_video_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Labels and regions
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([1000, 400])
        
        # Add scrollable widget to scroll area
        scroll_area.setWidget(scrollable_widget)
        main_layout.addWidget(scroll_area)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a video to start labeling")
        
    def create_video_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # File controls
        file_layout = QHBoxLayout()
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        file_layout.addWidget(self.load_video_btn)
        
        self.load_images_btn = QPushButton("Load Images")
        self.load_images_btn.clicked.connect(self.load_images)
        file_layout.addWidget(self.load_images_btn)
        
        self.video_info_label = QLabel("No video/images loaded")
        file_layout.addWidget(self.video_info_label)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Video display
        self.video_player = VideoPlayer()
        self.video_player.frame_changed.connect(self.on_frame_changed)
        self.video_player.region_added.connect(self.on_region_added)
        self.video_player.region_selected.connect(self.on_region_selected)
        layout.addWidget(self.video_player, 1)
        
        # Video controls
        controls_layout = QVBoxLayout()
        
        # Playback controls
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        playback_layout.addWidget(self.play_btn)
        
        self.prev_frame_btn = QPushButton("Previous Frame")
        self.prev_frame_btn.clicked.connect(self.previous_frame)
        self.prev_frame_btn.setEnabled(False)
        playback_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton("Next Frame")
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.next_frame_btn.setEnabled(False)
        playback_layout.addWidget(self.next_frame_btn)
        
        self.skip_btn = QPushButton("Skip 10 Frames")
        self.skip_btn.clicked.connect(self.skip_frames)
        self.skip_btn.setEnabled(False)
        playback_layout.addWidget(self.skip_btn)
        
        playback_layout.addStretch()
        controls_layout.addLayout(playback_layout)
        
        # Frame slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        self.frame_slider.setEnabled(False)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.valueChanged.connect(self.seek_frame_spinbox)
        self.frame_spinbox.setEnabled(False)
        slider_layout.addWidget(self.frame_spinbox)
        
        controls_layout.addLayout(slider_layout)
        layout.addLayout(controls_layout)
        
        return panel
    
    def create_control_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Auto-annotation controls at the top
        auto_group = QGroupBox("Auto-Annotation Settings")
        auto_layout = QVBoxLayout(auto_group)
        
        self.auto_annotation_checkbox = QCheckBox("Auto-annotate on frame change (0.5s delay)")
        self.auto_annotation_checkbox.setChecked(self.auto_annotation_enabled)
        self.auto_annotation_checkbox.stateChanged.connect(self.on_auto_annotation_toggled)
        auto_layout.addWidget(self.auto_annotation_checkbox)
        
        layout.addWidget(auto_group)
        
        # Create tab widget for different labeling modes
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Region labeling tab
        region_tab = QWidget()
        region_layout = QVBoxLayout(region_tab)
        
        # Label manager
        self.label_manager = LabelManager()
        region_layout.addWidget(self.label_manager)
        
        # Frame labeling
        frame_group = QGroupBox("Frame Labels")
        frame_layout = QVBoxLayout(frame_group)
        
        self.frame_label_input = QLineEdit()
        self.frame_label_input.setPlaceholderText("Enter frame label...")
        frame_layout.addWidget(self.frame_label_input)
        
        self.add_frame_label_btn = QPushButton("Add Frame Label")
        self.add_frame_label_btn.clicked.connect(self.add_frame_label)
        frame_layout.addWidget(self.add_frame_label_btn)
        
        self.frame_labels_list = QListWidget()
        frame_layout.addWidget(self.frame_labels_list)
        
        region_layout.addWidget(frame_group)
        
        # Region controls
        region_group = QGroupBox("Region Controls")
        region_controls_layout = QVBoxLayout(region_group)
        
        self.delete_region_btn = QPushButton("Delete Selected Region")
        self.delete_region_btn.clicked.connect(self.delete_selected_region)
        self.delete_region_btn.setEnabled(False)
        region_controls_layout.addWidget(self.delete_region_btn)
        
        self.clear_regions_btn = QPushButton("Clear All Regions")
        self.clear_regions_btn.clicked.connect(self.clear_all_regions)
        region_controls_layout.addWidget(self.clear_regions_btn)
        
        self.reset_persistence_btn = QPushButton("Reset Persistence Tracking")
        self.reset_persistence_btn.clicked.connect(self.reset_persistent_annotation_state)
        self.reset_persistence_btn.setToolTip("Reset deleted annotation tracking to allow previously deleted annotations to persist again")
        region_controls_layout.addWidget(self.reset_persistence_btn)
        
        region_layout.addWidget(region_group)
        region_layout.addStretch()
        
        self.tab_widget.addTab(region_tab, "Region Labeling")
        
        # Clip labeling tab (MoViNet)
        clip_tab = QWidget()
        clip_layout = QVBoxLayout(clip_tab)
        
        self.clip_manager = ClipManager()
        self.clip_manager.clip_created.connect(self.on_clip_created)
        self.clip_manager.clip_selected.connect(self.on_clip_selected)
        self.clip_manager.clip_deleted.connect(self.on_clip_deleted)
        self.clip_manager.clip_preview_requested.connect(self.on_clip_preview_requested)
        clip_layout.addWidget(self.clip_manager)
        
        self.tab_widget.addTab(clip_tab, "Clip Labeling (MoViNet)")
        
        # Auto Annotation tab
        auto_annotation_tab = QWidget()
        auto_annotation_layout = QVBoxLayout(auto_annotation_tab)
        
        self.auto_annotation_manager = AutoAnnotationManager()
        self.auto_annotation_manager.annotation_accepted.connect(self.on_auto_annotation_accepted)
        self.auto_annotation_manager.template_created.connect(self.on_template_created)
        auto_annotation_layout.addWidget(self.auto_annotation_manager)
        
        self.tab_widget.addTab(auto_annotation_tab, "Auto Annotation")
        
        # Synthetic Data tab
        synthetic_tab = QWidget()
        synthetic_layout = QVBoxLayout(synthetic_tab)
        
        self.synthetic_data_widget = SyntheticDataWidget()
        self.synthetic_data_widget.synthetic_image_ready.connect(self.load_synthetic_image)
        synthetic_layout.addWidget(self.synthetic_data_widget)
        
        self.tab_widget.addTab(synthetic_tab, "Synthetic Data Generator")
        
        # Export controls
        export_group = QGroupBox("Export Data")
        export_layout = QVBoxLayout(export_group)
        
        # Export format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLOv8", "COCO", "Pascal VOC", "Custom JSON", "MoViNet"])
        format_layout.addWidget(self.export_format)
        export_layout.addLayout(format_layout)
        
        # Data Augmentation Section
        augmentation_group = QGroupBox("Data Augmentation")
        augmentation_layout = QVBoxLayout(augmentation_group)
        
        # Augmentation mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.augmentation_mode = QComboBox()
        self.augmentation_mode.addItems([
            "No Augmentation",
            "Copy & create new data for each modification",
            "Modify frames directly, keep some originals", 
            "Copy & apply multiple random modifiers, keep originals",
            "Apply multiple random modifiers, no copies"
        ])
        mode_layout.addWidget(self.augmentation_mode)
        augmentation_layout.addLayout(mode_layout)
        
        # Augmentation options checkboxes
        augmentation_options_layout = QGridLayout()
        
        self.aug_pixelation = QCheckBox("Pixelation")
        self.aug_blur = QCheckBox("Blur")
        self.aug_rotation = QCheckBox("Rotation (±10°)")
        self.aug_mirror = QCheckBox("Mirror/Flip")
        self.aug_brightness = QCheckBox("Brightness")
        self.aug_contrast = QCheckBox("Contrast")
        self.aug_saturation = QCheckBox("Saturation")
        self.aug_noise = QCheckBox("Noise")
        
        augmentation_options_layout.addWidget(self.aug_pixelation, 0, 0)
        augmentation_options_layout.addWidget(self.aug_blur, 0, 1)
        augmentation_options_layout.addWidget(self.aug_rotation, 0, 2)
        augmentation_options_layout.addWidget(self.aug_mirror, 0, 3)
        augmentation_options_layout.addWidget(self.aug_brightness, 1, 0)
        augmentation_options_layout.addWidget(self.aug_contrast, 1, 1)
        augmentation_options_layout.addWidget(self.aug_saturation, 1, 2)
        augmentation_options_layout.addWidget(self.aug_noise, 1, 3)
        
        augmentation_layout.addLayout(augmentation_options_layout)
        export_layout.addWidget(augmentation_group)
        
        # Export button and progress
        self.export_btn = QPushButton("Export Dataset")
        self.export_btn.clicked.connect(self.export_dataset)
        export_layout.addWidget(self.export_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        export_layout.addWidget(self.progress_bar)
        
        layout.addWidget(export_group)
        
        return panel
    
    def connect_signals(self):
        """Connect signals between components"""
        self.label_manager.label_changed.connect(self.on_label_changed)
        self.label_manager.quick_opencv_annotation_requested.connect(self.on_quick_opencv_annotation_requested)
        self.label_manager.add_template_requested.connect(self.on_add_template_requested)
        
        # Auto-annotation manager signals
        self.auto_annotation_manager.annotation_accepted.connect(self.on_auto_annotation_accepted)
        self.auto_annotation_manager.template_created.connect(self.on_template_created)
    
    def setup_shortcuts(self):
        # Video control shortcuts
        QShortcut(QKeySequence("Space"), self, self.toggle_playback)
        QShortcut(QKeySequence("Left"), self, self.previous_frame)
        QShortcut(QKeySequence("Right"), self, self.next_frame)
        QShortcut(QKeySequence("Shift+Right"), self, self.skip_frames)
        QShortcut(QKeySequence("Shift+Left"), self, self.skip_frames_back)
        
        # Region shortcuts
        QShortcut(QKeySequence("Delete"), self, self.delete_selected_region)
        QShortcut(QKeySequence("Ctrl+A"), self, self.select_all_regions)
        QShortcut(QKeySequence("Escape"), self, self.clear_selection)
        
        # File shortcuts
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_video)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current_frame)
        QShortcut(QKeySequence("Ctrl+E"), self, self.export_dataset)
    
    def load_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Load Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;All Files (*)"
        )
        
        if video_path:
            if self.video_player.load_video(video_path):
                self.video_path = video_path
                self.images_path = None
                self.source_type = 'video'
                self.source_name = Path(video_path).stem
                
                self.frame_count = self.video_player.frame_count
                self.fps = self.video_player.fps
                
                # Update UI
                self.video_info_label.setText(f"Video: {Path(video_path).name} | Frames: {self.frame_count} | FPS: {self.fps:.2f}")
                
                # Enable controls
                self.play_btn.setEnabled(True)
                self.prev_frame_btn.setEnabled(True)
                self.next_frame_btn.setEnabled(True)
                self.skip_btn.setEnabled(True)
                
                # Setup frame slider
                self.frame_slider.setEnabled(True)
                self.frame_slider.setRange(0, self.frame_count - 1)
                self.frame_slider.setValue(0)
                
                # Setup frame spinbox
                self.frame_spinbox.setEnabled(True)
                self.frame_spinbox.setRange(0, self.frame_count - 1)
                self.frame_spinbox.setValue(0)
                
                # Reset tracking and persistent annotation state
                self.reset_persistent_annotation_state()
                
                self.status_bar.showMessage(f"Video loaded: {self.frame_count} frames")
            else:
                QMessageBox.critical(self, "Error", "Failed to load video file")

    def load_images(self):
        """Load a directory of images as a sequence"""
        folder_dialog = QFileDialog()
        images_dir = folder_dialog.getExistingDirectory(self, "Select Image Directory")
        
        if images_dir:
            # Get all image files from directory
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(images_dir).glob(f'*{ext}'))
                image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
            
            # Sort files by name
            image_files = sorted(image_files, key=lambda x: x.name.lower())
            
            if not image_files:
                QMessageBox.warning(self, "Warning", "No image files found in selected directory")
                return
            
            if self.video_player.load_images(image_files):
                self.images_path = images_dir
                self.video_path = None
                self.source_type = 'images'
                self.source_name = Path(images_dir).name
                
                self.frame_count = len(image_files)
                self.fps = 30  # Default FPS for image sequences
                
                # Update UI
                self.video_info_label.setText(f"Images: {len(image_files)} files from {Path(images_dir).name}")
                
                # Enable controls
                self.play_btn.setEnabled(True)
                self.prev_frame_btn.setEnabled(True)
                self.next_frame_btn.setEnabled(True)
                self.skip_btn.setEnabled(True)
                
                # Setup frame slider
                self.frame_slider.setEnabled(True)
                self.frame_slider.setRange(0, self.frame_count - 1)
                self.frame_slider.setValue(0)
                
                # Setup frame spinbox
                self.frame_spinbox.setEnabled(True)
                self.frame_spinbox.setRange(0, self.frame_count - 1)
                self.frame_spinbox.setValue(0)
                
                # Reset tracking and persistent annotation state
                self.reset_persistent_annotation_state()
                
                self.status_bar.showMessage(f"Images loaded: {self.frame_count} files")
            else:
                QMessageBox.critical(self, "Error", "Failed to load image sequence")
    
    def toggle_playback(self):
        if self.video_player.is_playing:
            self.video_player.pause()
            self.play_btn.setText("Play")
        else:
            self.video_player.play()
            self.play_btn.setText("Pause")
    
    def previous_frame(self):
        self.video_player.previous_frame()
    
    def next_frame(self):
        self.video_player.next_frame()
    
    def skip_frames(self):
        self.video_player.skip_frames(10)
    
    def skip_frames_back(self):
        self.video_player.skip_frames(-10)
    
    def seek_frame(self, frame_index):
        if frame_index != self.current_frame_index:
            self.video_player.seek_frame(frame_index)
            self.frame_spinbox.blockSignals(True)
            self.frame_spinbox.setValue(frame_index)
            self.frame_spinbox.blockSignals(False)
    
    def seek_frame_spinbox(self, frame_index):
        if frame_index != self.current_frame_index:
            self.video_player.seek_frame(frame_index)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(frame_index)
            self.frame_slider.blockSignals(False)
    
    def on_frame_changed(self, frame_index, frame):
        # Save current frame annotations before changing
        if self.current_frame_index is not None and self.current_frame_index != frame_index:
            self.save_frame_annotations()
        
        self.current_frame_index = frame_index
        self.current_frame = frame
        
        # Update slider and spinbox
        self.frame_slider.blockSignals(True)
        self.frame_spinbox.blockSignals(True)
        self.frame_slider.setValue(frame_index)
        self.frame_spinbox.setValue(frame_index)
        self.frame_slider.blockSignals(False)
        self.frame_spinbox.blockSignals(False)
        
        # Load frame annotations
        self.load_frame_annotations(frame_index)
        
        # Apply persistent annotations if moving to next consecutive frame
        self.apply_persistent_annotations(frame_index)
        
        # Update clip manager with the current frame
        self.clip_manager.set_current_frame(frame_index)
        
        # Update auto-annotation manager with current frame
        self.auto_annotation_manager.set_current_frame(frame, frame_index)
        
        # Update synthetic data widget with current frame and regions
        self.update_synthetic_data_widget()
        
        # Trigger auto-annotation if enabled
        if self.auto_annotation_enabled:
            self.auto_annotation_timer.stop()  # Cancel any pending auto-annotation
            self.auto_annotation_timer.start(self.auto_annotation_delay)
        
        self.status_bar.showMessage(f"Frame: {frame_index + 1}/{self.frame_count}")

    def apply_persistent_annotations(self, frame_index):
        """Apply persistent annotations from the previous frame when moving to next frame"""
        # Initialize _last_frame_index if it doesn't exist
        if not hasattr(self, '_last_frame_index') or self._last_frame_index is None:
            self._last_frame_index = frame_index
            return

        # Only apply persistence when moving forward to the next consecutive frame
        if self._last_frame_index == frame_index - 1:
            source_frame_index = self._last_frame_index
            source_annotation_file = self.output_dir / f"frame_{source_frame_index:06d}.json"

            if not source_annotation_file.exists():
                self._last_frame_index = frame_index
                return

            try:
                with open(source_annotation_file, 'r') as f:
                    source_data = json.load(f)
            except Exception as e:
                print(f"Error reading source annotation file: {e}")
                self._last_frame_index = frame_index
                return

            # Get regions and labels to persist
            source_regions = source_data.get('regions', [])
            source_frame_labels = source_data.get('frame_labels', [])

            if not source_regions and not source_frame_labels:
                self._last_frame_index = frame_index
                return

            # Get current frame's annotations from the UI elements
            current_regions = self.video_player.get_all_regions_data()
            current_frame_labels = [self.frame_labels_list.item(i).text() for i in range(self.frame_labels_list.count())]

            persistent_regions_to_add = []
            for i, region_data in enumerate(source_regions):
                # Preserve original_id if it exists, otherwise use the parent's id
                original_id = region_data.get('original_id', region_data.get('id'))
                if original_id in self.deleted_annotations:
                    continue

                # Check for overlap with existing regions on the new frame
                if self.find_overlapping_region(region_data, current_regions) is not None:
                    continue

                # Create new region with updated ID for this frame
                new_region_data = region_data.copy()
                new_region_data['original_id'] = original_id
                new_region_data['id'] = f"persistent_{frame_index}_{i}"
                new_region_data['persistent'] = True
                persistent_regions_to_add.append(new_region_data)

            # Persist frame labels (combine without duplicates)
            persistent_frame_labels_to_add = [label for label in source_frame_labels if label not in current_frame_labels]

            # If there's anything to add, update and save
            if persistent_regions_to_add or persistent_frame_labels_to_add:
                # Add to video player and UI immediately
                for region in persistent_regions_to_add:
                    self.video_player.add_region_from_data(region)
                for label in persistent_frame_labels_to_add:
                    self.frame_labels_list.addItem(QListWidgetItem(label))

                # Save the merged annotations
                self.save_frame_annotations()

                self.status_bar.showMessage(f"Persisted {len(persistent_regions_to_add)} annotations and {len(persistent_frame_labels_to_add)} labels.")

        # Update last frame index for next consecutive check
        self._last_frame_index = frame_index

    def on_region_added(self, region_data):
        # Add region to label manager
        self.label_manager.add_region(region_data)
        
        # Store in persistent annotations
        region_id = region_data.get('id')
        if region_id:
            self.persistent_annotations[region_id] = region_data.copy()
            # Remove from deleted annotations if it was previously deleted
            original_id = region_data.get('original_id', region_id)
            self.deleted_annotations.discard(original_id)
        
        # Automatically assign current label if one is selected
        current_label = self.label_manager.get_current_label()
        if current_label:
            if region_id:
                # Apply the label to the region
                self.video_player.set_region_label(region_id, current_label)
                # Update persistent annotations
                self.persistent_annotations[region_id]['label'] = current_label
                # Save frame automatically when label is applied
                self.save_frame_annotations()

    def on_region_selected(self, region_id):
        # Update UI based on selected region
        self.delete_region_btn.setEnabled(region_id is not None and region_id != "")
        if region_id and region_id != "":
            self.label_manager.select_region(region_id)
            
            # Get region data and notify auto-annotation manager
            region_data = self.video_player.get_region_data(region_id)
            if region_data:
                label = region_data.get('label', '')
                region_rect = {
                    'x': region_data.get('x', 0),
                    'y': region_data.get('y', 0), 
                    'width': region_data.get('width', 0),
                    'height': region_data.get('height', 0)
                }
                self.auto_annotation_manager.set_selected_region(region_rect, label)
                
                # Update synthetic data widget with selected region
                self.update_synthetic_data_widget()
    
    def on_label_changed(self, region_id, label):
        """Handle label change from label manager"""
        self.video_player.set_region_label(region_id, label)
        # Save frame automatically when label changes
        self.save_frame_annotations()
    
    def add_frame_label(self):
        label_text = self.frame_label_input.text().strip()
        if label_text:
            # Add to current frame
            item = QListWidgetItem(label_text)
            self.frame_labels_list.addItem(item)
            self.frame_label_input.clear()
            
            # Save frame annotation
            self.save_frame_annotations()
    
    def delete_selected_region(self):
        selected_region_id = self.video_player.get_selected_region()
        if selected_region_id:
            region_data = self.video_player.get_region_data(selected_region_id)

            # Add the original ID to deleted annotations to prevent persistence
            if region_data:
                original_id = region_data.get('original_id', selected_region_id)
                self.deleted_annotations.add(original_id)

            # Remove from persistent annotations dict
            self.persistent_annotations.pop(selected_region_id, None)

            self.video_player.delete_region(selected_region_id)
            self.label_manager.remove_region(selected_region_id)
            self.delete_region_btn.setEnabled(False)
            # Save frame automatically when region is deleted
            self.save_frame_annotations()
    
    def clear_all_regions(self):
        self.video_player.clear_regions()
        self.label_manager.clear_regions()
        self.delete_region_btn.setEnabled(False)
        # Save frame automatically when regions are cleared
        self.save_frame_annotations()
    
    def select_all_regions(self):
        self.video_player.select_all_regions()
    
    def clear_selection(self):
        self.video_player.clear_selection()
        self.delete_region_btn.setEnabled(False)
    
    def save_current_frame(self):
        if self.current_frame is not None:
            self.save_frame_annotations()
            self.status_bar.showMessage("Frame annotations saved", 1000)
    
    def load_frame_annotations(self, frame_index):
        # Load annotations for current frame
        annotation_file = self.output_dir / f"frame_{frame_index:06d}.json"
        
        # Clear current annotations but keep deletion tracking for persistence
        self.frame_labels_list.clear()
        self.video_player.clear_regions()
        # DO NOT clear deleted_annotations - keep them to prevent persistence of deleted regions
        
        if annotation_file.exists():
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                # Load frame labels
                for label in data.get('frame_labels', []):
                    item = QListWidgetItem(label)
                    self.frame_labels_list.addItem(item)
                
                # Load regions
                for region_data in data.get('regions', []):
                    self.video_player.add_region_from_data(region_data)
                    
            except Exception as e:
                print(f"Error loading annotations: {e}")
    
    def save_frame_annotations(self):
        if self.current_frame_index is None:
            return
        
        # Collect frame labels
        frame_labels = []
        for i in range(self.frame_labels_list.count()):
            frame_labels.append(self.frame_labels_list.item(i).text())
        
        # Collect region data
        regions = self.video_player.get_all_regions_data()
        
        # Create annotation data
        annotation_data = {
            'frame_index': self.current_frame_index,
            'video_path': self.video_path,
            'frame_labels': frame_labels,
            'regions': regions,
            'timestamp': time.time()
        }
        
        # Save to file
        annotation_file = self.output_dir / f"frame_{self.current_frame_index:06d}.json"
        try:
            with open(annotation_file, 'w') as f:
                json.dump(annotation_data, f, indent=2)
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def export_dataset(self):
        if not self.video_path and not self.images_path:
            QMessageBox.warning(self, "Warning", "No video or images loaded")
            return
        
        format_type = self.export_format.currentText()
        
        # Collect augmentation options
        augmentation_options = {
            'mode': self.augmentation_mode.currentText(),
            'enabled_augmentations': []
        }
        
        if self.aug_pixelation.isChecked():
            augmentation_options['enabled_augmentations'].append('pixelation')
        if self.aug_blur.isChecked():
            augmentation_options['enabled_augmentations'].append('blur')
        if self.aug_rotation.isChecked():
            augmentation_options['enabled_augmentations'].append('rotation')
        if self.aug_mirror.isChecked():
            augmentation_options['enabled_augmentations'].append('mirror')
        if self.aug_brightness.isChecked():
            augmentation_options['enabled_augmentations'].append('brightness')
        if self.aug_contrast.isChecked():
            augmentation_options['enabled_augmentations'].append('contrast')
        if self.aug_saturation.isChecked():
            augmentation_options['enabled_augmentations'].append('saturation')
        if self.aug_noise.isChecked():
            augmentation_options['enabled_augmentations'].append('noise')
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            exporter = DataExporter(self.output_dir, format_type, self.source_name)
            
            # Pass source path and type to exporter
            source_path = self.video_path if self.source_type == 'video' else self.images_path
            success = exporter.export_dataset(source_path, self.source_type, augmentation_options, self.progress_bar)
            
            if success:
                if augmentation_options['mode'] != "No Augmentation" and augmentation_options['enabled_augmentations']:
                    QMessageBox.information(self, "Success", 
                        f"Dataset exported successfully in {format_type} format with data augmentation")
                else:
                    QMessageBox.information(self, "Success", 
                        f"Dataset exported successfully in {format_type} format")
                self.status_bar.showMessage(f"Dataset exported in {format_type} format", 3000)
            else:
                QMessageBox.warning(self, "Warning", "Export completed with some errors")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
    
    def on_clip_created(self, clip_data):
        """Handle clip creation"""
        pass  # Clip is automatically saved by ClipManager
    
    def on_clip_selected(self, clip_id):
        """Handle clip selection - seek to clip start frame"""
        clip = self.clip_manager.get_clip_by_id(clip_id)
        if clip:
            self.seek_frame(clip.start_frame)
    
    def on_clip_deleted(self, clip_id):
        """Handle clip deletion"""
        pass  # Clip is automatically removed by ClipManager
    
    def on_clip_preview_requested(self, start_frame, end_frame):
        """Handle clip preview request"""
        self.seek_frame(start_frame)
    
    def on_auto_annotation_accepted(self, annotation_data, frame_index):
        """Handle accepted auto annotation"""
        if frame_index == self.current_frame_index:
            # Add the annotation as a new region
            region = annotation_data.get('region', {})
            label = annotation_data.get('label', 'auto')
            
            # Create region data structure
            region_data = {
                'id': f"auto_{len(self.video_player.regions)}_{frame_index}",
                'x': region.get('x', 0),
                'y': region.get('y', 0),
                'width': region.get('width', 0),
                'height': region.get('height', 0),
                'label': label,
                'confidence': annotation_data.get('confidence', 0),
                'auto_generated': True,
                'type': annotation_data.get('type', 'unknown')
            }
            
            # Add to video player
            self.video_player.add_region_from_data(region_data)
            
            # Save annotations
            self.save_frame_annotations()
            
            self.status_bar.showMessage(f"Auto annotation '{label}' added successfully")

    def on_template_created(self, template_id, label):
        """Handle template creation"""
        self.status_bar.showMessage(f"Template '{label}' created with ID: {template_id[:8]}...")
    
    def on_quick_opencv_annotation_requested(self):
        """Handle quick OpenCV annotation request from label manager"""
        success = self.auto_annotation_manager.run_quick_opencv_annotation()
        if not success:
            self.status_bar.showMessage("Quick OpenCV annotation failed: No templates available or no frame loaded")
        else:
            self.status_bar.showMessage("Running quick OpenCV annotation...")
    
    def closeEvent(self, event):
        # Save current frame before closing
        if self.current_frame_index is not None:
            self.save_frame_annotations()
        
        # Stop video player
        if hasattr(self, 'video_player'):
            self.video_player.stop()
        
        event.accept()

    def update_synthetic_data_widget(self):
        """Update synthetic data widget with current frame and region data"""
        if hasattr(self, 'synthetic_data_widget') and self.current_frame is not None:
            # Get current regions
            current_regions = self.video_player.get_all_regions_data()
            
            # Get selected region
            selected_region_id = self.video_player.get_selected_region()
            selected_region = None
            if selected_region_id:
                selected_region = self.video_player.get_region_data(selected_region_id)
            
            # Update the synthetic data widget
            self.synthetic_data_widget.set_source_data(
                self.current_frame,
                current_regions,
                selected_region
            )
    
    def load_synthetic_image(self, image_path):
        """Load a synthetic image for annotation"""
        try:
            # Load the synthetic image
            synthetic_image = cv2.imread(image_path)
            if synthetic_image is None:
                QMessageBox.warning(self, "Error", f"Failed to load synthetic image: {image_path}")
                return
            
            # Set up for image sequence mode with single synthetic image
            self.images_path = [image_path]
            self.video_path = None
            self.source_type = 'images'
            self.source_name = Path(image_path).stem
            
            # Load the synthetic image in video player
            if self.video_player.load_images([image_path]):
                self.frame_count = 1
                self.fps = 30  # Default FPS for synthetic images
                
                # Update UI
                self.video_info_label.setText(f"Synthetic Image: {self.source_name}")
                self.frame_slider.setRange(0, 0)
                self.frame_spinbox.setRange(0, 0)
                
                # Disable video controls (single image)
                self.play_btn.setEnabled(False)
                self.prev_frame_btn.setEnabled(False)
                self.next_frame_btn.setEnabled(False)
                self.skip_btn.setEnabled(False)
                self.frame_slider.setEnabled(False)
                self.frame_spinbox.setEnabled(False)
                
                # Switch to region labeling tab
                self.tab_widget.setCurrentIndex(0)  # Region Labeling tab
                
                # Load existing annotations if they exist
                annotation_file = Path(image_path).with_suffix('.json')
                if annotation_file.exists():
                    try:
                        with open(annotation_file, 'r') as f:
                            data = json.load(f)
                        
                        # Load regions from synthetic data annotation
                        for region_data in data.get('regions', []):
                            self.video_player.add_region_from_data(region_data)
                    except Exception as e:
                        print(f"Error loading synthetic image annotations: {e}")
                
                self.status_bar.showMessage(f"Loaded synthetic image for annotation: {self.source_name}")
                
            else:
                QMessageBox.warning(self, "Error", "Failed to load synthetic image in video player")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading synthetic image:\n{str(e)}")
            
            
    def find_overlapping_region(self, track_region, existing_regions):
        """Find if there's an existing region that overlaps significantly with the track region"""
        track_x = track_region.get('x', 0)
        track_y = track_region.get('y', 0)
        track_w = track_region.get('width', 0)
        track_h = track_region.get('height', 0)
        
        for i, existing_region in enumerate(existing_regions):
            existing_x = existing_region.get('x', 0)
            existing_y = existing_region.get('y', 0)
            existing_w = existing_region.get('width', 0)
            existing_h = existing_region.get('height', 0)
            
            # Calculate overlap
            overlap_x = max(0, min(track_x + track_w, existing_x + existing_w) - max(track_x, existing_x))
            overlap_y = max(0, min(track_y + track_h, existing_y + existing_h) - max(track_y, existing_y))
            overlap_area = overlap_x * overlap_y
            
            # Calculate areas
            track_area = track_w * track_h
            existing_area = existing_w * existing_h
            
            # Check if overlap is significant (more than 50% of either region)
            if track_area > 0 and existing_area > 0:
                overlap_ratio_track = overlap_area / track_area
                overlap_ratio_existing = overlap_area / existing_area
                
                if overlap_ratio_track > 0.5 or overlap_ratio_existing > 0.5:
                    return i
        
        return None

    def on_auto_annotation_toggled(self, state):
        """Handle auto-annotation checkbox toggled"""
        self.auto_annotation_enabled = state == Qt.Checked
        if not self.auto_annotation_enabled:
            self.auto_annotation_timer.stop()

    def run_auto_annotation(self):
        """Run auto-annotation for the current frame"""
        if not self.auto_annotation_enabled or self.current_frame is None:
            return
            
        # Don't run if no templates exist
        if not self.auto_annotation_manager.opencv_annotator.templates:
            return
            
        # Get existing annotations to avoid duplicates
        existing_annotations = []
        for region in self.video_player.regions:
            region_data = self.video_player.get_region_data(region.id)
            if region_data:
                existing_annotations.append(region_data)
        
        # Run OpenCV auto-annotation with existing annotations filter
        try:
            matches = self.auto_annotation_manager.opencv_annotator.find_matches(
                self.current_frame, 
                existing_annotations=existing_annotations
            )
            
            # Only apply matches if auto-apply is enabled and we have high-confidence matches
            high_confidence_matches = [match for match in matches if match.confidence > 0.8]
            
            # Apply the high-confidence matches as annotations
            if high_confidence_matches:
                for match in high_confidence_matches:
                    region_data = {
                        'x': match.region['x'],
                        'y': match.region['y'],
                        'width': match.region['width'],
                        'height': match.region['height'],
                        'label': match.label,
                        'id': f"auto_{match.template_id}_{self.current_frame_index}_{len(existing_annotations)}"
                    }
                    self.video_player.add_region_from_data(region_data)
                
                self.save_frame_annotations()
                self.status_bar.showMessage(f"Auto-annotation: {len(high_confidence_matches)} high-confidence matches found")
            elif matches:
                # For lower confidence matches, just show a subtle status message
                self.status_bar.showMessage(f"Auto-annotation: {len(matches)} potential matches found (low confidence)")
        except Exception as e:
            print(f"Auto-annotation error: {e}")
            # Don't show error messages to user for auto-annotation failures

    def is_auto_annotation_enabled(self):
        """Return whether auto-annotation is enabled"""
        return self.auto_annotation_enabled

    def get_auto_annotation_delay(self):
        """Return the current auto-annotation delay"""
        return self.auto_annotation_delay

    def set_auto_annotation_delay(self, delay):
        """Set the auto-annotation delay"""
        self.auto_annotation_delay = delay

    def on_add_template_requested(self, region_id, label):
        """Handle request to add current region as template"""
        if not self.current_frame is None and region_id:
            region_data = self.video_player.get_region_data(region_id)
            if region_data:
                region_rect = {
                    'x': region_data.get('x', 0),
                    'y': region_data.get('y', 0),
                    'width': region_data.get('width', 0),
                    'height': region_data.get('height', 0)
                }
                
                # Add template to OpenCV auto-annotator
                template_id = self.auto_annotation_manager.opencv_annotator.add_template_from_region(
                    self.current_frame, region_rect, label
                )
                
                if template_id:
                    # Save templates
                    self.auto_annotation_manager.opencv_annotator.save_templates()
                    # Update template list in auto-annotation manager
                    self.auto_annotation_manager.update_template_list()
                    # Show success message only in status bar
                    self.status_bar.showMessage(f"Template '{label}' created successfully")
                else:
                    self.status_bar.showMessage("Failed to create template") 