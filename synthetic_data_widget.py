"""
Synthetic Data Widget
GUI interface for generating synthetic training data
"""

import os
import sys
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QTextEdit, QSpinBox, QProgressBar,
                            QListWidget, QListWidgetItem, QGroupBox, QGridLayout,
                            QMessageBox, QComboBox, QCheckBox, QFileDialog,
                            QSplitter, QFrame, QScrollArea, QDialog, QDialogButtonBox,
                            QLineEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPalette, QFont

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from synthetic_dataset_generator import SyntheticDataGenerator

class SyntheticGenerationThread(QThread):
    """Thread for generating synthetic data without blocking UI"""
    progress_updated = pyqtSignal(int)
    generation_completed = pyqtSignal(list)
    generation_failed = pyqtSignal(str)
    
    def __init__(self, generator, source_image, description, num_images, region, original_regions):
        super().__init__()
        self.generator = generator
        self.source_image = source_image
        self.description = description
        self.num_images = num_images
        self.region = region
        self.original_regions = original_regions
    
    def run(self):
        try:
            def progress_callback(progress):
                self.progress_updated.emit(progress)
            
            generated_data = self.generator.create_synthetic_dataset(
                self.source_image,
                self.description,
                self.num_images,
                self.region,
                self.original_regions,
                progress_callback
            )
            
            self.generation_completed.emit(generated_data)
            
        except Exception as e:
            self.generation_failed.emit(str(e))

class SyntheticImageViewer(QDialog):
    """Dialog for viewing generated synthetic images"""
    
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.image_data = image_data
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Synthetic Image Viewer")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout(self)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setScaledContents(True)
        
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # Image info
        info_group = QGroupBox("Image Information")
        info_layout = QGridLayout(info_group)
        
        info_layout.addWidget(QLabel("Image ID:"), 0, 0)
        info_layout.addWidget(QLabel(self.image_data.get('image_id', 'N/A')), 0, 1)
        
        info_layout.addWidget(QLabel("Created:"), 1, 0)
        info_layout.addWidget(QLabel(self.image_data.get('created_at', 'N/A')), 1, 1)
        
        info_layout.addWidget(QLabel("User Description:"), 2, 0)
        desc_label = QLabel(self.image_data.get('user_description', 'N/A'))
        desc_label.setWordWrap(True)
        info_layout.addWidget(desc_label, 2, 1)
        
        layout.addWidget(info_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.annotate_btn = QPushButton("Open for Annotation")
        self.annotate_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.annotate_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        # Load and display image
        self.load_image()
    
    def load_image(self):
        """Load and display the synthetic image"""
        try:
            image_path = self.image_data.get('image_path')
            if image_path and os.path.exists(image_path):
                # Load with OpenCV
                cv_img = cv2.imread(image_path)
                if cv_img is not None:
                    # Convert BGR to RGB
                    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_img.shape
                    bytes_per_line = ch * w
                    
                    # Create QImage
                    qt_image = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    # Convert to QPixmap and display
                    pixmap = QPixmap.fromImage(qt_image)
                    self.image_label.setPixmap(pixmap)
                    
        except Exception as e:
            print(f"Error loading synthetic image: {e}")
            self.image_label.setText("Error loading image")

class SyntheticDataWidget(QWidget):
    """Main widget for synthetic data generation"""
    
    synthetic_image_ready = pyqtSignal(str)  # Emit path to load synthetic image
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.generator = SyntheticDataGenerator()
        self.current_source_image = None
        self.current_regions = []
        self.selected_region = None
        self.generation_thread = None
        
        self.init_ui()
        self.update_api_status()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # API Status
        status_group = QGroupBox("Synthetic Data Generation Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Checking OpenAI API...")
        status_layout.addWidget(self.status_label)
        
        self.setup_btn = QPushButton("Setup OpenAI API Key")
        self.setup_btn.clicked.connect(self.setup_api_key)
        status_layout.addWidget(self.setup_btn)
        
        layout.addWidget(status_group)
        
        # Generation Settings
        settings_group = QGroupBox("Generation Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Source selection
        settings_layout.addWidget(QLabel("Source:"), 0, 0)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Whole Frame", "Selected Region"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        settings_layout.addWidget(self.source_combo, 0, 1)
        
        # Number of images
        settings_layout.addWidget(QLabel("Number of Images:"), 1, 0)
        self.num_images_spin = QSpinBox()
        self.num_images_spin.setRange(1, 20)
        self.num_images_spin.setValue(5)
        settings_layout.addWidget(self.num_images_spin, 1, 1)
        
        # Description input
        settings_layout.addWidget(QLabel("Description:"), 2, 0)
        self.description_text = QTextEdit()
        self.description_text.setPlaceholderText("Describe the variations you want in the synthetic images...\n\nExample: Generate similar images with different lighting conditions, backgrounds, and angles. Include variations in weather and time of day.")
        self.description_text.setMaximumHeight(100)
        settings_layout.addWidget(self.description_text, 2, 1)
        
        # Quick preset buttons
        preset_layout = QHBoxLayout()
        
        presets = [
            ("Different Lighting", "Same scene with different lighting conditions - bright, dim, natural, artificial"),
            ("Weather Variations", "Same scene in different weather - sunny, cloudy, rainy, foggy"),
            ("Time Variations", "Same scene at different times - morning, noon, evening, night"),
            ("Angle Changes", "Same scene from slightly different camera angles and perspectives")
        ]
        
        for preset_name, preset_desc in presets:
            btn = QPushButton(preset_name)
            btn.clicked.connect(lambda checked, desc=preset_desc: self.description_text.setText(desc))
            preset_layout.addWidget(btn)
        
        settings_layout.addLayout(preset_layout, 3, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        # Generation controls
        controls_group = QGroupBox("Generation Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.generate_btn = QPushButton("Generate Synthetic Dataset")
        self.generate_btn.clicked.connect(self.generate_synthetic_data)
        self.generate_btn.setEnabled(False)
        controls_layout.addWidget(self.generate_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.status_text = QLabel("Ready to generate synthetic data")
        controls_layout.addWidget(self.status_text)
        
        layout.addWidget(controls_group)
        
        # Generated images list
        images_group = QGroupBox("Generated Synthetic Images")
        images_layout = QVBoxLayout(images_group)
        
        # Refresh button
        refresh_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("Refresh List")
        self.refresh_btn.clicked.connect(self.refresh_generated_images)
        refresh_layout.addWidget(self.refresh_btn)
        
        self.clear_all_btn = QPushButton("Clear All Synthetic Data")
        self.clear_all_btn.clicked.connect(self.clear_all_synthetic_data)
        refresh_layout.addWidget(self.clear_all_btn)
        
        refresh_layout.addStretch()
        images_layout.addLayout(refresh_layout)
        
        # Images list
        self.images_list = QListWidget()
        self.images_list.itemDoubleClicked.connect(self.view_synthetic_image)
        images_layout.addWidget(self.images_list)
        
        # Image action buttons
        image_actions_layout = QHBoxLayout()
        
        self.view_btn = QPushButton("View Image")
        self.view_btn.clicked.connect(self.view_selected_image)
        self.view_btn.setEnabled(False)
        image_actions_layout.addWidget(self.view_btn)
        
        self.load_btn = QPushButton("Load for Annotation")
        self.load_btn.clicked.connect(self.load_selected_image)
        self.load_btn.setEnabled(False)
        image_actions_layout.addWidget(self.load_btn)
        
        self.delete_btn = QPushButton("Delete Image")
        self.delete_btn.clicked.connect(self.delete_selected_image)
        self.delete_btn.setEnabled(False)
        image_actions_layout.addWidget(self.delete_btn)
        
        images_layout.addLayout(image_actions_layout)
        
        layout.addWidget(images_group)
        
        # Connect list selection
        self.images_list.itemSelectionChanged.connect(self.on_image_selection_changed)
        
        # Initial refresh
        QTimer.singleShot(100, self.refresh_generated_images)
    
    def update_api_status(self):
        """Update API status display"""
        if self.generator.is_available():
            self.status_label.setText("✅ OpenAI API ready for synthetic data generation")
            self.status_label.setStyleSheet("color: green;")
            self.generate_btn.setEnabled(True)
            self.setup_btn.setVisible(False)
        else:
            self.status_label.setText("❌ OpenAI API not configured")
            self.status_label.setStyleSheet("color: red;")
            self.generate_btn.setEnabled(False)
            self.setup_btn.setVisible(True)
    
    def setup_api_key(self):
        """Dialog to setup OpenAI API key"""
        current_key = os.getenv('OPENAI_API_KEY', '')
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Setup OpenAI API Key")
        dialog.setModal(True)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("Enter your OpenAI API Key:"))
        
        key_input = QLineEdit()
        key_input.setText(current_key)
        key_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(key_input)
        
        info_label = QLabel("You can get your API key from: https://platform.openai.com/api-keys")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info_label)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            api_key = key_input.text().strip()
            if api_key:
                # Set environment variable
                os.environ['OPENAI_API_KEY'] = api_key
                
                # Reinitialize generator
                self.generator = SyntheticDataGenerator()
                self.update_api_status()
                
                QMessageBox.information(self, "Success", "API key configured successfully!")
            else:
                QMessageBox.warning(self, "Error", "Please enter a valid API key")
    
    def set_source_data(self, frame, regions=None, selected_region=None):
        """Set the source frame and regions for synthetic generation"""
        self.current_source_image = frame
        self.current_regions = regions or []
        self.selected_region = selected_region
        
        # Update UI based on available data
        if self.selected_region:
            self.source_combo.setCurrentText("Selected Region")
        else:
            self.source_combo.setCurrentText("Whole Frame")
    
    def on_source_changed(self, source_type):
        """Handle source type change"""
        if source_type == "Selected Region" and not self.selected_region:
            QMessageBox.information(self, "No Region Selected", 
                                  "Please select a region in the main window first, then return to this tab.")
    
    def generate_synthetic_data(self):
        """Start synthetic data generation"""
        if not self.generator.is_available():
            QMessageBox.warning(self, "API Not Available", 
                              "OpenAI API is not configured. Please setup your API key first.")
            return
        
        if self.current_source_image is None:
            QMessageBox.warning(self, "No Source Image", 
                              "Please load a video/image and navigate to a frame first.")
            return
        
        description = self.description_text.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "No Description", 
                              "Please provide a description for the synthetic data generation.")
            return
        
        # Determine region to use
        region = None
        if self.source_combo.currentText() == "Selected Region":
            if self.selected_region:
                region = self.selected_region
            else:
                QMessageBox.warning(self, "No Region Selected", 
                                  "Selected Region mode requires a region to be selected.")
                return
        
        # Start generation in thread
        num_images = self.num_images_spin.value()
        
        self.generation_thread = SyntheticGenerationThread(
            self.generator,
            self.current_source_image,
            description,
            num_images,
            region,
            self.current_regions
        )
        
        self.generation_thread.progress_updated.connect(self.on_progress_updated)
        self.generation_thread.generation_completed.connect(self.on_generation_completed)
        self.generation_thread.generation_failed.connect(self.on_generation_failed)
        
        # Update UI for generation
        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_text.setText("Generating synthetic data...")
        
        self.generation_thread.start()
    
    def on_progress_updated(self, progress):
        """Handle progress update"""
        self.progress_bar.setValue(progress)
    
    def on_generation_completed(self, generated_data):
        """Handle generation completion"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        
        count = len(generated_data)
        self.status_text.setText(f"✅ Successfully generated {count} synthetic images")
        
        # Refresh the images list
        self.refresh_generated_images()
        
        QMessageBox.information(self, "Generation Complete", 
                              f"Successfully generated {count} synthetic images!\n\n"
                              f"Images are saved to: {self.generator.output_dir}")
    
    def on_generation_failed(self, error_msg):
        """Handle generation failure"""
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_text.setText("❌ Generation failed")
        
        QMessageBox.critical(self, "Generation Failed", 
                           f"Failed to generate synthetic data:\n\n{error_msg}")
    
    def refresh_generated_images(self):
        """Refresh the list of generated images"""
        self.images_list.clear()
        
        try:
            generated_images = self.generator.get_generated_images()
            
            for image_data in generated_images:
                item_text = f"{image_data.get('image_id', 'Unknown')} - {image_data.get('created_at', 'Unknown time')}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, image_data)
                self.images_list.addItem(item)
            
            self.status_text.setText(f"Found {len(generated_images)} synthetic images")
            
        except Exception as e:
            print(f"Error refreshing images: {e}")
            self.status_text.setText("Error loading synthetic images")
    
    def on_image_selection_changed(self):
        """Handle image selection change"""
        has_selection = len(self.images_list.selectedItems()) > 0
        self.view_btn.setEnabled(has_selection)
        self.load_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
    
    def view_synthetic_image(self, item):
        """View synthetic image (double-click handler)"""
        self.view_selected_image()
    
    def view_selected_image(self):
        """View the selected synthetic image"""
        selected_items = self.images_list.selectedItems()
        if not selected_items:
            return
        
        image_data = selected_items[0].data(Qt.UserRole)
        
        viewer = SyntheticImageViewer(image_data, self)
        if viewer.exec_() == QDialog.Accepted:
            # User wants to annotate this image
            self.load_selected_image()
    
    def load_selected_image(self):
        """Load selected synthetic image for annotation"""
        selected_items = self.images_list.selectedItems()
        if not selected_items:
            return
        
        image_data = selected_items[0].data(Qt.UserRole)
        image_path = image_data.get('image_path')
        
        if image_path and os.path.exists(image_path):
            self.synthetic_image_ready.emit(image_path)
        else:
            QMessageBox.warning(self, "Image Not Found", 
                              f"Synthetic image file not found: {image_path}")
    
    def delete_selected_image(self):
        """Delete the selected synthetic image"""
        selected_items = self.images_list.selectedItems()
        if not selected_items:
            return
        
        image_data = selected_items[0].data(Qt.UserRole)
        image_id = image_data.get('image_id')
        
        reply = QMessageBox.question(self, "Delete Synthetic Image", 
                                   f"Are you sure you want to delete synthetic image '{image_id}'?\n\n"
                                   f"This will delete the image, annotations, and metadata.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.generator.delete_synthetic_data(image_id):
                self.refresh_generated_images()
                QMessageBox.information(self, "Deleted", "Synthetic image deleted successfully.")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete synthetic image.")
    
    def clear_all_synthetic_data(self):
        """Clear all synthetic data"""
        generated_images = self.generator.get_generated_images()
        if not generated_images:
            QMessageBox.information(self, "No Data", "No synthetic data to clear.")
            return
        
        reply = QMessageBox.question(self, "Clear All Synthetic Data", 
                                   f"Are you sure you want to delete all {len(generated_images)} synthetic images?\n\n"
                                   f"This action cannot be undone.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            success_count = 0
            for image_data in generated_images:
                image_id = image_data.get('image_id')
                if image_id and self.generator.delete_synthetic_data(image_id):
                    success_count += 1
            
            self.refresh_generated_images()
            QMessageBox.information(self, "Cleared", 
                                  f"Successfully deleted {success_count} synthetic images.") 