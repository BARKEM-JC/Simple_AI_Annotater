"""
Auto Annotation Manager Widget
Manages OpenCV template matching and OpenAI vision-based auto annotation
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem, QTabWidget,
                            QGroupBox, QSlider, QSpinBox, QTextEdit, QProgressBar,
                            QMessageBox, QInputDialog, QCheckBox, QComboBox,
                            QSplitter, QFrame, QScrollArea, QLineEdit, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import numpy as np
from typing import List, Dict, Optional
import json
from opencv_auto_annotator import OpenCVAutoAnnotator, TemplateMatch
from openai_auto_annotator import OpenAIAutoAnnotator, AIAnnotation
import os

class AutoAnnotationWorker(QThread):
    """Worker thread for auto annotation to prevent UI blocking"""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, annotator, frame, method='opencv', custom_labels=None, existing_annotations=None):
        super().__init__()
        self.annotator = annotator
        self.frame = frame
        self.method = method
        self.custom_labels = custom_labels or []
        self.existing_annotations = existing_annotations or []
        # Add frame hash for caching
        self.frame_hash = hash(frame.tobytes()) if frame is not None else 0
    
    def run(self):
        try:
            if self.method == 'opencv':
                # Only run if we have templates
                if not self.annotator.templates:
                    result = {'success': True, 'annotations': []}
                else:
                    matches = self.annotator.find_matches(self.frame, existing_annotations=self.existing_annotations)
                    result = {
                        'success': True,
                        'annotations': [
                            {
                                'region': match.region,
                                'label': match.label,
                                'confidence': match.confidence,
                                'template_id': match.template_id,
                                'type': 'opencv'
                            }
                            for match in matches
                        ]
                    }
            elif self.method == 'openai':
                result = self.annotator.annotate_frame(self.frame, self.custom_labels)
                # Add type indicator
                for annotation in result.get('annotations', []):
                    annotation['type'] = 'openai'
                    
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))

class PendingAnnotation:
    """Represents a pending auto annotation that needs user acceptance"""
    def __init__(self, annotation_data: Dict, frame_index: int):
        self.annotation_data = annotation_data
        self.frame_index = frame_index
        self.accepted = False
        self.type = annotation_data.get('type', 'unknown')  # 'opencv' or 'openai'

class AutoAnnotationManager(QWidget):
    # Signals
    annotation_accepted = pyqtSignal(dict, int)  # annotation_data, frame_index
    template_created = pyqtSignal(str, str)  # template_id, label
    quick_opencv_annotation_requested = pyqtSignal()  # For quick use button
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize annotators
        self.opencv_annotator = OpenCVAutoAnnotator()
        self.openai_annotator = None  # Will be initialized when API key is provided
        
        # State
        self.current_frame = None
        self.current_frame_index = 0
        self.pending_annotations = []  # List of PendingAnnotation objects
        self.worker_thread = None
        
        # Auto-apply settings
        self.auto_apply_opencv = True  # New setting to auto-apply OpenCV annotations
        self.auto_apply_openai = True  # New setting to auto-apply OpenAI annotations
        
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget for different auto annotation methods
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # OpenCV Template Matching Tab
        opencv_tab = self.create_opencv_tab()
        self.tab_widget.addTab(opencv_tab, "OpenCV Template Matching")
        
        # OpenAI Vision Tab
        openai_tab = self.create_openai_tab()
        self.tab_widget.addTab(openai_tab, "OpenAI Vision")
        
        # Pending Annotations Tab
        pending_tab = self.create_pending_tab()
        self.tab_widget.addTab(pending_tab, "Pending Annotations")
        
        # Update template list to show any loaded templates
        self.update_template_list()
        
    def create_opencv_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Template Management Group
        template_group = QGroupBox("Template Management")
        template_layout = QVBoxLayout(template_group)
        
        # Create template from selected region
        create_template_layout = QHBoxLayout()
        self.create_template_btn = QPushButton("Create Template from Selected Region")
        self.create_template_btn.clicked.connect(self.create_template_from_region)
        self.create_template_btn.setEnabled(False)
        create_template_layout.addWidget(self.create_template_btn)
        template_layout.addLayout(create_template_layout)
        
        # Template import/export controls
        import_export_layout = QHBoxLayout()
        self.import_templates_btn = QPushButton("Import Templates")
        self.import_templates_btn.clicked.connect(self.import_templates)
        import_export_layout.addWidget(self.import_templates_btn)
        
        self.export_templates_btn = QPushButton("Export Templates")
        self.export_templates_btn.clicked.connect(self.export_templates)
        import_export_layout.addWidget(self.export_templates_btn)
        
        self.templates_location_btn = QPushButton("Show Templates File")
        self.templates_location_btn.clicked.connect(self.show_templates_location)
        import_export_layout.addWidget(self.templates_location_btn)
        
        template_layout.addLayout(import_export_layout)
        
        # Template list
        self.template_list = QListWidget()
        self.template_list.itemClicked.connect(self.on_template_selected)
        template_layout.addWidget(QLabel("Templates:"))
        template_layout.addWidget(self.template_list)
        
        # Template controls
        template_controls_layout = QHBoxLayout()
        self.remove_template_btn = QPushButton("Remove Template")
        self.remove_template_btn.clicked.connect(self.remove_selected_template)
        self.remove_template_btn.setEnabled(False)
        template_controls_layout.addWidget(self.remove_template_btn)
        
        self.template_threshold_slider = QSlider(Qt.Horizontal)
        self.template_threshold_slider.setRange(10, 100)
        self.template_threshold_slider.setValue(70)
        self.template_threshold_slider.valueChanged.connect(self.update_template_threshold)
        template_controls_layout.addWidget(QLabel("Threshold:"))
        template_controls_layout.addWidget(self.template_threshold_slider)
        
        self.threshold_label = QLabel("0.70")
        template_controls_layout.addWidget(self.threshold_label)
        template_layout.addLayout(template_controls_layout)
        
        # Template details section
        details_group = QGroupBox("Template Details")
        details_layout = QVBoxLayout(details_group)
        
        self.template_details = QTextEdit()
        self.template_details.setReadOnly(True)
        self.template_details.setMaximumHeight(80)
        details_layout.addWidget(self.template_details)
        
        template_layout.addWidget(details_group)
        
        layout.addWidget(template_group)
        
        # Auto Annotation Controls
        annotation_group = QGroupBox("Auto Annotation")
        annotation_layout = QVBoxLayout(annotation_group)
        
        # Auto-apply settings
        self.auto_apply_opencv_checkbox = QCheckBox("Auto-apply OpenCV annotations")
        self.auto_apply_opencv_checkbox.setChecked(self.auto_apply_opencv)
        self.auto_apply_opencv_checkbox.stateChanged.connect(self.on_auto_apply_opencv_changed)
        annotation_layout.addWidget(self.auto_apply_opencv_checkbox)
        
        # Run template matching
        self.run_opencv_btn = QPushButton("Find Template Matches")
        self.run_opencv_btn.clicked.connect(self.run_opencv_annotation)
        self.run_opencv_btn.setEnabled(False)
        annotation_layout.addWidget(self.run_opencv_btn)
        
        layout.addWidget(annotation_group)
        layout.addStretch()
        
        return tab
    
    def create_openai_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Configuration
        config_group = QGroupBox("OpenAI Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # API Key setup
        api_layout = QHBoxLayout()
        self.api_key_btn = QPushButton("Set API Key")
        self.api_key_btn.clicked.connect(self.set_openai_api_key)
        api_layout.addWidget(self.api_key_btn)
        
        self.api_status_label = QLabel("API Key: Not Set")
        api_layout.addWidget(self.api_status_label)
        config_layout.addLayout(api_layout)
        
        layout.addWidget(config_group)
        
        # Custom Labels
        labels_group = QGroupBox("Custom Labels (Optional)")
        labels_layout = QVBoxLayout(labels_group)
        
        self.custom_labels_text = QTextEdit()
        self.custom_labels_text.setPlaceholderText("Enter custom labels separated by commas, e.g.: person, car, building")
        self.custom_labels_text.setMaximumHeight(60)
        labels_layout.addWidget(self.custom_labels_text)
        
        layout.addWidget(labels_group)
        
        # Auto Annotation Controls
        annotation_group = QGroupBox("AI Auto Annotation")
        annotation_layout = QVBoxLayout(annotation_group)
        
        # Auto-apply settings
        self.auto_apply_openai_checkbox = QCheckBox("Auto-apply OpenAI annotations")
        self.auto_apply_openai_checkbox.setChecked(self.auto_apply_openai)
        self.auto_apply_openai_checkbox.stateChanged.connect(self.on_auto_apply_openai_changed)
        annotation_layout.addWidget(self.auto_apply_openai_checkbox)
        
        self.run_ai_btn = QPushButton("Analyze Frame with AI")
        self.run_ai_btn.clicked.connect(self.run_ai_annotation)
        self.run_ai_btn.setEnabled(False)
        annotation_layout.addWidget(self.run_ai_btn)
        
        # Progress bar
        self.ai_progress = QProgressBar()
        self.ai_progress.setVisible(False)
        annotation_layout.addWidget(self.ai_progress)
        
        layout.addWidget(annotation_group)
        layout.addStretch()
        
        return tab
    
    def create_pending_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Info label
        info_label = QLabel("Review and accept/reject auto-generated annotations:")
        layout.addWidget(info_label)
        
        # Pending annotations list
        self.pending_list = QListWidget()
        self.pending_list.itemClicked.connect(self.on_pending_annotation_selected)
        layout.addWidget(self.pending_list)
        
        # Annotation details
        details_group = QGroupBox("Annotation Details")
        details_layout = QVBoxLayout(details_group)
        
        self.annotation_details = QTextEdit()
        self.annotation_details.setReadOnly(True)
        self.annotation_details.setMaximumHeight(100)
        details_layout.addWidget(self.annotation_details)
        
        layout.addWidget(details_group)
        
        # Accept/Reject controls
        controls_layout = QHBoxLayout()
        
        self.accept_btn = QPushButton("Accept Annotation")
        self.accept_btn.clicked.connect(self.accept_selected_annotation)
        self.accept_btn.setEnabled(False)
        controls_layout.addWidget(self.accept_btn)
        
        self.reject_btn = QPushButton("Reject Annotation")
        self.reject_btn.clicked.connect(self.reject_selected_annotation)
        self.reject_btn.setEnabled(False)
        controls_layout.addWidget(self.reject_btn)
        
        self.accept_all_btn = QPushButton("Accept All")
        self.accept_all_btn.clicked.connect(self.accept_all_annotations)
        controls_layout.addWidget(self.accept_all_btn)
        
        self.reject_all_btn = QPushButton("Reject All")
        self.reject_all_btn.clicked.connect(self.reject_all_annotations)
        controls_layout.addWidget(self.reject_all_btn)
        
        layout.addLayout(controls_layout)
        
        return tab
    
    def set_current_frame(self, frame: np.ndarray, frame_index: int):
        """Set the current frame for annotation"""
        self.current_frame = frame
        self.current_frame_index = frame_index
        
        # Enable/disable buttons based on frame availability
        self.run_opencv_btn.setEnabled(len(self.opencv_annotator.templates) > 0)
        self.run_ai_btn.setEnabled(self.openai_annotator is not None)
    
    def set_selected_region(self, region: Dict[str, int], label: str = ""):
        """Called when a region is selected in the main application"""
        self.selected_region = region
        self.selected_label = label
        self.create_template_btn.setEnabled(True)
    
    def create_template_from_region(self):
        """Create a template from the currently selected region"""
        if not hasattr(self, 'selected_region') or self.current_frame is None:
            return
        
        # Get label from user
        label, ok = QInputDialog.getText(self, 'Template Label', 'Enter label for this template:')
        if not ok or not label.strip():
            return
        
        # Create template
        template_id = self.opencv_annotator.add_template_from_region(
            self.current_frame, self.selected_region, label.strip()
        )
        
        # Update template list
        self.update_template_list()
        
        # Enable annotation button
        self.run_opencv_btn.setEnabled(True)
        
        # Emit signal
        self.template_created.emit(template_id, label.strip())
        
        QMessageBox.information(self, "Template Created", f"Template '{label}' created successfully!")
    
    def update_template_list(self):
        """Update the template list widget"""
        self.template_list.clear()
        
        templates = self.opencv_annotator.list_templates()
        for template_info in templates:
            # Enhanced display with metadata
            usage_count = template_info.get('usage_count', 0)
            created_date = ""
            if template_info.get('created_timestamp'):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(template_info['created_timestamp'])
                    created_date = dt.strftime(" [%m/%d]")
                except:
                    pass
            
            item_text = f"{template_info['label']} (Used: {usage_count}x){created_date}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, template_info['id'])
            
            # Add tooltip with full details
            tooltip = f"Label: {template_info['label']}\n"
            tooltip += f"Template ID: {template_info['id']}\n"
            tooltip += f"Size: {template_info['size'][1]}x{template_info['size'][0]}\n"
            tooltip += f"Threshold: {template_info['threshold']:.2f}\n"
            tooltip += f"Usage Count: {usage_count}\n"
            if template_info.get('created_timestamp'):
                tooltip += f"Created: {template_info['created_timestamp']}"
            item.setToolTip(tooltip)
            
            self.template_list.addItem(item)
    
    def on_template_selected(self, item):
        """Handle template selection"""
        template_id = item.data(Qt.UserRole)
        template_info = self.opencv_annotator.get_template_info(template_id)
        
        if template_info:
            # Update threshold slider
            threshold = int(template_info['threshold'] * 100)
            self.template_threshold_slider.setValue(threshold)
            self.threshold_label.setText(f"{template_info['threshold']:.2f}")
            
            # Update details display
            details = f"Label: {template_info['label']}\n"
            details += f"Size: {template_info['template'].shape[1]}x{template_info['template'].shape[0]}\n"
            details += f"Usage Count: {template_info.get('usage_count', 0)}\n"
            if template_info.get('created_timestamp'):
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(template_info['created_timestamp'])
                    details += f"Created: {dt.strftime('%Y-%m-%d %H:%M')}\n"
                except:
                    details += f"Created: {template_info['created_timestamp']}\n"
            
            self.template_details.setText(details)
            
        self.remove_template_btn.setEnabled(True)
    
    def remove_selected_template(self):
        """Remove the selected template"""
        current_item = self.template_list.currentItem()
        if current_item:
            template_id = current_item.data(Qt.UserRole)
            self.opencv_annotator.remove_template(template_id)
            self.update_template_list()
            self.remove_template_btn.setEnabled(False)
            
            # Disable annotation button if no templates left
            if len(self.opencv_annotator.templates) == 0:
                self.run_opencv_btn.setEnabled(False)
    
    def update_template_threshold(self, value):
        """Update template threshold"""
        threshold = value / 100.0
        self.threshold_label.setText(f"{threshold:.2f}")
        
        # Update threshold for selected template
        current_item = self.template_list.currentItem()
        if current_item:
            template_id = current_item.data(Qt.UserRole)
            self.opencv_annotator.set_template_threshold(template_id, threshold)
    
    def set_openai_api_key(self):
        """Set OpenAI API key"""
        api_key, ok = QInputDialog.getText(self, 'OpenAI API Key', 'Enter your OpenAI API key:', 
                                          QLineEdit.Password)
        if ok and api_key.strip():
            try:
                self.openai_annotator = OpenAIAutoAnnotator(api_key.strip())
                self.api_status_label.setText("API Key: Set ✓")
                self.run_ai_btn.setEnabled(self.current_frame is not None)
                QMessageBox.information(self, "Success", "OpenAI API key set successfully!")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to initialize OpenAI client: {str(e)}")
    
    def run_opencv_annotation(self):
        """Run OpenCV template matching annotation"""
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No frame loaded")
            return
        
        # Get existing annotations to avoid duplicates if auto-apply is enabled
        existing_annotations = []
        if hasattr(self.parent(), 'video_player') and self.parent().video_player:
            for region in self.parent().video_player.regions:
                region_data = self.parent().video_player.get_region_data(region.id)
                if region_data:
                    existing_annotations.append(region_data)
        
        self.run_opencv_btn.setEnabled(False)
        self.worker_thread = AutoAnnotationWorker(
            self.opencv_annotator, 
            self.current_frame, 
            method='opencv',
            existing_annotations=existing_annotations
        )
        self.worker_thread.finished.connect(self.on_annotation_finished)
        self.worker_thread.error.connect(self.on_annotation_error)
        self.worker_thread.start()
    
    def run_ai_annotation(self):
        """Run OpenAI vision annotation"""
        if self.current_frame is None or self.openai_annotator is None:
            return
        
        # Get custom labels
        custom_labels_text = self.custom_labels_text.toPlainText().strip()
        custom_labels = [label.strip() for label in custom_labels_text.split(',') if label.strip()] if custom_labels_text else None
        
        # Show progress
        self.ai_progress.setVisible(True)
        self.ai_progress.setRange(0, 0)  # Indeterminate progress
        
        # Start worker thread
        self.worker_thread = AutoAnnotationWorker(
            self.openai_annotator, self.current_frame, 'openai', custom_labels
        )
        self.worker_thread.finished.connect(self.on_annotation_finished)
        self.worker_thread.error.connect(self.on_annotation_error)
        self.worker_thread.start()
        
        self.run_ai_btn.setEnabled(False)
        self.run_ai_btn.setText("Processing...")
    
    @pyqtSlot(dict)
    def on_annotation_finished(self, result):
        """Handle annotation completion"""
        # Reset UI
        self.run_opencv_btn.setEnabled(True)
        self.run_opencv_btn.setText("Find Template Matches")
        self.run_ai_btn.setEnabled(True)
        self.run_ai_btn.setText("Analyze Frame with AI")
        self.ai_progress.setVisible(False)
        
        if result.get('success', False):
            annotations = result.get('annotations', [])
            if annotations:
                # Check if we should auto-apply
                annotation_type = annotations[0].get('type', 'unknown')
                should_auto_apply = ((annotation_type == 'opencv' and self.auto_apply_opencv) or
                                   (annotation_type == 'openai' and self.auto_apply_openai))
                
                if should_auto_apply:
                    # Auto-apply all annotations
                    for annotation in annotations:
                        self.annotation_accepted.emit(annotation, self.current_frame_index)
                else:
                    # Add to pending annotations for manual review
                    for annotation in annotations:
                        pending = PendingAnnotation(annotation, self.current_frame_index)
                        self.pending_annotations.append(pending)
                    
                    self.update_pending_list()
                    
                    # Switch to pending tab
                    self.tab_widget.setCurrentIndex(2)
                    
                    QMessageBox.information(self, "Success", 
                                          f"Found {len(annotations)} annotations. Please review them in the Pending tab.")
            else:
                QMessageBox.information(self, "No Matches", "No annotations were found.")
        else:
            error_msg = result.get('error', 'Unknown error occurred')
            QMessageBox.warning(self, "Error", f"Annotation failed: {error_msg}")
    
    @pyqtSlot(str)
    def on_annotation_error(self, error_msg):
        """Handle annotation error"""
        # Reset UI
        self.run_opencv_btn.setEnabled(True)
        self.run_opencv_btn.setText("Find Template Matches")
        self.run_ai_btn.setEnabled(True)
        self.run_ai_btn.setText("Analyze Frame with AI")
        self.ai_progress.setVisible(False)
        
        QMessageBox.warning(self, "Error", f"Annotation failed: {error_msg}")
    
    def update_pending_list(self):
        """Update the pending annotations list"""
        self.pending_list.clear()
        
        for i, pending in enumerate(self.pending_annotations):
            annotation = pending.annotation_data
            type_indicator = "🔍" if pending.type == 'opencv' else "🤖"
            confidence = annotation.get('confidence', 0)
            
            if pending.type == 'opencv':
                conf_text = f"{confidence:.2f}"
            else:
                conf_text = f"{confidence*100:.0f}%" if confidence <= 1.0 else f"{confidence:.0f}%"
            
            item_text = f"{type_indicator} {annotation.get('label', 'Unknown')} (Conf: {conf_text})"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)
            
            # Color code accepted/pending
            if pending.accepted:
                item.setBackground(QColor(144, 238, 144))  # Light green
            
            self.pending_list.addItem(item)
    
    def on_pending_annotation_selected(self, item):
        """Handle pending annotation selection"""
        index = item.data(Qt.UserRole)
        if 0 <= index < len(self.pending_annotations):
            pending = self.pending_annotations[index]
            annotation = pending.annotation_data
            
            # Show details
            details = f"Label: {annotation.get('label', 'Unknown')}\n"
            details += f"Type: {pending.type.upper()}\n"
            details += f"Confidence: {annotation.get('confidence', 0)}\n"
            details += f"Region: {annotation.get('region', {})}\n"
            
            if 'description' in annotation:
                details += f"Description: {annotation['description']}\n"
            
            self.annotation_details.setText(details)
            
            # Enable buttons
            self.accept_btn.setEnabled(not pending.accepted)
            self.reject_btn.setEnabled(True)
    
    def accept_selected_annotation(self):
        """Accept the selected annotation"""
        current_item = self.pending_list.currentItem()
        if current_item:
            index = current_item.data(Qt.UserRole)
            if 0 <= index < len(self.pending_annotations):
                pending = self.pending_annotations[index]
                pending.accepted = True
                
                # Emit signal to main application
                self.annotation_accepted.emit(pending.annotation_data, pending.frame_index)
                
                self.update_pending_list()
                self.accept_btn.setEnabled(False)
    
    def reject_selected_annotation(self):
        """Reject the selected annotation"""
        current_item = self.pending_list.currentItem()
        if current_item:
            index = current_item.data(Qt.UserRole)
            if 0 <= index < len(self.pending_annotations):
                # Remove from pending list
                del self.pending_annotations[index]
                self.update_pending_list()
                
                # Clear details
                self.annotation_details.clear()
                self.accept_btn.setEnabled(False)
                self.reject_btn.setEnabled(False)
    
    def accept_all_annotations(self):
        """Accept all pending annotations"""
        for pending in self.pending_annotations:
            if not pending.accepted:
                pending.accepted = True
                self.annotation_accepted.emit(pending.annotation_data, pending.frame_index)
        
        self.update_pending_list()
    
    def reject_all_annotations(self):
        """Reject all pending annotations"""
        self.pending_annotations.clear()
        self.update_pending_list()
        self.annotation_details.clear()
        self.accept_btn.setEnabled(False)
        self.reject_btn.setEnabled(False)
    
    def on_auto_apply_opencv_changed(self, state):
        """Handle auto-apply OpenCV checkbox change"""
        self.auto_apply_opencv = state == Qt.Checked
    
    def on_auto_apply_openai_changed(self, state):
        """Handle auto-apply OpenAI checkbox change"""
        self.auto_apply_openai = state == Qt.Checked

    def run_quick_opencv_annotation(self):
        """Run OpenCV annotation quickly (for use with quick button)"""
        if self.current_frame is None or len(self.opencv_annotator.templates) == 0:
            return False
        
        # Temporarily enable auto-apply for quick annotation
        original_auto_apply = self.auto_apply_opencv
        self.auto_apply_opencv = True
        
        # Run annotation
        self.run_opencv_annotation()
        
        # Restore original setting
        self.auto_apply_opencv = original_auto_apply
        
        return True

    def import_templates(self):
        """Import templates from a JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Import Templates", 
            "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            reply = QMessageBox.question(
                self, 
                "Import Templates", 
                "Do you want to merge with existing templates or replace all templates?\n\n"
                "Click 'Yes' to merge (recommended)\n"
                "Click 'No' to replace all existing templates",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Cancel:
                return
            
            merge = reply == QMessageBox.Yes
            
            if self.opencv_annotator.import_templates(file_path, merge=merge):
                self.update_template_list()
                count = len(self.opencv_annotator.templates)
                action = "merged with" if merge else "replaced"
                QMessageBox.information(
                    self, 
                    "Import Success", 
                    f"Templates successfully {action} existing templates!\n"
                    f"Total templates: {count}"
                )
                
                # Enable annotation button if we have templates
                if count > 0:
                    self.run_opencv_btn.setEnabled(self.current_frame is not None)
            else:
                QMessageBox.warning(
                    self, 
                    "Import Failed", 
                    "Failed to import templates. Please check the file format."
                )
    
    def export_templates(self):
        """Export templates to a JSON file"""
        if not self.opencv_annotator.templates:
            QMessageBox.information(
                self, 
                "No Templates", 
                "No templates to export. Create some templates first."
            )
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Templates", 
            "opencv_templates.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            if self.opencv_annotator.export_templates(file_path):
                QMessageBox.information(
                    self, 
                    "Export Success", 
                    f"Successfully exported {len(self.opencv_annotator.templates)} templates to:\n{file_path}"
                )
            else:
                QMessageBox.warning(
                    self, 
                    "Export Failed", 
                    "Failed to export templates. Please check file permissions."
                )
    
    def show_templates_location(self):
        """Show the location of the templates file"""
        templates_path = self.opencv_annotator.get_templates_file_path()
        templates_exist = os.path.exists(templates_path)
        
        msg = f"Templates file location:\n{templates_path}\n\n"
        if templates_exist:
            msg += f"File exists with {len(self.opencv_annotator.templates)} templates."
        else:
            msg += "File does not exist yet (will be created when you add templates)."
        
        QMessageBox.information(self, "Templates File Location", msg)

    def run_auto_annotation(self, frame, frame_index, existing_annotations=None):
        """Run auto-annotation for a specific frame (called by main window)"""
        if frame is None:
            return
            
        self.set_current_frame(frame, frame_index)
        
        # Get existing annotations from the calling context or use provided ones
        if existing_annotations is None:
            existing_annotations = []
            if hasattr(self.parent(), 'video_player') and self.parent().video_player:
                for region in self.parent().video_player.regions:
                    region_data = self.parent().video_player.get_region_data(region.id)
                    if region_data:
                        existing_annotations.append(region_data)
        
        # Run OpenCV annotation with existing annotations filter
        matches = self.opencv_annotator.find_matches(frame, existing_annotations=existing_annotations)
        
        # Auto-apply matches if enabled
        if self.auto_apply_opencv and hasattr(self.parent(), 'video_player'):
            for match in matches:
                region_data = {
                    'x': match.region['x'],
                    'y': match.region['y'],
                    'width': match.region['width'],
                    'height': match.region['height'],
                    'label': match.label,
                    'id': f"auto_{match.template_id}_{frame_index}_{len(existing_annotations)}"
                }
                self.parent().video_player.add_region_from_data(region_data)
        
        return matches 