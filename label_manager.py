from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                            QComboBox, QGroupBox, QInputDialog, QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
import json
from pathlib import Path

class LabelManager(QWidget):
    label_changed = pyqtSignal(str, str)  # region_id, new_label
    quick_opencv_annotation_requested = pyqtSignal()  # For quick OpenCV annotation
    add_template_requested = pyqtSignal(str, str)  # region_id, label - New signal for adding template
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_region_id = None
        self.predefined_labels = ["person", "car", "truck", "bicycle", "motorcycle", 
                                 "bus", "train", "boat", "airplane", "animal"]
        self.custom_labels = []
        
        self.init_ui()
        self.load_custom_labels()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Predefined labels group
        predefined_group = QGroupBox("Predefined Labels")
        predefined_layout = QVBoxLayout(predefined_group)
        
        self.predefined_combo = QComboBox()
        self.predefined_combo.addItem("Select predefined label...")
        self.predefined_combo.addItems(self.predefined_labels)
        self.predefined_combo.currentTextChanged.connect(self.on_predefined_selected)
        predefined_layout.addWidget(self.predefined_combo)
        
        layout.addWidget(predefined_group)
        
        # Custom labels group
        custom_group = QGroupBox("Custom Labels")
        custom_layout = QVBoxLayout(custom_group)
        
        # Add custom label
        add_layout = QHBoxLayout()
        self.custom_label_input = QLineEdit()
        self.custom_label_input.setPlaceholderText("Enter custom label...")
        self.custom_label_input.returnPressed.connect(self.add_custom_label)
        add_layout.addWidget(self.custom_label_input)
        
        self.add_custom_btn = QPushButton("Add")
        self.add_custom_btn.clicked.connect(self.add_custom_label)
        add_layout.addWidget(self.add_custom_btn)
        
        custom_layout.addLayout(add_layout)
        
        # Custom labels list
        self.custom_labels_list = QListWidget()
        self.custom_labels_list.itemClicked.connect(self.on_custom_label_selected)
        custom_layout.addWidget(self.custom_labels_list)
        
        # Remove custom label button
        self.remove_custom_btn = QPushButton("Remove Selected")
        self.remove_custom_btn.clicked.connect(self.remove_custom_label)
        self.remove_custom_btn.setEnabled(False)
        custom_layout.addWidget(self.remove_custom_btn)
        
        layout.addWidget(custom_group)
        
        # Current region info
        region_group = QGroupBox("Current Region")
        region_layout = QVBoxLayout(region_group)
        
        self.region_info_label = QLabel("No region selected")
        region_layout.addWidget(self.region_info_label)
        
        # Current label input
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label:"))
        
        self.current_label_input = QLineEdit()
        self.current_label_input.setPlaceholderText("Enter region label...")
        self.current_label_input.textChanged.connect(self.on_label_text_changed)
        label_layout.addWidget(self.current_label_input)
        
        region_layout.addLayout(label_layout)
        
        # Apply label button
        self.apply_label_btn = QPushButton("Apply Label")
        self.apply_label_btn.clicked.connect(self.apply_current_label)
        self.apply_label_btn.setEnabled(False)
        region_layout.addWidget(self.apply_label_btn)
        
        # Add Template button (NEW)
        self.add_template_btn = QPushButton("Add as OpenCV Template")
        self.add_template_btn.clicked.connect(self.add_region_as_template)
        self.add_template_btn.setEnabled(False)
        self.add_template_btn.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        region_layout.addWidget(self.add_template_btn)
        
        # Quick OpenCV annotation button
        self.quick_opencv_btn = QPushButton("Quick OpenCV Auto-Annotate")
        self.quick_opencv_btn.clicked.connect(self.request_quick_opencv_annotation)
        self.quick_opencv_btn.setEnabled(False)
        self.quick_opencv_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        region_layout.addWidget(self.quick_opencv_btn)
        
        layout.addWidget(region_group)
        
        layout.addStretch()
    
    def add_custom_label(self):
        label_text = self.custom_label_input.text().strip()
        if label_text and label_text not in self.custom_labels:
            self.custom_labels.append(label_text)
            
            item = QListWidgetItem(label_text)
            self.custom_labels_list.addItem(item)
            
            self.custom_label_input.clear()
            self.save_custom_labels()
    
    def remove_custom_label(self):
        current_item = self.custom_labels_list.currentItem()
        if current_item:
            label_text = current_item.text()
            
            # Remove from list
            row = self.custom_labels_list.row(current_item)
            self.custom_labels_list.takeItem(row)
            
            # Remove from custom labels
            if label_text in self.custom_labels:
                self.custom_labels.remove(label_text)
            
            self.save_custom_labels()
            self.remove_custom_btn.setEnabled(False)
    
    def on_predefined_selected(self, label_text):
        if label_text != "Select predefined label...":
            self.current_label_input.setText(label_text)
            self.predefined_combo.setCurrentIndex(0)  # Reset to placeholder
    
    def on_custom_label_selected(self, item):
        self.current_label_input.setText(item.text())
        self.remove_custom_btn.setEnabled(True)
    
    def on_label_text_changed(self, text):
        has_text = bool(text.strip())
        has_region = self.current_region_id is not None
        self.apply_label_btn.setEnabled(has_text and has_region)
        self.add_template_btn.setEnabled(has_text and has_region)
    
    def apply_current_label(self):
        if self.current_region_id and self.current_label_input.text().strip():
            label = self.current_label_input.text().strip()
            self.label_changed.emit(self.current_region_id, label)
    
    def add_region_as_template(self):
        """Add current region as template to OpenCV auto-annotator"""
        if self.current_region_id and self.current_label_input.text().strip():
            label = self.current_label_input.text().strip()
            self.add_template_requested.emit(self.current_region_id, label)
    
    def request_quick_opencv_annotation(self):
        """Request quick OpenCV annotation"""
        self.quick_opencv_annotation_requested.emit()
    
    def add_region(self, region_data):
        # Called when a new region is added
        pass
    
    def select_region(self, region_id):
        self.current_region_id = region_id
        if region_id:
            self.region_info_label.setText(f"Region ID: {region_id[:8]}...")
            has_text = bool(self.current_label_input.text().strip())
            self.apply_label_btn.setEnabled(has_text)
            self.add_template_btn.setEnabled(has_text)
            self.quick_opencv_btn.setEnabled(True)
        else:
            self.region_info_label.setText("No region selected")
            self.apply_label_btn.setEnabled(False)
            self.add_template_btn.setEnabled(False)
            self.quick_opencv_btn.setEnabled(False)
    
    def remove_region(self, region_id):
        if self.current_region_id == region_id:
            self.current_region_id = None
            self.region_info_label.setText("No region selected")
            self.apply_label_btn.setEnabled(False)
            self.add_template_btn.setEnabled(False)
            self.quick_opencv_btn.setEnabled(False)
    
    def clear_regions(self):
        self.current_region_id = None
        self.region_info_label.setText("No region selected")
        self.apply_label_btn.setEnabled(False)
        self.add_template_btn.setEnabled(False)
        self.quick_opencv_btn.setEnabled(False)
    
    def load_custom_labels(self):
        labels_file = Path("custom_labels.json")
        if labels_file.exists():
            try:
                with open(labels_file, 'r') as f:
                    data = json.load(f)
                    self.custom_labels = data.get('labels', [])
                
                # Populate list widget
                for label in self.custom_labels:
                    item = QListWidgetItem(label)
                    self.custom_labels_list.addItem(item)
                    
            except Exception as e:
                print(f"Error loading custom labels: {e}")
    
    def save_custom_labels(self):
        labels_file = Path("custom_labels.json")
        try:
            data = {'labels': self.custom_labels}
            with open(labels_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving custom labels: {e}")
    
    def get_all_labels(self):
        return self.predefined_labels + self.custom_labels 
    
    def get_current_label(self):
        """Get the current label text from the input field"""
        return self.current_label_input.text().strip() 