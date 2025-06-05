from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QPushButton, QListWidget, QListWidgetItem,
                            QComboBox, QGroupBox, QInputDialog, QMessageBox,
                            QSpinBox, QTextEdit, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSignal
import json
from pathlib import Path
import uuid
import time

class VideoClip:
    def __init__(self, start_frame, end_frame, label="", clip_id=None):
        self.id = clip_id or str(uuid.uuid4())
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.label = label
        self.created_at = time.time()
        self.description = ""
        
    def duration_frames(self):
        return self.end_frame - self.start_frame + 1
    
    def to_dict(self):
        return {
            'id': self.id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'label': self.label,
            'description': self.description,
            'created_at': self.created_at,
            'duration_frames': self.duration_frames()
        }
    
    @classmethod
    def from_dict(cls, data):
        clip = cls(
            start_frame=data['start_frame'],
            end_frame=data['end_frame'],
            label=data.get('label', ''),
            clip_id=data.get('id')
        )
        clip.description = data.get('description', '')
        clip.created_at = data.get('created_at', time.time())
        return clip

class ClipManager(QWidget):
    clip_created = pyqtSignal(dict)  # clip_data
    clip_selected = pyqtSignal(str)  # clip_id
    clip_deleted = pyqtSignal(str)   # clip_id
    clip_preview_requested = pyqtSignal(int, int)  # start_frame, end_frame
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.clips = []
        self.current_clip = None
        self.current_frame_index = 0
        self.frame_count = 0
        
        # Predefined action labels for MoViNet
        self.predefined_labels = [
            "walking", "running", "jumping", "sitting", "standing",
            "waving", "clapping", "dancing", "eating", "drinking",
            "throwing", "catching", "kicking", "hitting",
            "bowling", "scoring", "celebrating", "winning", "losing"
        ]
        
        self.init_ui()
        self.load_clips()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Clip creation group
        creation_group = QGroupBox("Create Video Clip")
        creation_layout = QVBoxLayout(creation_group)
        
        # Frame range selection
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Start Frame:"))
        
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setMinimum(0)
        self.start_frame_spin.setMaximum(999999)
        frame_layout.addWidget(self.start_frame_spin)
        
        frame_layout.addWidget(QLabel("End Frame:"))
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setMinimum(0)
        self.end_frame_spin.setMaximum(999999)
        frame_layout.addWidget(self.end_frame_spin)
        
        creation_layout.addLayout(frame_layout)
        
        # Quick clip creation buttons
        quick_layout = QHBoxLayout()
        
        self.clip_5s_btn = QPushButton("5 Second Clip")
        self.clip_5s_btn.clicked.connect(lambda: self.create_quick_clip(5))
        quick_layout.addWidget(self.clip_5s_btn)
        
        self.clip_10s_btn = QPushButton("10 Second Clip")
        self.clip_10s_btn.clicked.connect(lambda: self.create_quick_clip(10))
        quick_layout.addWidget(self.clip_10s_btn)
        
        self.clip_30s_btn = QPushButton("30 Second Clip")
        self.clip_30s_btn.clicked.connect(lambda: self.create_quick_clip(30))
        quick_layout.addWidget(self.clip_30s_btn)
        
        creation_layout.addLayout(quick_layout)
        
        # Set current frame buttons
        current_frame_layout = QHBoxLayout()
        
        self.set_start_btn = QPushButton("Set Start to Current Frame")
        self.set_start_btn.clicked.connect(self.set_start_to_current)
        current_frame_layout.addWidget(self.set_start_btn)
        
        self.set_end_btn = QPushButton("Set End to Current Frame")
        self.set_end_btn.clicked.connect(self.set_end_to_current)
        current_frame_layout.addWidget(self.set_end_btn)
        
        creation_layout.addLayout(current_frame_layout)
        
        # Create clip button
        self.create_clip_btn = QPushButton("Create Clip")
        self.create_clip_btn.clicked.connect(self.create_clip)
        creation_layout.addWidget(self.create_clip_btn)
        
        layout.addWidget(creation_group)
        
        # Clip labeling group
        labeling_group = QGroupBox("Clip Labeling")
        labeling_layout = QVBoxLayout(labeling_group)
        
        # Predefined labels
        self.label_combo = QComboBox()
        self.label_combo.addItem("Select action label...")
        self.label_combo.addItems(self.predefined_labels)
        self.label_combo.currentTextChanged.connect(self.on_predefined_label_selected)
        labeling_layout.addWidget(self.label_combo)
        
        # Custom label input
        label_input_layout = QHBoxLayout()
        self.custom_label_input = QLineEdit()
        self.custom_label_input.setPlaceholderText("Enter custom action label...")
        label_input_layout.addWidget(self.custom_label_input)
        
        self.add_custom_label_btn = QPushButton("Add Custom")
        self.add_custom_label_btn.clicked.connect(self.add_custom_label)
        label_input_layout.addWidget(self.add_custom_label_btn)
        
        labeling_layout.addLayout(label_input_layout)
        
        # Description
        labeling_layout.addWidget(QLabel("Description (optional):"))
        self.description_text = QTextEdit()
        self.description_text.setMaximumHeight(60)
        self.description_text.setPlaceholderText("Describe the action in detail...")
        labeling_layout.addWidget(self.description_text)
        
        layout.addWidget(labeling_group)
        
        # Clips list
        clips_group = QGroupBox("Created Clips")
        clips_layout = QVBoxLayout(clips_group)
        
        self.clips_list = QListWidget()
        self.clips_list.itemClicked.connect(self.on_clip_selected)
        clips_layout.addWidget(self.clips_list)
        
        # Clip controls
        controls_layout = QHBoxLayout()
        
        self.preview_clip_btn = QPushButton("Preview Clip")
        self.preview_clip_btn.clicked.connect(self.preview_selected_clip)
        self.preview_clip_btn.setEnabled(False)
        controls_layout.addWidget(self.preview_clip_btn)
        
        self.update_clip_btn = QPushButton("Update Label")
        self.update_clip_btn.clicked.connect(self.update_selected_clip)
        self.update_clip_btn.setEnabled(False)
        controls_layout.addWidget(self.update_clip_btn)
        
        self.delete_clip_btn = QPushButton("Delete Clip")
        self.delete_clip_btn.clicked.connect(self.delete_selected_clip)
        self.delete_clip_btn.setEnabled(False)
        controls_layout.addWidget(self.delete_clip_btn)
        
        clips_layout.addLayout(controls_layout)
        layout.addWidget(clips_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("No clips created")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
    
    def set_video_info(self, frame_count, fps):
        """Set video information for clip creation"""
        self.frame_count = frame_count
        self.fps = fps
        self.start_frame_spin.setMaximum(frame_count - 1)
        self.end_frame_spin.setMaximum(frame_count - 1)
    
    def set_current_frame(self, frame_index):
        """Update current frame for clip creation"""
        self.current_frame_index = frame_index
    
    def create_quick_clip(self, duration_seconds):
        """Create a clip of specified duration from current frame"""
        if self.fps <= 0:
            return
        
        frames_duration = int(duration_seconds * self.fps)
        start_frame = max(0, self.current_frame_index - frames_duration // 2)
        end_frame = min(self.frame_count - 1, start_frame + frames_duration - 1)
        
        self.start_frame_spin.setValue(start_frame)
        self.end_frame_spin.setValue(end_frame)
    
    def set_start_to_current(self):
        """Set start frame to current frame"""
        self.start_frame_spin.setValue(self.current_frame_index)
    
    def set_end_to_current(self):
        """Set end frame to current frame"""
        self.end_frame_spin.setValue(self.current_frame_index)
    
    def create_clip(self):
        """Create a new video clip"""
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        
        if start_frame >= end_frame:
            QMessageBox.warning(self, "Invalid Range", "End frame must be greater than start frame")
            return
        
        # Check for minimum clip length
        duration_frames = end_frame - start_frame + 1
        min_frames = max(5, int(0.5 * self.fps)) if self.fps > 0 else 5
        
        if duration_frames < min_frames:
            QMessageBox.warning(self, "Clip Too Short", 
                              f"Clip must be at least {min_frames} frames ({min_frames/self.fps:.1f}s)")
            return
        
        clip = VideoClip(start_frame, end_frame)
        self.clips.append(clip)
        self.update_clips_list()
        self.save_clips()
        
        # Signal clip creation
        self.clip_created.emit(clip.to_dict())
        
        # Clear inputs
        self.custom_label_input.clear()
        self.description_text.clear()
    
    def on_predefined_label_selected(self, label_text):
        """Handle predefined label selection"""
        if label_text != "Select action label...":
            self.custom_label_input.setText(label_text)
            self.label_combo.setCurrentIndex(0)  # Reset to placeholder
    
    def add_custom_label(self):
        """Add custom label to predefined list"""
        label_text = self.custom_label_input.text().strip()
        if label_text and label_text not in self.predefined_labels:
            self.predefined_labels.append(label_text)
            self.label_combo.addItem(label_text)
    
    def update_clips_list(self):
        """Update the clips list widget"""
        self.clips_list.clear()
        
        for clip in self.clips:
            duration_s = (clip.end_frame - clip.start_frame + 1) / self.fps if self.fps > 0 else 0
            label_text = clip.label if clip.label else "Unlabeled"
            
            item_text = f"{label_text} | Frames {clip.start_frame}-{clip.end_frame} ({duration_s:.1f}s)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, clip.id)
            self.clips_list.addItem(item)
        
        self.update_statistics()
    
    def update_statistics(self):
        """Update statistics display"""
        total_clips = len(self.clips)
        labeled_clips = len([c for c in self.clips if c.label])
        
        if total_clips == 0:
            self.stats_label.setText("No clips created")
        else:
            total_duration = sum((c.end_frame - c.start_frame + 1) for c in self.clips)
            duration_s = total_duration / self.fps if self.fps > 0 else 0
            
            stats_text = f"Total clips: {total_clips} | Labeled: {labeled_clips} | Duration: {duration_s:.1f}s"
            self.stats_label.setText(stats_text)
    
    def on_clip_selected(self, item):
        """Handle clip selection"""
        clip_id = item.data(Qt.UserRole)
        self.current_clip = next((c for c in self.clips if c.id == clip_id), None)
        
        if self.current_clip:
            self.custom_label_input.setText(self.current_clip.label)
            self.description_text.setPlainText(self.current_clip.description)
            
            self.preview_clip_btn.setEnabled(True)
            self.update_clip_btn.setEnabled(True)
            self.delete_clip_btn.setEnabled(True)
            
            self.clip_selected.emit(clip_id)
    
    def preview_selected_clip(self):
        """Preview the selected clip"""
        if self.current_clip:
            # This will be handled by the main window
            self.clip_preview_requested.emit(self.current_clip.start_frame, self.current_clip.end_frame)
    
    def update_selected_clip(self):
        """Update the selected clip's label and description"""
        if self.current_clip:
            self.current_clip.label = self.custom_label_input.text().strip()
            self.current_clip.description = self.description_text.toPlainText().strip()
            
            self.update_clips_list()
            self.save_clips()
    
    def delete_selected_clip(self):
        """Delete the selected clip"""
        if self.current_clip:
            reply = QMessageBox.question(self, "Delete Clip", 
                                       "Are you sure you want to delete this clip?",
                                       QMessageBox.Yes | QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                clip_id = self.current_clip.id
                self.clips.remove(self.current_clip)
                self.current_clip = None
                
                self.update_clips_list()
                self.save_clips()
                
                self.preview_clip_btn.setEnabled(False)
                self.update_clip_btn.setEnabled(False)
                self.delete_clip_btn.setEnabled(False)
                
                self.clip_deleted.emit(clip_id)
    
    def get_clip_by_id(self, clip_id):
        """Get clip by ID"""
        return next((c for c in self.clips if c.id == clip_id), None)
    
    def save_clips(self):
        """Save clips to file"""
        clips_file = Path("labeled_data") / "clips.json"
        clips_file.parent.mkdir(exist_ok=True)
        
        try:
            clips_data = [clip.to_dict() for clip in self.clips]
            with open(clips_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
                
            # Also save individual clip files for MoViNet export compatibility
            self.save_individual_clip_files()
        except Exception as e:
            print(f"Error saving clips: {e}")
    
    def save_individual_clip_files(self):
        """Save individual clip files for MoViNet export compatibility"""
        labeled_data_dir = Path("labeled_data")
        
        # Remove existing individual clip files
        for clip_file in labeled_data_dir.glob("clip_*.json"):
            clip_file.unlink()
        
        # Save each clip as individual file
        for i, clip in enumerate(self.clips, 1):
            if clip.label:  # Only save labeled clips
                clip_file = labeled_data_dir / f"clip_{i:03d}.json"
                try:
                    with open(clip_file, 'w') as f:
                        json.dump(clip.to_dict(), f, indent=2)
                except Exception as e:
                    print(f"Error saving individual clip file {clip_file}: {e}")
    
    def load_clips(self):
        """Load clips from file"""
        clips_file = Path("labeled_data") / "clips.json"
        
        if clips_file.exists():
            try:
                with open(clips_file, 'r') as f:
                    clips_data = json.load(f)
                
                self.clips = [VideoClip.from_dict(data) for data in clips_data]
                self.update_clips_list()
                
            except Exception as e:
                print(f"Error loading clips: {e}")
    
    def clear_all_clips(self):
        """Clear all clips"""
        reply = QMessageBox.question(self, "Clear All Clips", 
                                   "Are you sure you want to delete all clips?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.clips.clear()
            self.current_clip = None
            self.update_clips_list()
            self.save_clips()
            
            self.preview_clip_btn.setEnabled(False)
            self.update_clip_btn.setEnabled(False)
            self.delete_clip_btn.setEnabled(False)
    
    def export_clips_metadata(self):
        """Export clips metadata for MoViNet training"""
        export_file = Path("labeled_data") / "movinet_clips.json"
        
        try:
            clips_data = []
            for clip in self.clips:
                if clip.label:  # Only export labeled clips
                    clips_data.append(clip.to_dict())
            
            metadata = {
                "clips": clips_data,
                "total_clips": len(clips_data),
                "classes": list(set(c.label for c in self.clips if c.label))
            }
            
            with open(export_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return str(export_file)
            
        except Exception as e:
            print(f"Error exporting clips metadata: {e}")
            return None 