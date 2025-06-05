import cv2
import numpy as np
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush
import uuid
import time

class Region:
    def __init__(self, x, y, width, height, label="", region_id=None):
        self.id = region_id or str(uuid.uuid4())
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label
        self.selected = False
        self.created_at = time.time()
        
        # For resizing
        self.resize_handles = []
        self.calculate_resize_handles()
    
    def calculate_resize_handles(self):
        handle_size = 8
        self.resize_handles = [
            QRect(self.x - handle_size//2, self.y - handle_size//2, handle_size, handle_size),  # Top-left
            QRect(self.x + self.width - handle_size//2, self.y - handle_size//2, handle_size, handle_size),  # Top-right
            QRect(self.x - handle_size//2, self.y + self.height - handle_size//2, handle_size, handle_size),  # Bottom-left
            QRect(self.x + self.width - handle_size//2, self.y + self.height - handle_size//2, handle_size, handle_size),  # Bottom-right
        ]
    
    def contains_point(self, x, y):
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def get_handle_at_point(self, x, y):
        for i, handle in enumerate(self.resize_handles):
            if handle.contains(QPoint(x, y)):
                return i
        return -1
    
    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.calculate_resize_handles()
    
    def resize(self, handle_index, dx, dy):
        if handle_index == 0:  # Top-left
            self.x += dx
            self.y += dy
            self.width -= dx
            self.height -= dy
        elif handle_index == 1:  # Top-right
            self.y += dy
            self.width += dx
            self.height -= dy
        elif handle_index == 2:  # Bottom-left
            self.x += dx
            self.width -= dx
            self.height += dy
        elif handle_index == 3:  # Bottom-right
            self.width += dx
            self.height += dy
        
        # Ensure minimum size
        self.width = max(20, self.width)
        self.height = max(20, self.height)
        self.calculate_resize_handles()
    
    def to_dict(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'label': self.label,
            'created_at': self.created_at
        }

class VideoPlayer(QLabel):
    frame_changed = pyqtSignal(int, np.ndarray)
    region_added = pyqtSignal(dict)
    region_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.setScaledContents(False)
        self.setAlignment(Qt.AlignCenter)
        
        # Video properties
        self.cap = None
        self.frame_count = 0
        self.fps = 30
        self.current_frame_index = 0
        self.current_frame = None
        self.original_frame = None
        
        # Image sequence support
        self.image_files = []  # List of image file paths
        self.is_image_sequence = False
        
        # Playback
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        # Regions
        self.regions = []
        self.selected_region = None
        self.drawing_region = False
        self.start_point = None
        self.current_rect = None
        
        # Mouse interaction
        self.dragging = False
        self.resizing = False
        self.resize_handle = -1
        self.last_mouse_pos = None
        
        # Display scaling
        self.scale_factor = 1.0
        self.display_size = (640, 480)
        
        self.setMouseTracking(True)
        
    def load_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                return False
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.is_image_sequence = False
            self.image_files = []
            
            # Load first frame
            self.seek_frame(0)
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def load_images(self, image_files):
        """Load a sequence of images"""
        try:
            if not image_files:
                return False
            
            self.image_files = [str(f) for f in image_files]
            self.frame_count = len(image_files)
            self.fps = 30  # Default FPS for image sequences
            self.is_image_sequence = True
            
            # Release video capture if it exists
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Load first image
            self.seek_frame(0)
            return True
            
        except Exception as e:
            print(f"Error loading images: {e}")
            return False
    
    def seek_frame(self, frame_index):
        frame_index = max(0, min(frame_index, self.frame_count - 1))
        
        if self.is_image_sequence:
            # Load image from file
            try:
                if 0 <= frame_index < len(self.image_files):
                    frame = cv2.imread(self.image_files[frame_index])
                    if frame is not None:
                        self.current_frame_index = frame_index
                        self.original_frame = frame.copy()
                        self.current_frame = frame
                        self.update_display()
                        self.frame_changed.emit(frame_index, frame)
            except Exception as e:
                print(f"Error loading image {self.image_files[frame_index]}: {e}")
        else:
            # Load frame from video
            if not self.cap:
                return
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_index = frame_index
                self.original_frame = frame.copy()
                self.current_frame = frame
                self.update_display()
                self.frame_changed.emit(frame_index, frame)
    
    def play(self):
        if not self.cap:
            return
        
        self.is_playing = True
        interval = int(1000 / self.fps) if self.fps > 0 else 33
        self.timer.start(interval)
    
    def pause(self):
        self.is_playing = False
        self.timer.stop()
    
    def stop(self):
        self.pause()
        if self.cap:
            self.cap.release()
    
    def next_frame(self):
        if self.current_frame_index < self.frame_count - 1:
            self.seek_frame(self.current_frame_index + 1)
        else:
            self.pause()
    
    def previous_frame(self):
        if self.current_frame_index > 0:
            self.seek_frame(self.current_frame_index - 1)
    
    def skip_frames(self, count):
        target_frame = self.current_frame_index + count
        self.seek_frame(target_frame)
    
    def update_display(self):
        if self.current_frame is None:
            return
        
        # Create a copy for drawing
        display_frame = self.current_frame.copy()
        
        # Draw regions
        self.draw_regions(display_frame)
        
        # Convert to Qt format and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit widget
        widget_size = self.size()
        scaled_pixmap = QPixmap.fromImage(q_image).scaled(
            widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Calculate scaling factor for mouse coordinate conversion
        self.scale_factor = min(widget_size.width() / width, widget_size.height() / height)
        self.display_size = (scaled_pixmap.width(), scaled_pixmap.height())
        
        self.setPixmap(scaled_pixmap)
    
    def draw_regions(self, frame):
        for region in self.regions:
            color = (0, 255, 0) if region.selected else (255, 0, 0)
            thickness = 3 if region.selected else 2
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (int(region.x), int(region.y)), 
                         (int(region.x + region.width), int(region.y + region.height)), 
                         color, thickness)
            
            # Draw label
            if region.label:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_size = cv2.getTextSize(region.label, font, font_scale, font_thickness)[0]
                
                # Background for text
                cv2.rectangle(frame,
                            (int(region.x), int(region.y - text_size[1] - 10)),
                            (int(region.x + text_size[0] + 10), int(region.y)),
                            color, -1)
                
                # Text
                cv2.putText(frame, region.label,
                           (int(region.x + 5), int(region.y - 5)),
                           font, font_scale, (255, 255, 255), font_thickness)
            
            # Draw resize handles for selected region
            if region.selected:
                for handle in region.resize_handles:
                    cv2.rectangle(frame,
                                (handle.x(), handle.y()),
                                (handle.x() + handle.width(), handle.y() + handle.height()),
                                (255, 255, 0), -1)
    
    def mousePressEvent(self, event):
        if self.current_frame is None:
            return
        
        # Convert mouse coordinates to image coordinates
        mouse_pos = self.map_to_image_coords(event.pos())
        if mouse_pos is None:
            return
        
        x, y = mouse_pos
        
        if event.button() == Qt.LeftButton:
            # Check if clicking on a region or resize handle
            clicked_region = None
            resize_handle = -1
            
            for region in self.regions:
                handle_idx = region.get_handle_at_point(x, y)
                if handle_idx >= 0 and region.selected:
                    clicked_region = region
                    resize_handle = handle_idx
                    break
                elif region.contains_point(x, y):
                    clicked_region = region
                    break
            
            if clicked_region and resize_handle >= 0:
                # Start resizing
                self.resizing = True
                self.resize_handle = resize_handle
                self.selected_region = clicked_region
                self.last_mouse_pos = (x, y)
            elif clicked_region:
                # Select and start dragging
                self.clear_selection()
                clicked_region.selected = True
                self.selected_region = clicked_region
                self.dragging = True
                self.last_mouse_pos = (x, y)
                self.region_selected.emit(clicked_region.id)
            else:
                # Start drawing new region
                self.clear_selection()
                self.drawing_region = True
                self.start_point = (x, y)
                self.current_rect = QRect(x, y, 0, 0)
        
        elif event.button() == Qt.RightButton:
            # Context menu or label editing could go here
            pass
        
        self.update_display()
    
    def mouseMoveEvent(self, event):
        if self.current_frame is None:
            return
        
        mouse_pos = self.map_to_image_coords(event.pos())
        if mouse_pos is None:
            return
        
        x, y = mouse_pos
        
        if self.drawing_region and self.start_point:
            # Update current rectangle
            start_x, start_y = self.start_point
            self.current_rect = QRect(
                min(start_x, x), min(start_y, y),
                abs(x - start_x), abs(y - start_y)
            )
            
            # Draw temporary rectangle
            temp_frame = self.current_frame.copy()
            cv2.rectangle(temp_frame,
                         (self.current_rect.x(), self.current_rect.y()),
                         (self.current_rect.x() + self.current_rect.width(),
                          self.current_rect.y() + self.current_rect.height()),
                         (0, 255, 255), 2)
            
            # Update display temporarily
            height, width, channel = temp_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(temp_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            widget_size = self.size()
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                widget_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            
        elif self.dragging and self.selected_region and self.last_mouse_pos:
            # Move selected region
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.selected_region.move(dx, dy)
            self.last_mouse_pos = (x, y)
            self.update_display()
            
        elif self.resizing and self.selected_region and self.last_mouse_pos:
            # Resize selected region
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.selected_region.resize(self.resize_handle, dx, dy)
            self.last_mouse_pos = (x, y)
            self.update_display()
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.drawing_region and self.current_rect and self.current_rect.width() > 10 and self.current_rect.height() > 10:
                # Create new region
                region = Region(
                    self.current_rect.x(),
                    self.current_rect.y(),
                    self.current_rect.width(),
                    self.current_rect.height()
                )
                self.regions.append(region)
                region.selected = True
                self.selected_region = region
                
                # Emit signal
                self.region_added.emit(region.to_dict())
                self.region_selected.emit(region.id)
            
            # Reset states
            self.drawing_region = False
            self.dragging = False
            self.resizing = False
            self.start_point = None
            self.current_rect = None
            self.resize_handle = -1
            self.last_mouse_pos = None
            
            self.update_display()
    
    def map_to_image_coords(self, widget_pos):
        if self.current_frame is None:
            return None
        
        # Get widget center offset
        widget_size = self.size()
        display_width, display_height = self.display_size
        
        offset_x = (widget_size.width() - display_width) // 2
        offset_y = (widget_size.height() - display_height) // 2
        
        # Adjust for offset
        adj_x = widget_pos.x() - offset_x
        adj_y = widget_pos.y() - offset_y
        
        # Check if point is within display area
        if adj_x < 0 or adj_y < 0 or adj_x >= display_width or adj_y >= display_height:
            return None
        
        # Scale to original image coordinates
        frame_height, frame_width = self.current_frame.shape[:2]
        img_x = int(adj_x / self.scale_factor)
        img_y = int(adj_y / self.scale_factor)
        
        return (img_x, img_y)
    
    def clear_selection(self):
        for region in self.regions:
            region.selected = False
        self.selected_region = None
        self.region_selected.emit("")
    
    def clear_regions(self):
        self.regions.clear()
        self.selected_region = None
        self.update_display()
    
    def delete_region(self, region_id):
        self.regions = [r for r in self.regions if r.id != region_id]
        if self.selected_region and self.selected_region.id == region_id:
            self.selected_region = None
        self.update_display()
    
    def get_selected_region(self):
        return self.selected_region.id if self.selected_region else None
    
    def select_all_regions(self):
        for region in self.regions:
            region.selected = True
        self.update_display()
    
    def add_region_from_data(self, region_data):
        region = Region(
            region_data['x'],
            region_data['y'],
            region_data['width'],
            region_data['height'],
            region_data.get('label', ''),
            region_data.get('id')
        )
        self.regions.append(region)
        self.update_display()
    
    def get_all_regions_data(self):
        return [region.to_dict() for region in self.regions]
    
    def get_region_data(self, region_id):
        """Get region data by ID"""
        for region in self.regions:
            if region.id == region_id:
                return region.to_dict()
        return None
    
    def set_region_label(self, region_id, label):
        for region in self.regions:
            if region.id == region_id:
                region.label = label
                self.update_display()
                break 