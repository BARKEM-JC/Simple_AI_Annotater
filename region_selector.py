"""
Region selection and manipulation utilities
"""

from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor
import cv2
import numpy as np

class RegionUtils:
    @staticmethod
    def normalize_rect(x1, y1, x2, y2):
        """Normalize rectangle coordinates to ensure proper bounds"""
        return (
            min(x1, x2),
            min(y1, y2),
            abs(x2 - x1),
            abs(y2 - y1)
        )
    
    @staticmethod
    def rect_intersection(rect1, rect2):
        """Calculate intersection of two rectangles"""
        x1 = max(rect1['x'], rect2['x'])
        y1 = max(rect1['y'], rect2['y'])
        x2 = min(rect1['x'] + rect1['width'], rect2['x'] + rect2['width'])
        y2 = min(rect1['y'] + rect1['height'], rect2['y'] + rect2['height'])
        
        if x1 < x2 and y1 < y2:
            return {
                'x': x1,
                'y': y1,
                'width': x2 - x1,
                'height': y2 - y1
            }
        return None
    
    @staticmethod
    def rect_area(rect):
        """Calculate rectangle area"""
        return rect['width'] * rect['height']
    
    @staticmethod
    def point_in_rect(point, rect):
        """Check if point is inside rectangle"""
        x, y = point
        return (rect['x'] <= x <= rect['x'] + rect['width'] and
                rect['y'] <= y <= rect['y'] + rect['height'])
    
    @staticmethod
    def expand_rect(rect, margin):
        """Expand rectangle by margin on all sides"""
        return {
            'x': max(0, rect['x'] - margin),
            'y': max(0, rect['y'] - margin),
            'width': rect['width'] + 2 * margin,
            'height': rect['height'] + 2 * margin
        }
    
    @staticmethod
    def crop_frame(frame, rect):
        """Crop frame using rectangle coordinates"""
        x, y, w, h = rect['x'], rect['y'], rect['width'], rect['height']
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        return frame[y:y+h, x:x+w]
    
    @staticmethod
    def draw_crosshair(frame, center, size=20, color=(0, 255, 0), thickness=2):
        """Draw crosshair at specified center point"""
        x, y = center
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    
    @staticmethod
    def draw_grid(frame, spacing=50, color=(100, 100, 100), thickness=1):
        """Draw grid overlay on frame"""
        h, w = frame.shape[:2]
        
        # Vertical lines
        for x in range(0, w, spacing):
            cv2.line(frame, (x, 0), (x, h), color, thickness)
        
        # Horizontal lines
        for y in range(0, h, spacing):
            cv2.line(frame, (0, y), (w, y), color, thickness)
    
    @staticmethod
    def calculate_iou(rect1, rect2):
        """Calculate Intersection over Union (IoU) of two rectangles"""
        intersection = RegionUtils.rect_intersection(rect1, rect2)
        if intersection is None:
            return 0.0
        
        intersection_area = RegionUtils.rect_area(intersection)
        union_area = (RegionUtils.rect_area(rect1) + 
                     RegionUtils.rect_area(rect2) - 
                     intersection_area)
        
        return intersection_area / union_area if union_area > 0 else 0.0

class SelectionMode:
    """Enumeration for different selection modes"""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    POLYGON = "polygon"
    FREEHAND = "freehand"

class RegionSelector:
    """Main class for handling region selection operations"""
    
    def __init__(self):
        self.selection_mode = SelectionMode.RECTANGLE
        self.min_region_size = 10
        self.snap_to_grid = False
        self.grid_size = 10
        
    def set_selection_mode(self, mode):
        """Set the current selection mode"""
        if mode in [SelectionMode.RECTANGLE, SelectionMode.CIRCLE, 
                   SelectionMode.POLYGON, SelectionMode.FREEHAND]:
            self.selection_mode = mode
    
    def validate_region(self, region):
        """Validate region meets minimum requirements"""
        return (region['width'] >= self.min_region_size and 
                region['height'] >= self.min_region_size)
    
    def snap_point(self, point):
        """Snap point to grid if enabled"""
        if not self.snap_to_grid:
            return point
        
        x, y = point
        snapped_x = round(x / self.grid_size) * self.grid_size
        snapped_y = round(y / self.grid_size) * self.grid_size
        return (snapped_x, snapped_y)
    
    def create_region_from_points(self, start_point, end_point):
        """Create region from two points based on current selection mode"""
        start_point = self.snap_point(start_point)
        end_point = self.snap_point(end_point)
        
        if self.selection_mode == SelectionMode.RECTANGLE:
            x, y, w, h = RegionUtils.normalize_rect(
                start_point[0], start_point[1],
                end_point[0], end_point[1]
            )
            
            region = {
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'type': 'rectangle'
            }
            
            return region if self.validate_region(region) else None
        
        # Add other selection modes here as needed
        return None 