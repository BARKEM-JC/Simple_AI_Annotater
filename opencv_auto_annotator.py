"""
OpenCV-based Auto Annotation Module
Provides template matching and other OpenCV-based automatic annotation features
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import uuid
from dataclasses import dataclass
import json
import os
import base64

@dataclass
class TemplateMatch:
    region: Dict[str, int]  # {'x': int, 'y': int, 'width': int, 'height': int}
    confidence: float
    label: str
    template_id: str

class OpenCVAutoAnnotator:
    def __init__(self, templates_file: str = "opencv_templates.json"):
        self.templates = {}  # template_id -> {'template': np.array, 'label': str, 'threshold': float}
        self.min_confidence = 0.7
        self.max_matches_per_template = 10
        self.templates_file = templates_file
        
        # Load existing templates on startup
        self.load_templates()
        
    def save_templates(self) -> bool:
        """
        Save all templates to disk for persistence across sessions
        Returns True if successful, False otherwise
        """
        try:
            templates_data = {}
            for template_id, template_info in self.templates.items():
                # Convert numpy array to base64 string for JSON serialization
                _, buffer = cv2.imencode('.png', template_info['template'])
                template_b64 = base64.b64encode(buffer).decode('utf-8')
                
                templates_data[template_id] = {
                    'label': template_info['label'],
                    'threshold': template_info['threshold'],
                    'template_data': template_b64,
                    'original_region': template_info.get('original_region', {}),
                    'created_timestamp': template_info.get('created_timestamp', ''),
                    'usage_count': template_info.get('usage_count', 0)
                }
            
            with open(self.templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving templates: {e}")
            return False
    
    def load_templates(self) -> bool:
        """
        Load templates from disk
        Returns True if successful, False otherwise
        """
        if not os.path.exists(self.templates_file):
            return True  # No templates file exists yet, which is fine
        
        try:
            with open(self.templates_file, 'r') as f:
                templates_data = json.load(f)
            
            self.templates = {}
            for template_id, template_info in templates_data.items():
                # Convert base64 string back to numpy array
                template_b64 = template_info['template_data']
                buffer = base64.b64decode(template_b64)
                template_array = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
                
                self.templates[template_id] = {
                    'template': template_array,
                    'label': template_info['label'],
                    'threshold': template_info['threshold'],
                    'original_region': template_info.get('original_region', {}),
                    'created_timestamp': template_info.get('created_timestamp', ''),
                    'usage_count': template_info.get('usage_count', 0)
                }
            
            return True
        except Exception as e:
            print(f"Error loading templates: {e}")
            return False
    
    def get_templates_file_path(self) -> str:
        """Return the path to the templates file"""
        return os.path.abspath(self.templates_file)
    
    def export_templates(self, export_path: str) -> bool:
        """
        Export templates to a specified file path
        """
        try:
            templates_data = {}
            for template_id, template_info in self.templates.items():
                # Convert numpy array to base64 string for JSON serialization
                _, buffer = cv2.imencode('.png', template_info['template'])
                template_b64 = base64.b64encode(buffer).decode('utf-8')
                
                templates_data[template_id] = {
                    'label': template_info['label'],
                    'threshold': template_info['threshold'],
                    'template_data': template_b64,
                    'original_region': template_info.get('original_region', {}),
                    'created_timestamp': template_info.get('created_timestamp', ''),
                    'usage_count': template_info.get('usage_count', 0)
                }
            
            with open(export_path, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error exporting templates: {e}")
            return False
    
    def import_templates(self, import_path: str, merge: bool = True) -> bool:
        """
        Import templates from a specified file path
        Args:
            import_path: Path to the templates JSON file
            merge: If True, merge with existing templates. If False, replace all templates.
        """
        try:
            with open(import_path, 'r') as f:
                templates_data = json.load(f)
            
            if not merge:
                self.templates = {}
            
            for template_id, template_info in templates_data.items():
                # Convert base64 string back to numpy array
                template_b64 = template_info['template_data']
                buffer = base64.b64decode(template_b64)
                template_array = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
                
                # Generate new ID if template already exists (to avoid conflicts)
                final_template_id = template_id
                if final_template_id in self.templates:
                    final_template_id = str(uuid.uuid4())
                
                self.templates[final_template_id] = {
                    'template': template_array,
                    'label': template_info['label'],
                    'threshold': template_info['threshold'],
                    'original_region': template_info.get('original_region', {}),
                    'created_timestamp': template_info.get('created_timestamp', ''),
                    'usage_count': template_info.get('usage_count', 0)
                }
            
            # Save the updated templates
            self.save_templates()
            return True
        except Exception as e:
            print(f"Error importing templates: {e}")
            return False

    def add_template_from_region(self, frame: np.ndarray, region: Dict[str, int], label: str) -> str:
        """
        Extract a template from a user-annotated region
        
        Args:
            frame: Current frame
            region: Region dictionary with x, y, width, height
            label: Label for this template
            
        Returns:
            template_id: Unique identifier for the template
        """
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        
        # Ensure coordinates are within frame bounds
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        # Extract template
        template = frame[y:y+h, x:x+w].copy()
        
        # Generate unique template ID
        template_id = str(uuid.uuid4())
        
        # Store template with metadata
        from datetime import datetime
        self.templates[template_id] = {
            'template': template,
            'label': label,
            'threshold': self.min_confidence,
            'original_region': region.copy(),
            'created_timestamp': datetime.now().isoformat(),
            'usage_count': 0
        }
        
        # Save templates to disk
        self.save_templates()
        
        return template_id
    
    def remove_template(self, template_id: str) -> bool:
        """Remove a template by ID"""
        if template_id in self.templates:
            del self.templates[template_id]
            # Save templates to disk after removal
            self.save_templates()
            return True
        return False
    
    def get_template_info(self, template_id: str) -> Optional[Dict]:
        """Get information about a template"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[Dict]:
        """List all templates with their info"""
        result = []
        for template_id, template_data in self.templates.items():
            result.append({
                'id': template_id,
                'label': template_data['label'],
                'threshold': template_data['threshold'],
                'size': template_data['template'].shape[:2],
                'created_timestamp': template_data.get('created_timestamp', ''),
                'usage_count': template_data.get('usage_count', 0)
            })
        return result
    
    def set_template_threshold(self, template_id: str, threshold: float) -> bool:
        """Set the matching threshold for a specific template"""
        if template_id in self.templates:
            self.templates[template_id]['threshold'] = threshold
            # Save templates to disk after threshold change
            self.save_templates()
            return True
        return False
    
    def find_matches(self, frame: np.ndarray, template_ids: Optional[List[str]] = None, 
                    existing_annotations: Optional[List[Dict]] = None) -> List[TemplateMatch]:
        """
        Find all template matches in the frame
        
        Args:
            frame: Frame to search in
            template_ids: Optional list of specific template IDs to use
            existing_annotations: Optional list of existing annotations to avoid duplicates
            
        Returns:
            List of TemplateMatch objects
        """
        matches = []
        
        templates_to_use = template_ids if template_ids else list(self.templates.keys())
        
        for template_id in templates_to_use:
            if template_id not in self.templates:
                continue
                
            template_info = self.templates[template_id]
            template_matches = self._match_template(
                frame, 
                template_info['template'], 
                template_info['threshold']
            )
            
            # Add template info to matches
            for match in template_matches:
                template_match = TemplateMatch(
                    region=match['region'],
                    confidence=match['confidence'],
                    label=template_info['label'],
                    template_id=template_id
                )
                matches.append(template_match)
        
        # Remove overlapping matches - keep only the best one for each overlapping group
        matches = self._filter_overlapping_matches(matches, overlap_threshold=0.25)
        
        # Filter out matches that overlap significantly with existing annotations
        if existing_annotations:
            matches = self._filter_existing_overlaps(matches, existing_annotations)
        
        return matches
    
    def _filter_overlapping_matches(self, matches: List[TemplateMatch], overlap_threshold: float = 0.25) -> List[TemplateMatch]:
        """
        Filter overlapping matches to keep only the best one from each overlapping group
        
        Args:
            matches: List of template matches
            overlap_threshold: Threshold for considering matches as overlapping (0.25 = 25%)
            
        Returns:
            Filtered list of matches
        """
        if not matches:
            return matches
        
        # Sort matches by confidence (descending)
        sorted_matches = sorted(matches, key=lambda x: x.confidence, reverse=True)
        filtered_matches = []
        
        for current_match in sorted_matches:
            # Check if this match overlaps significantly with any already accepted match
            overlaps = False
            for accepted_match in filtered_matches:
                if self._regions_overlap(current_match.region, accepted_match.region, overlap_threshold):
                    # Check which one is bigger (by area) and has higher confidence
                    current_area = current_match.region['width'] * current_match.region['height']
                    accepted_area = accepted_match.region['width'] * accepted_match.region['height']
                    
                    # If current match is significantly larger or has much higher confidence, replace the accepted match
                    area_ratio = current_area / accepted_area if accepted_area > 0 else 1
                    confidence_diff = current_match.confidence - accepted_match.confidence
                    
                    if area_ratio > 1.5 or (area_ratio > 1.1 and confidence_diff > 0.1):
                        # Replace the accepted match with the current one
                        filtered_matches.remove(accepted_match)
                        break
                    else:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_matches.append(current_match)
        
        return filtered_matches
    
    def _filter_existing_overlaps(self, matches: List[TemplateMatch], existing_annotations: List[Dict]) -> List[TemplateMatch]:
        """
        Filter out matches that overlap with existing annotations
        
        Args:
            matches: List of template matches
            existing_annotations: List of existing annotation regions
            
        Returns:
            Filtered list of matches that don't overlap with existing annotations
        """
        filtered_matches = []
        
        for match in matches:
            overlaps_existing = False
            
            for existing_annotation in existing_annotations:
                # Convert existing annotation to region format if needed
                if 'region' in existing_annotation:
                    existing_region = existing_annotation['region']
                else:
                    existing_region = {
                        'x': existing_annotation.get('x', 0),
                        'y': existing_annotation.get('y', 0),
                        'width': existing_annotation.get('width', 0),
                        'height': existing_annotation.get('height', 0)
                    }
                
                # Check for overlap
                if self._regions_overlap(match.region, existing_region, overlap_threshold=0.25):
                    overlaps_existing = True
                    break
            
            if not overlaps_existing:
                filtered_matches.append(match)
        
        return filtered_matches
    
    def _match_template(self, frame: np.ndarray, template: np.ndarray, threshold: float) -> List[Dict]:
        """
        Perform template matching using optimized methods to reduce CPU usage
        """
        matches = []
        
        # Convert to grayscale for better matching and performance
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        template_h, template_w = template_gray.shape
        
        # Skip if template is larger than frame
        if template_h >= frame_gray.shape[0] or template_w >= frame_gray.shape[1]:
            return matches
        
        # Optimize for performance: resize frame if it's very large
        frame_h, frame_w = frame_gray.shape
        scale_factor = 1.0
        if frame_w > 1920 or frame_h > 1080:
            # Scale down large frames to improve performance
            scale_factor = min(1920 / frame_w, 1080 / frame_h)
            new_w = int(frame_w * scale_factor)
            new_h = int(frame_h * scale_factor)
            frame_gray = cv2.resize(frame_gray, (new_w, new_h))
            template_gray = cv2.resize(template_gray, 
                                     (int(template_w * scale_factor), int(template_h * scale_factor)))
            template_h, template_w = template_gray.shape
        
        # Use normalized cross correlation with reduced precision for speed
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find peaks more efficiently
        locations = np.where(result >= threshold)
        
        # Limit the number of locations to process to prevent CPU overload
        max_locations = 50  # Reasonable limit
        if len(locations[0]) > max_locations:
            # Sort by confidence and take top matches
            confidences = result[locations]
            top_indices = np.argsort(confidences)[-max_locations:]
            locations = (locations[0][top_indices], locations[1][top_indices])
        
        processed_matches = []
        for pt in zip(*locations[::-1]):  # Switch x and y
            confidence = result[pt[1], pt[0]]
            
            # Scale coordinates back if we resized the frame
            x = int(pt[0] / scale_factor)
            y = int(pt[1] / scale_factor)
            w = int(template_w / scale_factor)
            h = int(template_h / scale_factor)
            
            region = {
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }
            
            # Check for overlapping matches and keep only the best
            overlaps = False
            for existing_match in processed_matches:
                if self._regions_overlap(region, existing_match['region'], overlap_threshold=0.3):
                    # If new match has higher confidence, replace the old one
                    if confidence > existing_match['confidence']:
                        processed_matches.remove(existing_match)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                processed_matches.append({
                    'region': region,
                    'confidence': float(confidence)
                })
        
        # Sort by confidence and limit results to prevent overwhelming the UI
        processed_matches.sort(key=lambda x: x['confidence'], reverse=True)
        return processed_matches[:min(self.max_matches_per_template, 10)]  # Further limit results
    
    def _regions_overlap(self, region1: Dict[str, int], region2: Dict[str, int], 
                        overlap_threshold: float = 0.3) -> bool:
        """Check if two regions overlap significantly"""
        x1, y1, w1, h1 = region1['x'], region1['y'], region1['width'], region1['height']
        x2, y2, w2, h2 = region2['x'], region2['y'], region2['width'], region2['height']
        
        # Calculate intersection
        ix = max(x1, x2)
        iy = max(y1, y2)
        iw = min(x1 + w1, x2 + w2) - ix
        ih = min(y1 + h1, y2 + h2) - iy
        
        if iw <= 0 or ih <= 0:
            return False
        
        intersection_area = iw * ih
        union_area = w1 * h1 + w2 * h2 - intersection_area
        
        return (intersection_area / union_area) > overlap_threshold
    
    def find_similar_regions(self, frame: np.ndarray, reference_region: Dict[str, int], 
                           similarity_threshold: float = 0.8) -> List[Dict]:
        """
        Find regions similar to a reference region using feature matching
        """
        # Extract reference region
        x, y, w, h = reference_region['x'], reference_region['y'], reference_region['width'], reference_region['height']
        ref_roi = frame[y:y+h, x:x+w]
        
        # Convert to grayscale
        ref_gray = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2GRAY) if len(ref_roi.shape) == 3 else ref_roi
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Initialize feature detector (SIFT or ORB)
        detector = cv2.SIFT_create() if hasattr(cv2, 'SIFT_create') else cv2.ORB_create()
        
        # Find keypoints and descriptors in reference region
        kp1, des1 = detector.detectAndCompute(ref_gray, None)
        
        if des1 is None or len(des1) < 4:
            return []
        
        # Find keypoints and descriptors in full frame
        kp2, des2 = detector.detectAndCompute(frame_gray, None)
        
        if des2 is None or len(des2) < 4:
            return []
        
        # Match features
        if hasattr(cv2, 'SIFT_create'):
            # Use FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
        else:
            # Use BruteForce matcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter matches based on distance threshold
            max_distance = min(50, matches[0].distance * 3) if matches else 50
            good_matches = [m for m in matches if m.distance <= max_distance]
        
        # Filter matches based on similarity threshold
        if len(good_matches) < 4:
            return []
        
        # Get coordinates of matched keypoints in the full frame
        matched_points = []
        for match in good_matches:
            # Get keypoint from full frame (accounting for reference region offset)
            frame_kp = kp2[match.trainIdx]
            matched_points.append((frame_kp.pt[0], frame_kp.pt[1]))
        
        # Apply additional similarity threshold filter
        similarity_score = len(good_matches) / max(len(kp1), 1)
        if similarity_score < similarity_threshold:
            return []
        
        # Cluster matched points to find similar regions
        similar_regions = self._cluster_points_to_regions(matched_points, w, h, frame.shape)
        
        return similar_regions

    def _cluster_points_to_regions(self, points: List[Tuple[float, float]], 
                                  ref_width: int, ref_height: int, 
                                  frame_shape: Tuple[int, int]) -> List[Dict]:
        """
        Cluster matched points to find coherent regions similar to reference
        
        Args:
            points: List of (x, y) coordinates of matched keypoints
            ref_width: Width of reference region
            ref_height: Height of reference region  
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            List of region dictionaries
        """
        if len(points) < 3:
            return []
        
        regions = []
        frame_h, frame_w = frame_shape[:2]
        
        # Simple clustering approach: group points that are close together
        used_points = set()
        cluster_radius = max(ref_width, ref_height) * 0.7  # Allow some overlap
        
        for i, (x1, y1) in enumerate(points):
            if i in used_points:
                continue
            
            # Find nearby points to form a cluster
            cluster_points = [(x1, y1)]
            used_points.add(i)
            
            for j, (x2, y2) in enumerate(points):
                if j in used_points:
                    continue
                
                # Check if point is within cluster radius
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance <= cluster_radius:
                    cluster_points.append((x2, y2))
                    used_points.add(j)
            
            # Need at least 3 points to form a meaningful region
            if len(cluster_points) >= 3:
                # Calculate bounding box for this cluster
                xs = [p[0] for p in cluster_points]
                ys = [p[1] for p in cluster_points]
                
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                
                # Expand bounding box to approximate size of reference region
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                # Use reference region size but allow some variation
                half_w = ref_width // 2
                half_h = ref_height // 2
                
                region_x = max(0, int(center_x - half_w))
                region_y = max(0, int(center_y - half_h))
                region_w = min(ref_width, frame_w - region_x)
                region_h = min(ref_height, frame_h - region_y)
                
                # Only add region if it's reasonably sized
                if region_w >= ref_width * 0.5 and region_h >= ref_height * 0.5:
                    region = {
                        'x': region_x,
                        'y': region_y,
                        'width': region_w,
                        'height': region_h,
                        'confidence': len(cluster_points) / len(points),  # Confidence based on point density
                        'match_count': len(cluster_points)
                    }
                    regions.append(region)
        
        # Sort regions by confidence and return
        regions.sort(key=lambda r: r['confidence'], reverse=True)
        return regions 