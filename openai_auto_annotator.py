"""
OpenAI Vision API Auto Annotation Module
Provides AI-powered automatic annotation using OpenAI's vision models
"""

import cv2
import numpy as np
import base64
from typing import List, Dict, Optional, Union
import json
from dataclasses import dataclass
from openai import OpenAI
import os
from pathlib import Path

@dataclass
class AIAnnotation:
    region: Dict[str, int]  # {'x': int, 'y': int, 'width': int, 'height': int}
    label: str
    confidence: float
    description: str
    category: str  # e.g., 'object', 'person', 'vehicle', etc.

class OpenAIAutoAnnotator:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI Auto Annotator
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4.1-nano"
        self.max_tokens = 4096
        
        # Predefined categories for better organization
        self.categories = [
            #"person", "vehicle", "animal", "object", "building", 
            #"furniture", "electronics", "food", "clothing", "nature"
        ]
        
    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Encode to JPEG bytes
        success, buffer = cv2.imencode('.jpg', image_rgb)
        if not success:
            raise ValueError("Failed to encode image")
            
        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def create_annotation_prompt(self, custom_labels: List[str] = None) -> str:
        """Create a structured prompt for annotation"""
        labels_section = ""
        if custom_labels:
            labels_section = f"\nPreferred labels to use when applicable: {', '.join(custom_labels)}"
        
        prompt = f"""
        Analyze this image and identify all objects, people, and notable features. For each item you detect, provide:
        
        1. A bounding box with coordinates (x, y, width, height) as percentages of image dimensions (0-100)
        2. A descriptive label 
        3. A confidence score (0-100)
        4. A brief description
        5. A category from: {', '.join(self.categories)}
        
        {labels_section}
        
        Please respond with a JSON object with this exact structure:
        {{
            "annotations": [
                {{
                    "region": {{"x": 10, "y": 20, "width": 30, "height": 40}},
                    "label": "person",
                    "confidence": 85,
                    "description": "Person walking in the center of the image",
                    "category": "person"
                }}
            ],
            "image_description": "Overall description of the image scene"
        }}
        
        Important: 
        - Coordinates should be percentages (0-100) of the image dimensions
        - Only include objects you can clearly identify
        - Provide accurate bounding boxes that tightly fit the objects
        - Use descriptive but concise labels
        """
        return prompt
    
    def annotate_frame(self, frame: np.ndarray, custom_labels: List[str] = None) -> Dict:
        """
        Send frame to OpenAI Vision API for annotation
        
        Args:
            frame: Input frame as numpy array
            custom_labels: Optional list of preferred labels to use
            
        Returns:
            Dictionary containing annotations and metadata
        """
        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(frame)
            
            # Create prompt
            prompt = self.create_annotation_prompt(custom_labels)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=0.1  # Low temperature for more consistent results
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            result = self._parse_ai_response(response_text, frame.shape)
            
            return {
                'success': True,
                'annotations': result.get('annotations', []),
                'image_description': result.get('image_description', ''),
                'raw_response': response_text
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'annotations': [],
                'image_description': '',
                'raw_response': ''
            }
    
    def _parse_ai_response(self, response_text: str, frame_shape: tuple) -> Dict:
        """Parse AI response and convert percentage coordinates to pixels"""
        height, width = frame_shape[:2]
        
        try:
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[json_start:json_end]
            parsed_data = json.loads(json_str)
            
            # Convert percentage coordinates to pixels
            for annotation in parsed_data.get('annotations', []):
                region = annotation.get('region', {})
                
                # Convert percentage to pixels
                region['x'] = int((region.get('x', 0) / 100.0) * width)
                region['y'] = int((region.get('y', 0) / 100.0) * height)
                region['width'] = int((region.get('width', 0) / 100.0) * width)
                region['height'] = int((region.get('height', 0) / 100.0) * height)
                
                # Ensure coordinates are within bounds
                region['x'] = max(0, min(region['x'], width - 1))
                region['y'] = max(0, min(region['y'], height - 1))
                region['width'] = max(1, min(region['width'], width - region['x']))
                region['height'] = max(1, min(region['height'], height - region['y']))
                
                annotation['region'] = region
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract information manually
            return self._fallback_parse(response_text, frame_shape)
    
    def _fallback_parse(self, response_text: str, frame_shape: tuple) -> Dict:
        """Fallback parsing when JSON parsing fails"""
        # This is a simplified fallback - in a real implementation,
        # you might want more sophisticated text parsing
        return {
            'annotations': [],
            'image_description': response_text[:200] + "..." if len(response_text) > 200 else response_text
        }
    
    def convert_to_annotation_objects(self, api_result: Dict) -> List[AIAnnotation]:
        """Convert API result to AIAnnotation objects"""
        annotations = []
        
        for annotation_data in api_result.get('annotations', []):
            try:
                annotation = AIAnnotation(
                    region=annotation_data.get('region', {}),
                    label=annotation_data.get('label', 'unknown'),
                    confidence=annotation_data.get('confidence', 0) / 100.0,  # Convert to 0-1 scale
                    description=annotation_data.get('description', ''),
                    category=annotation_data.get('category', 'object')
                )
                annotations.append(annotation)
            except Exception as e:
                print(f"Error creating annotation object: {e}")
                continue
        
        return annotations
    
    def batch_annotate_frames(self, frames: List[np.ndarray], 
                            custom_labels: List[str] = None) -> List[Dict]:
        """
        Annotate multiple frames (for batch processing)
        Note: This makes multiple API calls, so be mindful of rate limits
        """
        results = []
        
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}...")
            result = self.annotate_frame(frame, custom_labels)
            results.append(result)
            
            # Add a small delay to respect rate limits
            import time
            time.sleep(0.5)
        
        return results
    
    def save_annotations_to_file(self, annotations: List[Dict], 
                               output_path: Union[str, Path]) -> bool:
        """Save annotations to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving annotations: {e}")
            return False
    
    def load_annotations_from_file(self, input_path: Union[str, Path]) -> List[Dict]:
        """Load annotations from JSON file"""
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return [] 