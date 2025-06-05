"""
Synthetic Dataset Generator
Uses OpenAI's GPT-4-vision and DALL-E to generate synthetic training data
"""

import os
import cv2
import base64
import json
import uuid
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
import io

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from region_selector import RegionUtils

class SyntheticDataGenerator:
    def __init__(self, output_dir="labeled_data/synthetic_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.annotations_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # OpenAI client
        self.client = None
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment")
        elif OpenAI:
            self.client = OpenAI(api_key=self.api_key)
        else:
            print("Warning: openai package not installed")
    
    def is_available(self):
        """Check if synthetic generation is available"""
        return self.client is not None and self.api_key is not None
    
    def encode_image_to_base64(self, image):
        """Convert CV2 image to base64 string"""
        # Convert BGR to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def analyze_image_content(self, image, region=None):
        """Use GPT-4-vision to analyze image content"""
        if not self.client:
            return "Unable to analyze image content - OpenAI client not available"
        
        try:
            # Crop image to region if specified
            if region:
                cropped_image = RegionUtils.crop_frame(image, region)
            else:
                cropped_image = image
            
            # Convert to base64
            base64_image = self.encode_image_to_base64(cropped_image)
            
            # Analyze with GPT-4-vision
            response = self.client.chat.completions.create(
                model="gpt-image-1",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in detail, focusing on objects, scene, lighting, and composition. This will be used to generate similar synthetic images."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def generate_synthetic_prompt(self, base_description, user_description, region_info=None):
        """Generate a comprehensive prompt for synthetic data generation"""
        prompt_parts = []
        
        # Base description from image analysis
        if base_description and not base_description.startswith("Error"):
            prompt_parts.append(f"Based on this reference: {base_description}")
        
        # User's custom description
        if user_description:
            prompt_parts.append(f"User requirements: {user_description}")
        
        # Region-specific instructions
        if region_info:
            prompt_parts.append(f"Focus on the {region_info} area of the image")
        
        # Style and quality instructions
        prompt_parts.append("Create a high-quality, realistic image suitable for training computer vision models")
        prompt_parts.append("Maintain good lighting and clear details")
        
        return ". ".join(prompt_parts)
    
    def generate_synthetic_image(self, prompt, size="1024x1024"):
        """Generate synthetic image using DALL-E"""
        if not self.client:
            raise Exception("OpenAI client not available")
        
        try:
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality="standard",
                n=1,
            )
            
            # Get the image URL
            image_url = response.data[0].url
            
            # Download and convert to OpenCV format
            import requests
            response = requests.get(image_url)
            if response.status_code == 200:
                # Convert to numpy array
                nparr = np.frombuffer(response.content, np.uint8)
                # Decode image
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
            else:
                raise Exception(f"Failed to download generated image: {response.status_code}")
                
        except Exception as e:
            print(f"Error generating synthetic image: {e}")
            raise e
    
    def create_synthetic_dataset(self, source_image, user_description, num_images=5, 
                               region=None, original_regions=None, progress_callback=None):
        """Create a synthetic dataset based on source image and description"""
        
        if not self.is_available():
            raise Exception("Synthetic data generation not available - check OpenAI API key and installation")
        
        generated_data = []
        
        try:
            # Analyze the source image
            print("Analyzing source image...")
            base_description = self.analyze_image_content(source_image, region)
            
            # Generate synthetic prompt
            region_info = "selected region" if region else "entire image"
            full_prompt = self.generate_synthetic_prompt(base_description, user_description, region_info)
            
            print(f"Generated prompt: {full_prompt}")
            
            # Generate synthetic images
            for i in range(num_images):
                try:
                    print(f"Generating synthetic image {i+1}/{num_images}...")
                    
                    # Add variation to prompt for each image
                    varied_prompt = f"{full_prompt}. Variation {i+1}: slightly different angle, lighting, or composition"
                    
                    # Generate image
                    synthetic_image = self.generate_synthetic_image(varied_prompt)
                    
                    if synthetic_image is not None:
                        # Create unique filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_id = f"synthetic_{timestamp}_{uuid.uuid4().hex[:8]}_{i+1}"
                        image_filename = f"{image_id}.jpg"
                        
                        # Save synthetic image
                        image_path = self.images_dir / image_filename
                        cv2.imwrite(str(image_path), synthetic_image)
                        
                        # Create annotation template (if original regions exist)
                        annotation_data = {
                            'frame_index': i,
                            'image_id': image_id,
                            'image_path': str(image_path),
                            'source_type': 'synthetic',
                            'generation_prompt': varied_prompt,
                            'user_description': user_description,
                            'base_description': base_description,
                            'created_at': datetime.now().isoformat(),
                            'regions': []
                        }
                        
                        # If we have original regions and were generating from a region, 
                        # create template regions for annotation
                        if original_regions and region:
                            # Scale regions to synthetic image size
                            synth_h, synth_w = synthetic_image.shape[:2]
                            orig_h, orig_w = source_image.shape[:2]
                            
                            scale_x = synth_w / orig_w
                            scale_y = synth_h / orig_h
                            
                            for orig_region in original_regions:
                                scaled_region = {
                                    'id': str(uuid.uuid4()),
                                    'x': int(orig_region['x'] * scale_x),
                                    'y': int(orig_region['y'] * scale_y),
                                    'width': int(orig_region['width'] * scale_x),
                                    'height': int(orig_region['height'] * scale_y),
                                    'label': orig_region.get('label', ''),
                                    'confidence': 0.8,  # Template confidence
                                    'needs_review': True
                                }
                                annotation_data['regions'].append(scaled_region)
                        
                        # Save annotation
                        annotation_filename = f"{image_id}.json"
                        annotation_path = self.annotations_dir / annotation_filename
                        with open(annotation_path, 'w') as f:
                            json.dump(annotation_data, f, indent=2)
                        
                        # Save metadata
                        metadata = {
                            'image_id': image_id,
                            'generation_method': 'dall-e-3',
                            'source_region': region,
                            'prompt': varied_prompt,
                            'user_description': user_description,
                            'created_at': datetime.now().isoformat()
                        }
                        
                        metadata_filename = f"{image_id}_meta.json"
                        metadata_path = self.metadata_dir / metadata_filename
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2)
                        
                        generated_data.append({
                            'image_path': str(image_path),
                            'annotation_path': str(annotation_path),
                            'metadata_path': str(metadata_path),
                            'image_data': annotation_data
                        })
                        
                        print(f"Successfully generated synthetic image {i+1}")
                        
                    if progress_callback:
                        progress_callback((i + 1) * 100 // num_images)
                        
                except Exception as e:
                    print(f"Failed to generate image {i+1}: {e}")
                    continue
            
            print(f"Successfully generated {len(generated_data)} synthetic images")
            return generated_data
            
        except Exception as e:
            print(f"Error creating synthetic dataset: {e}")
            raise e
    
    def get_generated_images(self):
        """Get list of all generated synthetic images"""
        generated_images = []
        
        for annotation_file in self.annotations_dir.glob("*.json"):
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    if data.get('source_type') == 'synthetic':
                        generated_images.append(data)
            except Exception as e:
                print(f"Error reading annotation file {annotation_file}: {e}")
        
        return generated_images
    
    def load_synthetic_image(self, image_path):
        """Load a synthetic image"""
        try:
            return cv2.imread(str(image_path))
        except Exception as e:
            print(f"Error loading synthetic image {image_path}: {e}")
            return None
    
    def delete_synthetic_data(self, image_id):
        """Delete synthetic data by image ID"""
        try:
            # Delete image
            image_path = self.images_dir / f"{image_id}.jpg"
            if image_path.exists():
                image_path.unlink()
            
            # Delete annotation
            annotation_path = self.annotations_dir / f"{image_id}.json"
            if annotation_path.exists():
                annotation_path.unlink()
            
            # Delete metadata
            metadata_path = self.metadata_dir / f"{image_id}_meta.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"Error deleting synthetic data {image_id}: {e}")
            return False 