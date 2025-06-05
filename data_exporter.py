import json
import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil
import random

class DataExporter:
    """Export annotated data to various formats for AI training"""
    
    def __init__(self, data_dir, format_type, source_name=None):
        self.data_dir = Path(data_dir)
        self.format_type = format_type
        self.source_name = source_name or "dataset"
        self.export_dir = self.data_dir / f"export_{format_type.lower().replace(' ', '_')}"
        
    def export_dataset(self, source_path, source_type='video', augmentation_options=None, progress_callback=None):
        """Export the complete dataset"""
        try:
            # Create export directory
            self.export_dir.mkdir(exist_ok=True)
            
            # Load all annotation files
            annotation_files = list(self.data_dir.glob("frame_*.json"))
            clip_files = list(self.data_dir.glob("clip_*.json"))
            
            if not annotation_files and not clip_files:
                print("No annotations found to export")
                return False
            
            total_files = len(annotation_files) + len(clip_files)
            
            # Set default augmentation options
            if augmentation_options is None:
                augmentation_options = {'mode': 'No Augmentation', 'enabled_augmentations': []}
            
            if self.format_type == "YOLOv8":
                return self.export_yolo_format(source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files)
            elif self.format_type == "COCO":
                return self.export_coco_format(source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files)
            elif self.format_type == "Pascal VOC":
                return self.export_pascal_format(source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files)
            elif self.format_type == "Custom JSON":
                return self.export_custom_format(source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files)
            elif self.format_type == "MoViNet":
                if not clip_files:
                    print("ERROR: MoViNet export requires clip annotations, but no clip files found!")
                    print(f"Found {len(annotation_files)} frame annotations, but MoViNet needs video clips.")
                    print("")
                    print("To create clips for MoViNet export:")
                    print("1. Go to the 'Clip Labeling (MoViNet)' tab in the application")
                    print("2. Create video clips by setting start/end frames") 
                    print("3. Label each clip with an action (e.g., 'walking', 'running')")
                    print("4. Save the clips")
                    print("5. Then try exporting again")
                    print("")
                    print("Clip requirements:")
                    print("- Minimum: 5 frames per clip")
                    print("- Each clip must have an action label")
                    print("")
                    print("Alternative: You can try auto-converting frame annotations to clips")
                    print("if your frame annotations represent sequential actions.")
                    return False
                return self.export_movinet_format(source_path, source_type, clip_files, progress_callback, len(clip_files))
            else:
                print(f"Unsupported format: {self.format_type}")
                return False
                
        except Exception as e:
            print(f"Export error: {e}")
            return False
    
    def export_yolo_format(self, source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files):
        """Export in YOLO format"""
        try:
            # Create YOLO directory structure
            images_dir = self.export_dir / "images"
            labels_dir = self.export_dir / "labels"
            images_dir.mkdir(exist_ok=True)
            labels_dir.mkdir(exist_ok=True)
            
            # Collect all unique labels
            all_labels = set()
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    for region in data.get('regions', []):
                        if region.get('label'):
                            all_labels.add(region['label'])
            
            # Create classes file
            classes_list = sorted(list(all_labels))
            classes_file = self.export_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                for class_name in classes_list:
                    f.write(f"{class_name}\n")
            
            # Create label mapping
            label_to_id = {label: idx for idx, label in enumerate(classes_list)}
            
            # Determine augmentation settings
            use_augmentation = (augmentation_options['mode'] != "No Augmentation" and 
                              augmentation_options['enabled_augmentations'])
            
            # Process each annotation
            processed = 0
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                frame_index = data['frame_index']
                
                # Load frame from source
                frame = self.load_frame_from_source(source_path, source_type, frame_index)
                if frame is None:
                    continue
                
                frame_height, frame_width = frame.shape[:2]
                
                # Generate frames to export (original + augmented)
                frames_to_export = [(None, frame)]  # (suffix, frame_data)
                
                if use_augmentation:
                    mode = augmentation_options['mode']
                    augmentations = augmentation_options['enabled_augmentations']
                    
                    if mode == "Copy & create new data for each modification":
                        # Create separate copy for each augmentation
                        aug_frames = self.apply_augmentations(frame, augmentations)
                        frames_to_export.extend(aug_frames)
                        
                    elif mode == "Modify frames directly, keep some originals":
                        # Replace original with augmented versions, keep some originals
                        if frame_index % 3 == 0:  # Keep every 3rd frame as original
                            pass  # Keep original
                        else:
                            # Replace with random augmentation
                            aug_frames = self.apply_augmentations(frame, [augmentations[frame_index % len(augmentations)]])
                            if aug_frames:
                                frames_to_export = [aug_frames[0]]  # Replace original
                                
                    elif mode == "Copy & apply multiple random modifiers, keep originals":
                        # Keep original + add random combinations
                        aug_frames = self.apply_random_augmentations(frame, augmentations)
                        frames_to_export.extend(aug_frames)
                        
                    elif mode == "Apply multiple random modifiers, no copies":
                        # Replace with random combinations
                        aug_frames = self.apply_random_augmentations(frame, augmentations, num_versions=1)
                        if aug_frames:
                            frames_to_export = [aug_frames[0]]  # Replace original
                
                # Export all frames (original and/or augmented)
                for suffix, frame_data in frames_to_export:
                    # Create filename with source name prefix
                    if suffix:
                        image_filename = f"{self.source_name}_{frame_index:06d}_{suffix}.jpg"
                        label_filename = f"{self.source_name}_{frame_index:06d}_{suffix}.txt"
                    else:
                        image_filename = f"{self.source_name}_{frame_index:06d}.jpg"
                        label_filename = f"{self.source_name}_{frame_index:06d}.txt"
                    
                    # Save image
                    image_path = images_dir / image_filename
                    cv2.imwrite(str(image_path), frame_data)
                    
                    # Create YOLO annotation
                    label_path = labels_dir / label_filename
                    
                    with open(label_path, 'w') as label_file:
                        for region in data.get('regions', []):
                            if not region.get('label'):
                                continue
                            
                            label = region['label']
                            if label not in label_to_id:
                                continue
                            
                            class_id = label_to_id[label]
                            
                            # Adjust coordinates for mirrored images
                            if suffix and 'mirrored' in suffix:
                                # Flip x coordinates for mirrored images
                                x_center = 1.0 - (region['x'] + region['width'] / 2) / frame_width
                            else:
                                x_center = (region['x'] + region['width'] / 2) / frame_width
                            
                            y_center = (region['y'] + region['height'] / 2) / frame_height
                            width = region['width'] / frame_width
                            height = region['height'] / frame_height
                            
                            label_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
                processed += 1
                if progress_callback:
                    progress_callback.setValue(int(100 * processed / total_files))
            
            # Create dataset YAML file
            yaml_content = f"""
# YOLO Dataset Configuration
path: {self.export_dir.absolute()}
train: images
val: images  # You may want to split this

# Classes
nc: {len(classes_list)}
names: {classes_list}

# Source: {self.source_name}
# Augmentation: {augmentation_options['mode'] if use_augmentation else 'None'}
"""
            
            yaml_file = self.export_dir / "dataset.yaml"
            with open(yaml_file, 'w') as f:
                f.write(yaml_content.strip())
            
            return True
            
        except Exception as e:
            print(f"YOLO export error: {e}")
            return False
    
    def export_coco_format(self, source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files):
        """Export in COCO format"""
        try:
            # Create COCO directory structure
            images_dir = self.export_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            # Initialize COCO structure
            coco_data = {
                "info": {
                    "description": f"AI Labeler Export - {self.source_name}",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().isoformat(),
                    "source": self.source_name,
                    "augmentation": augmentation_options['mode'] if augmentation_options['mode'] != "No Augmentation" else "None"
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }
            
            # Collect categories
            all_labels = set()
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                    for region in data.get('regions', []):
                        if region.get('label'):
                            all_labels.add(region['label'])
            
            # Create categories
            for idx, label in enumerate(sorted(all_labels)):
                coco_data["categories"].append({
                    "id": idx + 1,
                    "name": label,
                    "supercategory": ""
                })
            
            label_to_id = {label: idx + 1 for idx, label in enumerate(sorted(all_labels))}
            
            # Determine augmentation settings
            use_augmentation = (augmentation_options['mode'] != "No Augmentation" and 
                              augmentation_options['enabled_augmentations'])
            
            # Process annotations
            annotation_id = 1
            image_id = 1
            processed = 0
            
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                frame_index = data['frame_index']
                
                # Load frame from source
                frame = self.load_frame_from_source(source_path, source_type, frame_index)
                if frame is None:
                    continue
                
                frame_height, frame_width = frame.shape[:2]
                
                # Generate frames to export (original + augmented)
                frames_to_export = [(None, frame)]  # (suffix, frame_data)
                
                if use_augmentation:
                    mode = augmentation_options['mode']
                    augmentations = augmentation_options['enabled_augmentations']
                    
                    if mode == "Copy & create new data for each modification":
                        aug_frames = self.apply_augmentations(frame, augmentations)
                        frames_to_export.extend(aug_frames)
                    elif mode == "Copy & apply multiple random modifiers, keep originals":
                        aug_frames = self.apply_random_augmentations(frame, augmentations)
                        frames_to_export.extend(aug_frames)
                    # For other modes, use similar logic as YOLO
                
                # Export all frames
                for suffix, frame_data in frames_to_export:
                    # Create filename with source name prefix
                    if suffix:
                        image_filename = f"{self.source_name}_{frame_index:06d}_{suffix}.jpg"
                    else:
                        image_filename = f"{self.source_name}_{frame_index:06d}.jpg"
                    
                    # Save image
                    image_path = images_dir / image_filename
                    cv2.imwrite(str(image_path), frame_data)
                    
                    # Add image info
                    image_info = {
                        "id": image_id,
                        "width": frame_width,
                        "height": frame_height,
                        "file_name": image_filename
                    }
                    coco_data["images"].append(image_info)
                    
                    # Add annotations for this image
                    for region in data.get('regions', []):
                        if not region.get('label'):
                            continue
                        
                        label = region['label']
                        if label not in label_to_id:
                            continue
                        
                        # Adjust coordinates for mirrored images
                        if suffix and 'mirrored' in suffix:
                            # Flip x coordinates for mirrored images
                            x = frame_width - (region['x'] + region['width'])
                            bbox = [x, region['y'], region['width'], region['height']]
                        else:
                            bbox = [region['x'], region['y'], region['width'], region['height']]
                        
                        annotation = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": label_to_id[label],
                            "bbox": bbox,
                            "area": region['width'] * region['height'],
                            "iscrowd": 0
                        }
                        coco_data["annotations"].append(annotation)
                        annotation_id += 1
                    
                    image_id += 1
                
                processed += 1
                if progress_callback:
                    progress_callback.setValue(int(100 * processed / total_files))
            
            # Save COCO JSON
            coco_file = self.export_dir / "annotations.json"
            with open(coco_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"COCO export error: {e}")
            return False
    
    def export_pascal_format(self, source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files):
        """Export in Pascal VOC format"""
        try:
            # Create Pascal directory structure
            images_dir = self.export_dir / "JPEGImages"
            annotations_dir = self.export_dir / "Annotations"
            images_dir.mkdir(exist_ok=True)
            annotations_dir.mkdir(exist_ok=True)
            
            processed = 0
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                frame_index = data['frame_index']
                
                # Load frame from source using standardized method
                frame = self.load_frame_from_source(source_path, source_type, frame_index)
                if frame is None:
                    continue
                
                frame_height, frame_width, frame_depth = frame.shape
                
                # Save image
                image_filename = f"frame_{frame_index:06d}.jpg"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), frame)
                
                # Create Pascal VOC XML
                root = ET.Element("annotation")
                
                # Add folder
                folder = ET.SubElement(root, "folder")
                folder.text = "JPEGImages"
                
                # Add filename
                filename = ET.SubElement(root, "filename")
                filename.text = image_filename
                
                # Add size
                size = ET.SubElement(root, "size")
                width_elem = ET.SubElement(size, "width")
                width_elem.text = str(frame_width)
                height_elem = ET.SubElement(size, "height")
                height_elem.text = str(frame_height)
                depth_elem = ET.SubElement(size, "depth")
                depth_elem.text = str(frame_depth)
                
                # Add segmented
                segmented = ET.SubElement(root, "segmented")
                segmented.text = "0"
                
                # Add objects
                for region in data.get('regions', []):
                    if not region.get('label'):
                        continue
                    
                    obj = ET.SubElement(root, "object")
                    
                    name = ET.SubElement(obj, "name")
                    name.text = region['label']
                    
                    pose = ET.SubElement(obj, "pose")
                    pose.text = "Unspecified"
                    
                    truncated = ET.SubElement(obj, "truncated")
                    truncated.text = "0"
                    
                    difficult = ET.SubElement(obj, "difficult")
                    difficult.text = "0"
                    
                    bndbox = ET.SubElement(obj, "bndbox")
                    
                    xmin = ET.SubElement(bndbox, "xmin")
                    xmin.text = str(int(region['x']))
                    
                    ymin = ET.SubElement(bndbox, "ymin")
                    ymin.text = str(int(region['y']))
                    
                    xmax = ET.SubElement(bndbox, "xmax")
                    xmax.text = str(int(region['x'] + region['width']))
                    
                    ymax = ET.SubElement(bndbox, "ymax")
                    ymax.text = str(int(region['y'] + region['height']))
                
                # Save XML
                xml_filename = f"frame_{frame_index:06d}.xml"
                xml_path = annotations_dir / xml_filename
                
                tree = ET.ElementTree(root)
                tree.write(str(xml_path), encoding='utf-8', xml_declaration=True)
                
                processed += 1
                if progress_callback:
                    progress_callback.setValue(int(100 * processed / total_files))
            
            return True
            
        except Exception as e:
            print(f"Pascal VOC export error: {e}")
            return False
    
    def export_custom_format(self, source_path, source_type, annotation_files, augmentation_options, progress_callback, total_files):
        """Export in custom JSON format"""
        try:
            # Create custom directory structure
            images_dir = self.export_dir / "images"
            images_dir.mkdir(exist_ok=True)
            
            export_data = {
                "dataset_info": {
                    "name": "AI Labeler Dataset",
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "video_source": str(source_path),
                    "total_frames": len(annotation_files)
                },
                "frames": []
            }
            
            processed = 0
            for annotation_file in annotation_files:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)
                
                frame_index = data['frame_index']
                
                # Load frame from source using standardized method
                frame = self.load_frame_from_source(source_path, source_type, frame_index)
                if frame is None:
                    continue
                
                frame_height, frame_width = frame.shape[:2]
                
                # Save image
                image_filename = f"frame_{frame_index:06d}.jpg"
                image_path = images_dir / image_filename
                cv2.imwrite(str(image_path), frame)
                
                # Create frame data
                frame_data = {
                    "frame_index": frame_index,
                    "image_file": image_filename,
                    "width": frame_width,
                    "height": frame_height,
                    "frame_labels": data.get('frame_labels', []),
                    "regions": data.get('regions', [])
                }
                
                export_data["frames"].append(frame_data)
                
                processed += 1
                if progress_callback:
                    progress_callback.setValue(int(100 * processed / total_files))
            
            # Save custom JSON
            json_file = self.export_dir / "dataset.json"
            with open(json_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Custom export error: {e}")
            return False
    
    def export_movinet_format(self, source_path, source_type, clip_files, progress_callback, total_files):
        """Export in MoViNet format for video classification"""
        try:
            # Create MoViNet directory structure
            clips_dir = self.export_dir / "clips"
            clips_dir.mkdir(exist_ok=True)
            
            # Collect all unique labels
            all_labels = set()
            clip_data = []
            
            for clip_file in clip_files:
                with open(clip_file, 'r') as f:
                    data = json.load(f)
                    label = data.get('label', '')
                    if label:
                        all_labels.add(label)
                    clip_data.append(data)
            
            classes_list = sorted(list(all_labels))
            if not classes_list:
                print("No class labels found in clips")
                return False
            
            # Save classes file
            classes_file = self.export_dir / "classes.txt"
            with open(classes_file, 'w') as f:
                for label in classes_list:
                    f.write(f"{label}\n")
            
            label_to_id = {label: idx for idx, label in enumerate(classes_list)}
            
            # Create dataset metadata
            dataset_metadata = {
                "info": {
                    "description": "MoViNet Dataset Export",
                    "version": "1.0",
                    "year": datetime.now().year,
                    "date_created": datetime.now().isoformat()
                },
                "classes": classes_list,
                "clips": []
            }
            
            # Process each clip
            processed = 0
            for clip_info in clip_data:
                start_frame = clip_info['start_frame']
                end_frame = clip_info['end_frame']
                label = clip_info['label']
                clip_id = clip_info.get('id', f"clip_{start_frame}_{end_frame}")
                
                # Create clip directory
                clip_dir = clips_dir / clip_id
                clip_dir.mkdir(exist_ok=True)
                
                # Extract frames for the clip
                frames_extracted = []
                frame_count = end_frame - start_frame + 1
                
                # Use all frames - no maximum limit enforced
                frame_indices = list(range(start_frame, end_frame + 1))
                
                for i, frame_idx in enumerate(frame_indices):
                    # Load frame from source using standardized method
                    frame = self.load_frame_from_source(source_path, source_type, frame_idx)
                    if frame is None:
                        continue
                    
                    # Resize to 224x224 for MoViNet
                    frame_resized = cv2.resize(frame, (224, 224))
                    
                    # Save frame
                    frame_filename = f"frame_{i:04d}.jpg"
                    frame_path = clip_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame_resized)
                    frames_extracted.append(frame_filename)
                
                # Add to dataset metadata
                clip_metadata = {
                    "clip_id": clip_id,
                    "label": label,
                    "class_id": label_to_id[label],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "frames": frames_extracted,
                    "frame_count": len(frames_extracted),
                    "resolution": [224, 224]
                }
                
                dataset_metadata["clips"].append(clip_metadata)
                
                processed += 1
                if progress_callback:
                    progress_callback.setValue(int(100 * processed / total_files))
            
            # Save dataset metadata
            metadata_file = self.export_dir / "dataset.json"
            with open(metadata_file, 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            # Create train/val split
            self.create_movinet_splits(dataset_metadata)
            
            # Create PyTorch dataset configuration
            self.create_pytorch_config(dataset_metadata)
            
            return True
            
        except Exception as e:
            print(f"MoViNet export error: {e}")
            return False
    
    def create_movinet_splits(self, dataset_metadata, train_ratio=0.8):
        """Create train/validation splits for MoViNet dataset"""
        clips = dataset_metadata["clips"]
        random.shuffle(clips)
        
        split_index = int(len(clips) * train_ratio)
        train_clips = clips[:split_index]
        val_clips = clips[split_index:]
        
        # Save splits
        splits = {
            "train": [clip["clip_id"] for clip in train_clips],
            "validation": [clip["clip_id"] for clip in val_clips]
        }
        
        splits_file = self.export_dir / "splits.json"
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
    
    def create_pytorch_config(self, dataset_metadata):
        """Create PyTorch dataset configuration file"""
        config = {
            "dataset_name": "movinet_export",
            "num_classes": len(dataset_metadata["classes"]),
            "classes": dataset_metadata["classes"],
            "input_resolution": [224, 224],
            "sequence_length_range": [5, 64],
            "fps": 30,
            "data_path": str(self.export_dir / "clips"),
            "metadata_path": str(self.export_dir / "dataset.json"),
            "splits_path": str(self.export_dir / "splits.json")
        }
        
        config_file = self.export_dir / "pytorch_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create example PyTorch dataset loader code
        example_code = '''
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MoViNetDataset(Dataset):
    def __init__(self, config_path, split='train', transform=None):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        with open(self.config['metadata_path'], 'r') as f:
            self.metadata = json.load(f)
        
        with open(self.config['splits_path'], 'r') as f:
            splits = json.load(f)
        
        self.clip_ids = splits[split]
        self.clips = [clip for clip in self.metadata['clips'] if clip['clip_id'] in self.clip_ids]
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip = self.clips[idx]
        clip_path = os.path.join(self.config['data_path'], clip['clip_id'])
        
        frames = []
        for frame_name in sorted(clip['frames']):
            frame_path = os.path.join(clip_path, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        
        # Stack frames into tensor (T, C, H, W)
        video_tensor = torch.stack(frames)
        label = clip['class_id']
        
        return video_tensor, label

# Example usage:
# dataset = MoViNetDataset('pytorch_config.json', split='train')
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
'''
        
        example_file = self.export_dir / "example_pytorch_loader.py"
        with open(example_file, 'w') as f:
            f.write(example_code)
    
    def convert_frame_annotations_to_clips(self, video_path, annotation_files, clip_duration_frames=30):
        """
        Convert frame annotations to clips for MoViNet export
        
        Args:
            video_path: Path to the video file
            annotation_files: List of frame annotation files
            clip_duration_frames: Number of frames per clip (default 30 = ~1 second at 30fps)
            
        Returns:
            List of clip file paths created
        """
        print("Converting frame annotations to clips for MoViNet...")
        
        # Group frame annotations by label to create coherent clips
        frames_by_label = {}
        
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            frame_index = data['frame_index']
            
            # Use frame labels if available, otherwise use region labels
            labels = data.get('frame_labels', [])
            if not labels and data.get('regions'):
                labels = [region.get('label') for region in data['regions'] if region.get('label')]
            
            for label in labels:
                if label not in frames_by_label:
                    frames_by_label[label] = []
                frames_by_label[label].append(frame_index)
        
        if not frames_by_label:
            print("No frame labels found to convert to clips")
            return []
        
        # Create clips from grouped frames
        clip_files = []
        clip_id = 1
        
        for label, frame_indices in frames_by_label.items():
            frame_indices.sort()
            
            # Group consecutive frames into sequences
            sequences = []
            current_sequence = [frame_indices[0]]
            
            for i in range(1, len(frame_indices)):
                if frame_indices[i] == frame_indices[i-1] + 1:
                    # Consecutive frame
                    current_sequence.append(frame_indices[i])
                else:
                    # Gap found, start new sequence
                    sequences.append(current_sequence)
                    current_sequence = [frame_indices[i]]
            
            # Add the last sequence
            sequences.append(current_sequence)
            
            # Create clips from sequences
            for sequence in sequences:
                start_frame = sequence[0]
                end_frame = sequence[-1]
                
                # Extend short sequences to meet minimum requirements
                if len(sequence) < 5:
                    # Try to extend the sequence to 5 frames
                    needed_frames = 5 - len(sequence)
                    # Extend forward first, then backward if needed
                    extended_end = end_frame + needed_frames
                    if extended_end - start_frame + 1 >= 5:
                        end_frame = extended_end
                    else:
                        # Try extending backward
                        extended_start = max(0, start_frame - needed_frames)
                        start_frame = extended_start
                        if end_frame - start_frame + 1 < 5:
                            end_frame = start_frame + 4  # Minimum 5 frames
                
                clip_length = end_frame - start_frame + 1
                
                if clip_length >= 5:  # MoViNet minimum
                    clip_data = {
                        "id": f"auto_clip_{clip_id:03d}",
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "label": label,
                        "description": f"Auto-generated clip from frame annotations ({label})",
                        "auto_generated": True
                    }
                    
                    # Save clip file
                    clip_file = self.data_dir / f"clip_{clip_id:03d}.json"
                    with open(clip_file, 'w') as f:
                        json.dump(clip_data, f, indent=2)
                    
                    clip_files.append(clip_file)
                    clip_id += 1
                    
                    print(f"Created clip: {label} (frames {start_frame}-{end_frame}, {clip_length} frames)")
        
        print(f"Created {len(clip_files)} clips from frame annotations")
        return clip_files
    
    def apply_augmentations(self, frame, augmentations):
        """Apply selected augmentations to a frame"""
        augmented_frames = []
        
        # Always keep original if mode requires it
        modes_keep_original = [
            "Copy & create new data for each modification",
            "Modify frames directly, keep some originals",
            "Copy & apply multiple random modifiers, keep originals"
        ]
        
        for aug_type in augmentations:
            if aug_type == 'pixelation':
                augmented_frames.append(('pixelated', self.apply_pixelation(frame)))
            elif aug_type == 'blur':
                augmented_frames.append(('blurred', self.apply_blur(frame)))
            elif aug_type == 'rotation':
                augmented_frames.append(('rotated', self.apply_rotation(frame)))
            elif aug_type == 'mirror':
                augmented_frames.append(('mirrored', self.apply_mirror(frame)))
            elif aug_type == 'brightness':
                augmented_frames.append(('bright', self.apply_brightness(frame)))
            elif aug_type == 'contrast':
                augmented_frames.append(('contrast', self.apply_contrast(frame)))
            elif aug_type == 'saturation':
                augmented_frames.append(('saturated', self.apply_saturation(frame)))
            elif aug_type == 'noise':
                augmented_frames.append(('noisy', self.apply_noise(frame)))
        
        return augmented_frames
    
    def apply_random_augmentations(self, frame, augmentations, num_versions=3):
        """Apply random combinations of augmentations"""
        import random as rnd
        augmented_frames = []
        
        for i in range(num_versions):
            # Select random subset of augmentations
            selected_augs = rnd.sample(augmentations, rnd.randint(1, min(3, len(augmentations))))
            
            aug_frame = frame.copy()
            suffix_parts = []
            
            for aug in selected_augs:
                if aug == 'pixelation':
                    aug_frame = self.apply_pixelation(aug_frame)
                    suffix_parts.append('pix')
                elif aug == 'blur':
                    aug_frame = self.apply_blur(aug_frame)
                    suffix_parts.append('blur')
                elif aug == 'rotation':
                    aug_frame = self.apply_rotation(aug_frame)
                    suffix_parts.append('rot')
                elif aug == 'mirror':
                    aug_frame = self.apply_mirror(aug_frame)
                    suffix_parts.append('mir')
                elif aug == 'brightness':
                    aug_frame = self.apply_brightness(aug_frame)
                    suffix_parts.append('brt')
                elif aug == 'contrast':
                    aug_frame = self.apply_contrast(aug_frame)
                    suffix_parts.append('con')
                elif aug == 'saturation':
                    aug_frame = self.apply_saturation(aug_frame)
                    suffix_parts.append('sat')
                elif aug == 'noise':
                    aug_frame = self.apply_noise(aug_frame)
                    suffix_parts.append('noi')
            
            suffix = '_'.join(suffix_parts)
            augmented_frames.append((f'combo_{suffix}', aug_frame))
        
        return augmented_frames
    
    def apply_pixelation(self, frame):
        """Apply pixelation effect"""
        height, width = frame.shape[:2]
        # Downscale and upscale to create pixelation
        factor = 8
        small = cv2.resize(frame, (width//factor, height//factor), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    def apply_blur(self, frame):
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, (15, 15), 0)
    
    def apply_rotation(self, frame):
        """Apply random rotation ±10 degrees"""
        import random as rnd
        height, width = frame.shape[:2]
        angle = rnd.uniform(-10, 10)
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, matrix, (width, height))
    
    def apply_mirror(self, frame):
        """Apply horizontal flip"""
        return cv2.flip(frame, 1)
    
    def apply_brightness(self, frame):
        """Adjust brightness"""
        import random as rnd
        factor = rnd.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    def apply_contrast(self, frame):
        """Adjust contrast"""
        import random as rnd
        factor = rnd.uniform(0.8, 1.2)
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0)
    
    def apply_saturation(self, frame):
        """Adjust saturation"""
        import random as rnd
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        factor = rnd.uniform(0.7, 1.3)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], factor)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def apply_noise(self, frame):
        """Add random noise"""
        import random as rnd
        noise = np.random.normal(0, 15, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)
    
    def load_frame_from_source(self, source_path, source_type, frame_index):
        """Load a frame from either video or image sequence"""
        if source_type == 'video':
            cap = cv2.VideoCapture(str(source_path))
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
        else:  # images
            # For image directories, we need to get the image file corresponding to frame_index
            # This assumes the annotation files contain the correct frame_index
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = []
            source_path = Path(source_path)
            for ext in image_extensions:
                image_files.extend(source_path.glob(f"*{ext}"))
                image_files.extend(source_path.glob(f"*{ext.upper()}"))
            image_files = sorted(image_files)
            
            if 0 <= frame_index < len(image_files):
                return cv2.imread(str(image_files[frame_index]))
            return None

    def ensure_clean_frame_export(self, frame):
        """
        Ensure the frame is clean for export (no UI overlays, annotations, or text)
        This method validates that the frame contains only the original video/image data
        without any UI elements that might have been added for display purposes.
        """
        if frame is None:
            return None
        
        # Return a copy to ensure the original source frame is not modified
        # This prevents any accidental modification of the source data
        clean_frame = frame.copy()
        
        # Additional validation could be added here if needed
        # For example, checking for specific colors or patterns that indicate UI elements
        
        return clean_frame 