#!/usr/bin/env python3
"""
Debug frame-to-clip conversion
"""

from pathlib import Path
import json

def debug_frame_annotations():
    print("Debugging Frame Annotations")
    print("=" * 30)
    
    data_dir = Path('labeled_data')
    frame_files = list(data_dir.glob("frame_*.json"))
    
    print(f"Found {len(frame_files)} frame files")
    
    frames_by_label = {}
    
    for frame_file in frame_files:
        with open(frame_file, 'r') as f:
            data = json.load(f)
        
        frame_index = data['frame_index']
        print(f"\nFrame {frame_index}:")
        
        # Check frame labels
        frame_labels = data.get('frame_labels', [])
        print(f"  Frame labels: {frame_labels}")
        
        # Check region labels
        regions = data.get('regions', [])
        region_labels = [region.get('label') for region in regions if region.get('label')]
        print(f"  Region labels: {region_labels}")
        
        # Collect all labels
        all_labels = frame_labels + region_labels
        print(f"  All labels: {all_labels}")
        
        for label in all_labels:
            if label not in frames_by_label:
                frames_by_label[label] = []
            frames_by_label[label].append(frame_index)
    
    print(f"\nGrouped frames by label:")
    for label, frame_indices in frames_by_label.items():
        print(f"  {label}: {sorted(frame_indices)}")
    
    return frames_by_label

if __name__ == "__main__":
    debug_frame_annotations() 