#!/usr/bin/env python3
"""
Script to create individual clip files from clips.json for MoViNet export
"""

import json
from pathlib import Path

def create_individual_clips():
    """Create individual clip files from clips.json"""
    labeled_data_dir = Path("labeled_data")
    clips_file = labeled_data_dir / "clips.json"
    
    if not clips_file.exists():
        print("clips.json not found!")
        return
    
    # Load clips data
    with open(clips_file, 'r') as f:
        clips_data = json.load(f)
    
    # Remove existing individual clip files
    for clip_file in labeled_data_dir.glob("clip_*.json"):
        clip_file.unlink()
        print(f"Removed existing file: {clip_file}")
    
    # Save each clip as individual file
    created_files = 0
    for i, clip in enumerate(clips_data, 1):
        if clip.get('label'):  # Only save labeled clips
            clip_file = labeled_data_dir / f"clip_{i:03d}.json"
            try:
                with open(clip_file, 'w') as f:
                    json.dump(clip, f, indent=2)
                print(f"Created: {clip_file}")
                created_files += 1
            except Exception as e:
                print(f"Error saving individual clip file {clip_file}: {e}")
    
    print(f"\nCreated {created_files} individual clip files for MoViNet export")

if __name__ == "__main__":
    create_individual_clips() 