#!/usr/bin/env python3
"""
MoViNet Export Helper
Helps users fix MoViNet export issues by converting frame annotations to clips
"""

from data_exporter import DataExporter
from pathlib import Path
import json
import sys

def analyze_annotations():
    """Analyze existing annotations to understand what data is available"""
    print("Analyzing existing annotations...")
    print("=" * 40)
    
    data_dir = Path('labeled_data')
    
    # Check for frame annotations
    frame_files = list(data_dir.glob("frame_*.json"))
    clip_files = list(data_dir.glob("clip_*.json"))
    
    print(f"Frame annotation files: {len(frame_files)}")
    print(f"Clip annotation files: {len(clip_files)}")
    
    if not frame_files and not clip_files:
        print("❌ No annotations found!")
        print("Please create some annotations first using the AI Labeler application.")
        return False
    
    if clip_files:
        print("✅ Clip files found - MoViNet export should work")
        return True
    
    if frame_files:
        print("⚠️  Only frame annotations found - need clips for MoViNet")
        
        # Analyze frame labels
        frame_labels = set()
        region_labels = set()
        
        for frame_file in frame_files:
            with open(frame_file, 'r') as f:
                data = json.load(f)
                frame_labels.update(data.get('frame_labels', []))
                for region in data.get('regions', []):
                    if region.get('label'):
                        region_labels.add(region['label'])
        
        print(f"\nFrame labels found: {list(frame_labels)}")
        print(f"Region labels found: {list(region_labels)}")
        
        if frame_labels or region_labels:
            print("\n✅ Labels found - can convert to clips!")
            return "convert"
        else:
            print("\n❌ No labels found in frame annotations")
            return False
    
    return False

def convert_frames_to_clips():
    """Convert frame annotations to clips for MoViNet export"""
    print("\nConverting frame annotations to clips...")
    print("=" * 40)
    
    exporter = DataExporter('labeled_data', 'MoViNet')
    annotation_files = list(Path('labeled_data').glob("frame_*.json"))
    
    # Ask user for clip duration
    print("Clip duration options:")
    print("1. Short clips (15 frames ≈ 0.5 seconds)")
    print("2. Medium clips (30 frames ≈ 1 second)")
    print("3. Long clips (60 frames ≈ 2 seconds)")
    
    try:
        choice = input("Choose clip duration (1-3) [default: 2]: ").strip()
        if choice == "1":
            duration = 15
        elif choice == "3":
            duration = 60
        else:
            duration = 30
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return False
    
    print(f"Using clip duration: {duration} frames")
    
    # Convert annotations
    clip_files = exporter.convert_frame_annotations_to_clips(
        'sample_video.mp4', 
        annotation_files, 
        clip_duration_frames=duration
    )
    
    if clip_files:
        print(f"\n✅ Successfully created {len(clip_files)} clip files")
        return True
    else:
        print("\n❌ Failed to create clips from frame annotations")
        return False

def test_movinet_export():
    """Test MoViNet export after conversion"""
    print("\nTesting MoViNet export...")
    print("=" * 30)
    
    exporter = DataExporter('labeled_data', 'MoViNet')
    result = exporter.export_dataset('sample_video.mp4')
    
    if result:
        print("✅ MoViNet export successful!")
        print("\nExported files can be found in: labeled_data/export_movinet/")
        print("- clips/: Individual clip directories with frames")
        print("- dataset.json: Metadata for all clips")
        print("- classes.txt: List of action classes")
        print("- splits.json: Train/validation split")
        print("- pytorch_config.json: PyTorch dataset configuration")
        return True
    else:
        print("❌ MoViNet export failed")
        return False

def main():
    print("MoViNet Export Helper")
    print("=" * 50)
    print("This tool helps fix MoViNet export issues by converting")
    print("frame annotations to video clips.")
    print("")
    
    # Analyze current state
    analysis_result = analyze_annotations()
    
    if analysis_result is False:
        print("\n❌ Cannot proceed with MoViNet export.")
        print("Please create annotations using the AI Labeler application first.")
        sys.exit(1)
    
    if analysis_result is True:
        print("\n✅ Clips already exist - trying MoViNet export...")
        if test_movinet_export():
            print("\n🎉 MoViNet export completed successfully!")
        else:
            print("\n❌ MoViNet export failed despite having clips.")
            print("Check the error messages above for details.")
        return
    
    if analysis_result == "convert":
        print("\n🔄 Frame annotations found - conversion needed")
        
        try:
            proceed = input("Convert frame annotations to clips? (y/n) [default: y]: ").strip().lower()
            if proceed and proceed != 'y' and proceed != 'yes':
                print("Operation cancelled.")
                return
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
        
        # Convert frames to clips
        if convert_frames_to_clips():
            # Test export
            if test_movinet_export():
                print("\n🎉 MoViNet export completed successfully!")
            else:
                print("\n❌ MoViNet export failed after conversion.")
        else:
            print("\n❌ Failed to convert frame annotations to clips.")

if __name__ == "__main__":
    main() 