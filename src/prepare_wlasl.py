#!/usr/bin/env python3
"""
Prepare WLASL dataset for training
Downloads and organizes WLASL dataset into the format needed for training
"""

import os
import json
import shutil
from pathlib import Path
import argparse
import requests
from tqdm import tqdm
import zipfile


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def organize_wlasl_videos(wlasl_dir: Path, output_dir: Path, class_mapping_file: Path = None):
    """
    Organize WLASL videos into class directories
    
    WLASL structure is typically:
    - videos/ directory with all videos
    - JSON file with annotations mapping video names to glosses (sign names)
    """
    videos_dir = wlasl_dir / 'videos'
    if not videos_dir.exists():
        print(f"Error: {videos_dir} not found")
        print("Expected WLASL structure: wlasl_root/videos/")
        return None
    
    # Try to find annotation file
    annotation_files = list(wlasl_dir.glob('*.json'))
    if not annotation_files:
        print("Warning: No JSON annotation file found")
        print("You may need to download WLASL annotations separately")
        return None
    
    # Load annotations
    annotations = {}
    for ann_file in annotation_files:
        try:
            with open(ann_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if 'gloss' in item and 'video_id' in item:
                            annotations[item['video_id']] = item['gloss']
                elif isinstance(data, dict):
                    annotations.update(data)
        except Exception as e:
            print(f"Warning: Could not parse {ann_file}: {e}")
    
    if not annotations:
        print("Error: No annotations found in JSON files")
        return None
    
    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Organize videos by gloss (sign name)
    video_files = list(videos_dir.glob('*.mp4'))
    organized = 0
    missing_annotations = []
    
    print(f"Organizing {len(video_files)} videos...")
    
    for video_file in tqdm(video_files, desc="Organizing videos"):
        # Extract video ID from filename (WLASL format: video_id.mp4)
        video_id = video_file.stem
        
        if video_id in annotations:
            gloss = annotations[video_id]
            # Sanitize gloss name for directory
            gloss_dir = output_dir / gloss.replace(' ', '_').replace('/', '_')
            gloss_dir.mkdir(exist_ok=True)
            
            # Copy video to appropriate directory
            dest = gloss_dir / video_file.name
            if not dest.exists():
                shutil.copy2(video_file, dest)
            organized += 1
        else:
            missing_annotations.append(video_id)
    
    print(f"\nOrganized {organized} videos into {len(list(output_dir.iterdir()))} sign classes")
    
    if missing_annotations:
        print(f"Warning: {len(missing_annotations)} videos had no annotations")
    
    # Create class mapping
    class_dirs = sorted([d.name for d in output_dir.iterdir() if d.is_dir()])
    class_mapping = {name: idx for idx, name in enumerate(class_dirs)}
    
    # Save class mapping
    if class_mapping_file:
        with open(class_mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=2)
        print(f"Saved class mapping to {class_mapping_file}")
    
    return class_mapping


def main():
    parser = argparse.ArgumentParser(description='Prepare WLASL dataset for training')
    parser.add_argument('--wlasl-dir', type=str, required=True,
                       help='Directory containing WLASL dataset (should have videos/ subdirectory and JSON annotations)')
    parser.add_argument('--output-dir', type=str, default='dataset/wlasl',
                       help='Output directory for organized dataset')
    parser.add_argument('--class-mapping', type=str, default='models/wlasl_class_mapping.json',
                       help='Output file for class mapping JSON')
    
    args = parser.parse_args()
    
    wlasl_path = Path(args.wlasl_dir)
    output_path = Path(args.output_dir)
    mapping_path = Path(args.class_mapping)
    
    if not wlasl_path.exists():
        print(f"Error: WLASL directory not found: {wlasl_path}")
        print("\nTo download WLASL dataset:")
        print("1. Visit: https://github.com/dxli94/WLASL")
        print("2. Follow their download instructions")
        print("3. Extract the dataset to a directory")
        print("4. Run this script with --wlasl-dir pointing to that directory")
        return
    
    print(f"Preparing WLASL dataset from: {wlasl_path}")
    print(f"Output directory: {output_path}")
    
    # Ensure output directory exists
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Organize videos
    class_mapping = organize_wlasl_videos(wlasl_path, output_path, mapping_path)
    
    if class_mapping:
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  - Organized videos: {output_path}")
        print(f"  - Class mapping: {mapping_path}")
        print(f"  - Number of classes: {len(class_mapping)}")
        print(f"\nNext step: Train the model with:")
        print(f"  python src/train_asl_model.py --data-dir {output_path} --output-dir models/wlasl")


if __name__ == '__main__':
    main()

