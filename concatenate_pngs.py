#!/usr/bin/env python3
"""
Script to concatenate PNG files with the same name from two directories.
Reads PNGs from generated_dpg and generated_dpg_base, concatenates them,
and saves the result to pairwise_comp.
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def concatenate_images_horizontally(img1, img2):
    """
    Concatenate two images horizontally (side by side).
    If images have different heights, resize the smaller one to match.
    """
    # Get dimensions
    width1, height1 = img1.size
    width2, height2 = img2.size
    
    # Resize images to have the same height (use the maximum height)
    max_height = max(height1, height2)
    
    if height1 != max_height:
        # Resize img1 to match max_height, maintaining aspect ratio
        ratio = max_height / height1
        new_width1 = int(width1 * ratio)
        img1 = img1.resize((new_width1, max_height), Image.Resampling.LANCZOS)
        width1 = new_width1
    
    if height2 != max_height:
        # Resize img2 to match max_height, maintaining aspect ratio
        ratio = max_height / height2
        new_width2 = int(width2 * ratio)
        img2 = img2.resize((new_width2, max_height), Image.Resampling.LANCZOS)
        width2 = new_width2
    
    # Create new image with combined width
    total_width = width1 + width2
    concatenated = Image.new('RGB', (total_width, max_height))
    
    # Paste images side by side
    concatenated.paste(img1, (0, 0))
    concatenated.paste(img2, (width1, 0))
    
    return concatenated


def main():
    parser = argparse.ArgumentParser(description='Concatenate PNG files from two directories')
    parser.add_argument('--dir1', type=str, 
                       default='/home/aiops/zhangfz/geneval_jc/results/generated_dpg',
                       help='First directory containing PNG files')
    parser.add_argument('--dir2', type=str,
                       default='/home/aiops/zhangfz/geneval_jc/results/generated_dpg_base',
                       help='Second directory containing PNG files')
    parser.add_argument('--output', type=str,
                       default='/home/aiops/zhangfz/geneval_jc/results/pairwise_comp',
                       help='Output directory for concatenated images')
    
    args = parser.parse_args()
    
    dir1 = Path(args.dir1)
    dir2 = Path(args.dir2)
    output_dir = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PNG files in both directories
    png_files_dir1 = {f.name: f for f in dir1.glob('*.png')}
    png_files_dir2 = {f.name: f for f in dir2.glob('*.png')}
    
    # Find common filenames
    common_files = set(png_files_dir1.keys()) & set(png_files_dir2.keys())
    
    print(f"Found {len(png_files_dir1)} PNG files in {dir1}")
    print(f"Found {len(png_files_dir2)} PNG files in {dir2}")
    print(f"Found {len(common_files)} common PNG files to concatenate")
    
    # Process each common file
    success_count = 0
    error_count = 0
    
    for filename in sorted(common_files):
        try:
            # Load images
            img1_path = png_files_dir1[filename]
            img2_path = png_files_dir2[filename]
            
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img1.mode != 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode != 'RGB':
                img2 = img2.convert('RGB')
            
            # Concatenate images
            concatenated = concatenate_images_horizontally(img1, img2)
            
            # Save concatenated image
            output_path = output_dir / filename
            concatenated.save(output_path, 'PNG')
            
            success_count += 1
            if success_count % 100 == 0:
                print(f"Processed {success_count} images...")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            error_count += 1
    
    print(f"\nCompleted!")
    print(f"Successfully concatenated: {success_count} images")
    print(f"Errors: {error_count}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
