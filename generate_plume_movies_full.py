#!/usr/bin/env python3
"""Generate full movies from both HDF5 plume datasets at native frame rates."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import yaml

def load_intensity_range(plume_type):
    """Load intensity range from metadata files."""
    # Load from plume_intensity_stats.yaml
    with open('configs/plume_intensity_stats.yaml', 'r') as f:
        stats = yaml.safe_load(f)
    
    if plume_type == 'crimaldi':
        vmin = stats['CRIM']['min']
        vmax = stats['CRIM']['max']
        print(f"  Loaded Crimaldi range from metadata: [{vmin:.6f}, {vmax:.6f}]")
    elif plume_type == 'smoke':
        # Smoke was scaled to Crimaldi range
        vmin = stats['CRIM']['min']
        vmax = stats['CRIM']['max']
        print(f"  Using Crimaldi range for scaled smoke: [{vmin:.6f}, {vmax:.6f}]")
    else:
        raise ValueError(f"Unknown plume type: {plume_type}")
    
    return vmin, vmax

def create_plume_movie(h5_path, dataset_name, output_name, title_prefix, native_fps, plume_type):
    """Create a full movie from HDF5 plume data at native frame rate.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_name: Name of dataset in HDF5 file
        output_name: Output movie filename (will be saved to palmer_scratch/plume/)
        title_prefix: Prefix for plot title
        native_fps: Native frame rate of the data
        plume_type: 'crimaldi' or 'smoke' for loading correct metadata
    """
    print(f"\nCreating full movie for {title_prefix}...")
    print(f"  Native frame rate: {native_fps} Hz")
    
    # Load value range from metadata
    vmin, vmax = load_intensity_range(plume_type)
    
    # Set output path to palmer_scratch
    output_dir = "/home/snb6/palmer_scratch/plume"
    output_path = os.path.join(output_dir, output_name)
    print(f"  Output path: {output_path}")
    
    # Open HDF5 file
    with h5py.File(h5_path, 'r') as f:
        data = f[dataset_name]
        shape = data.shape
        n_frames = shape[0]
        print(f"  Data shape: {shape}")
        print(f"  Total frames: {n_frames}")
        print(f"  Duration: {n_frames/native_fps:.1f} seconds")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Initial plot
        first_frame = data[0, :, :]
        im = ax.imshow(first_frame, cmap='hot', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{title_prefix}')
        ax.set_xlabel(f'Width: {shape[2]} pixels')
        ax.set_ylabel(f'Height: {shape[1]} pixels')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Intensity')
        
        # Add time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add data info text
        info_text = ax.text(0.98, 0.98, f'Range: [{vmin:.3f}, {vmax:.3f}]', 
                           transform=ax.transAxes, fontsize=10, 
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        def animate(frame_idx):
            # Read frame directly from HDF5
            frame_data = data[frame_idx, :, :]
            im.set_array(frame_data)
            
            # Update time text
            time_sec = frame_idx / native_fps
            time_text.set_text(f'Frame {frame_idx}/{n_frames-1}\nTime: {time_sec:.2f}s')
            
            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"    Processing frame {frame_idx}/{n_frames-1} ({100*frame_idx/n_frames:.1f}%)")
            
            return [im, time_text]
        
        # Create animation
        print("  Creating animation...")
        print("  This will take several minutes for the full movie...")
        
        # Calculate interval to match native frame rate
        interval = 1000.0 / native_fps  # milliseconds per frame
        
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                       interval=interval, blit=True)
        
        # Save as MP4 at native frame rate
        print(f"  Saving to {output_path} at {native_fps} fps...")
        writer = animation.FFMpegWriter(fps=native_fps, bitrate=3000, 
                                        metadata={'title': title_prefix})
        anim.save(output_path, writer=writer)
        plt.close()
    
    print(f"  ✓ Saved movie to {output_path}")
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

def main():
    """Generate movies for both plumes."""
    print("=" * 60)
    print("Generating FULL plume movies at native frame rates")
    print("Using intensity ranges from metadata files")
    print("Saving to /home/snb6/palmer_scratch/plume/")
    print("=" * 60)
    
    # Check if FFmpeg is available
    try:
        import subprocess
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✓ FFmpeg is available")
    except:
        print("ERROR: FFmpeg not found. Please ensure FFmpeg is installed.")
        return
    
    # Check output directory exists
    output_dir = "/home/snb6/palmer_scratch/plume"
    if not os.path.exists(output_dir):
        print(f"ERROR: Output directory does not exist: {output_dir}")
        return
    
    # Crimaldi plume - 15 Hz
    create_plume_movie(
        'data/10302017_10cms_bounded.hdf5',
        '/dataset2',
        'crimaldi_plume_full_15Hz.mp4',
        'Crimaldi Plume',
        native_fps=15,
        plume_type='crimaldi'
    )
    
    # Smoke plume - 60 Hz
    create_plume_movie(
        'data/smoke_1a_rotated_3d.h5',
        '/dataset2',
        'smoke_plume_full_60Hz.mp4',
        'Smoke Plume',
        native_fps=60,
        plume_type='smoke'
    )
    
    print("\n" + "=" * 60)
    print("✓ Movie generation complete!")
    print("=" * 60)
    print("\nGenerated files in /home/snb6/palmer_scratch/plume/:")
    print("  - crimaldi_plume_full_15Hz.mp4 (4 minutes @ 15 fps)")
    print("  - smoke_plume_full_60Hz.mp4 (1 minute @ 60 fps)")
    print("\nThese movies show the complete datasets at their native frame rates")
    print("with intensity ranges from the metadata files.")

if __name__ == "__main__":
    main()
