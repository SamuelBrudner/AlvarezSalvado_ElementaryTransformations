#!/usr/bin/env python3
"""Diagnose where the smoke data transformation goes wrong."""

import sys
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our utilities
from Code.video_intensity import extract_intensities_from_video
from Code.plume_utils import get_plume_frame

def load_h5_data(path, dataset_name='/dataset1'):
    """Load HDF5 data and attributes."""
    with h5py.File(path, 'r') as f:
        data = f[dataset_name][:]
        attrs = dict(f[dataset_name].attrs)
        root_attrs = dict(f.attrs)
        attrs.update(root_attrs)
    return data, attrs

def try_reshape_orders(data_1d, height, width, frames):
    """Try different reshape orders to find the correct one."""
    reshapes = {}
    
    # Order 1: [frames, height, width] (C-style, used by plume_pipeline)
    try:
        data_3d = data_1d.reshape((frames, height, width))
        reshapes['F,H,W'] = data_3d
    except:
        reshapes['F,H,W'] = None
    
    # Order 2: [height, width, frames] (MATLAB natural)
    try:
        data_3d = data_1d.reshape((height, width, frames))
        reshapes['H,W,F'] = data_3d
    except:
        reshapes['H,W,F'] = None
    
    # Order 3: [width, height, frames] transposed
    try:
        data_3d = data_1d.reshape((width, height, frames))
        reshapes['W,H,F'] = data_3d
    except:
        reshapes['W,H,F'] = None
    
    return reshapes

def main():
    print("Diagnosing smoke data pipeline...\n")
    
    # Paths
    avi_path = "data/smoke_1a_orig_backgroundsubtracted.avi"
    base_path = Path("/home/snb6/palmer_scratch/plume")
    
    files_to_check = [
        ("raw", "smoke_1a_orig_backgroundsubtracted_raw.h5"),
        ("scaled", "smoke_1a_orig_backgroundsubtracted_scaled.h5"),
        ("rotated", "smoke_1a_orig_backgroundsubtracted_rotated.h5")
    ]
    
    # Step 1: Load a frame from the original AVI
    print("=== STEP 1: Original AVI ===")
    try:
        # Use imageio to read a frame
        import imageio.v3 as iio
        frames = list(iio.imiter(avi_path))
        print(f"AVI has {len(frames)} frames")
        print(f"Frame shape: {frames[0].shape}")
        
        middle_idx = len(frames) // 2
        avi_frame = frames[middle_idx]
        if avi_frame.ndim == 3:
            avi_frame = np.mean(avi_frame, axis=2)  # Convert to grayscale
        avi_frame = avi_frame.astype(float) / 255.0
        print(f"Middle frame index: {middle_idx}")
        print(f"Frame range: [{avi_frame.min():.3f}, {avi_frame.max():.3f}]")
    except Exception as e:
        print(f"Error loading AVI: {e}")
        avi_frame = None
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(4, 4, figure=fig)
    
    # Plot original AVI frame
    if avi_frame is not None:
        ax = fig.add_subplot(gs[0, :2])
        im = ax.imshow(avi_frame, cmap='gray', aspect='auto')
        ax.set_title(f'Original AVI - Frame {middle_idx}')
        ax.set_xlabel(f'Width: {avi_frame.shape[1]}')
        ax.set_ylabel(f'Height: {avi_frame.shape[0]}')
        plt.colorbar(im, ax=ax)
    
    # Step 2: Check each HDF5 file
    print("\n=== STEP 2: Checking HDF5 files ===")
    
    for idx, (name, filename) in enumerate(files_to_check):
        filepath = base_path / filename
        print(f"\nChecking {name}: {filename}")
        
        try:
            data, attrs = load_h5_data(filepath)
            print(f"  Dataset shape: {data.shape}")
            print(f"  Data range: [{data.min():.3f}, {data.max():.3f}]")
            
            # Get dimensions from attributes
            height = attrs.get('height', None)
            width = attrs.get('width', None)
            frames = attrs.get('frames', None)
            
            print(f"  Attributes: height={height}, width={width}, frames={frames}")
            
            if data.ndim == 1 and all(x is not None for x in [height, width, frames]):
                # Try different reshape orders
                middle_frame_idx = frames // 2
                reshapes = try_reshape_orders(data, height, width, frames)
                
                # Plot each reshape attempt
                for j, (order, data_3d) in enumerate(reshapes.items()):
                    if data_3d is not None:
                        try:
                            if order == 'F,H,W':
                                frame = data_3d[middle_frame_idx, :, :]
                            elif order == 'H,W,F':
                                frame = data_3d[:, :, middle_frame_idx]
                            elif order == 'W,H,F':
                                frame = data_3d[:, :, middle_frame_idx].T
                            
                            ax = fig.add_subplot(gs[idx+1, j])
                            im = ax.imshow(frame, cmap='hot', aspect='auto')
                            ax.set_title(f'{name} - {order}')
                            ax.set_xlabel(f'{frame.shape[1]}')
                            ax.set_ylabel(f'{frame.shape[0]}')
                            plt.colorbar(im, ax=ax)
                        except Exception as e:
                            print(f"    Error with {order}: {e}")
            
            elif data.ndim == 3:
                # Already 3D
                middle_frame_idx = data.shape[2] // 2
                frame = data[:, :, middle_frame_idx]
                
                ax = fig.add_subplot(gs[idx+1, 0])
                im = ax.imshow(frame, cmap='hot', aspect='auto')
                ax.set_title(f'{name} - 3D data')
                plt.colorbar(im, ax=ax)
                
        except Exception as e:
            print(f"  Error: {e}")
    
    plt.suptitle('Smoke Pipeline Diagnostic - Middle Frames', fontsize=14)
    plt.tight_layout()
    plt.savefig('smoke_pipeline_diagnostic_py.png', dpi=150)
    print("\nSaved diagnostic plot to smoke_pipeline_diagnostic_py.png")
    
    # Step 3: Test rotation manually
    print("\n=== STEP 3: Testing rotation ===")
    
    # Load scaled data and test rotation
    scaled_path = base_path / "smoke_1a_orig_backgroundsubtracted_scaled.h5"
    data, attrs = load_h5_data(scaled_path)
    
    if data.ndim == 1:
        height, width, frames = 1088, 1728, 3600
        middle_idx = frames // 2
        
        # Use the correct reshape order (from plume_pipeline.py)
        data_3d = data.reshape((frames, height, width))
        frame = data_3d[middle_idx, :, :]
        
        # Test rotations
        fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(frame, cmap='hot')
        axes[0].set_title('Original scaled frame')
        axes[0].set_xlabel(f'Width: {frame.shape[1]}')
        axes[0].set_ylabel(f'Height: {frame.shape[0]}')
        
        # Rotate 90 clockwise
        frame_rot90_cw = np.rot90(frame, k=-1)
        axes[1].imshow(frame_rot90_cw, cmap='hot')
        axes[1].set_title('Rotated 90° CW (k=-1)')
        axes[1].set_xlabel(f'Width: {frame_rot90_cw.shape[1]}')
        axes[1].set_ylabel(f'Height: {frame_rot90_cw.shape[0]}')
        
        # Rotate 90 counter-clockwise
        frame_rot90_ccw = np.rot90(frame, k=1)
        axes[2].imshow(frame_rot90_ccw, cmap='hot')
        axes[2].set_title('Rotated 90° CCW (k=1)')
        axes[2].set_xlabel(f'Width: {frame_rot90_ccw.shape[1]}')
        axes[2].set_ylabel(f'Height: {frame_rot90_ccw.shape[0]}')
        
        plt.tight_layout()
        plt.savefig('smoke_rotation_test_py.png', dpi=150)
        print("Saved rotation test to smoke_rotation_test_py.png")

if __name__ == "__main__":
    main()
