#!/usr/bin/env python3
"""Convert Crimaldi HDF5 from (frames, height, width) to (height, width, frames) format."""

import h5py
import numpy as np
from pathlib import Path
import shutil

def convert_crimaldi_to_standard_format():
    """Convert the Crimaldi HDF5 to standard (height, width, frames) format."""
    
    # Paths
    input_path = Path("data/10302017_10cms_bounded.hdf5")
    backup_path = Path("data/10302017_10cms_bounded_original.hdf5")
    
    print("Converting Crimaldi HDF5 to standard format...")
    
    # First, make a backup
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}...")
        shutil.copy2(input_path, backup_path)
    
    # Read the data
    print("Reading original data...")
    with h5py.File(input_path, 'r') as f:
        data_old = f['/dataset2'][:]
        print(f"Original shape: {data_old.shape} (frames, height, width)")
    
    # Transpose to (height, width, frames)
    print("Transposing to (height, width, frames)...")
    data_new = np.transpose(data_old, (1, 2, 0))
    print(f"New shape: {data_new.shape}")
    
    # Create new file with corrected format
    temp_path = Path("data/10302017_10cms_bounded_fixed.hdf5")
    print(f"Writing corrected data to temporary file...")
    with h5py.File(temp_path, 'w') as f:
        dset = f.create_dataset('/dataset2', data=data_new.astype(np.float32))
        # Add attributes
        f.attrs['height'] = data_new.shape[0]
        f.attrs['width'] = data_new.shape[1] 
        f.attrs['frames'] = data_new.shape[2]
    
    # Replace original with fixed version
    print("Replacing original with fixed version...")
    import os
    os.replace(temp_path, input_path)
    
    # Verify the fix
    print("\nVerifying the conversion...")
    with h5py.File(input_path, 'r') as f:
        data_check = f['/dataset2']
        print(f"Final shape: {data_check.shape}")
        
        # Check middle frame
        middle_idx = data_check.shape[2] // 2
        middle_frame = data_check[:, :, middle_idx]
        print(f"Middle frame shape: {middle_frame.shape}")
        print(f"Middle frame range: [{middle_frame.min():.6f}, {middle_frame.max():.6f}]")
    
    # Create a verification plot
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show middle frame
    with h5py.File(input_path, 'r') as f:
        middle_frame = f['/dataset2'][:, :, middle_idx]
    
    im1 = ax1.imshow(middle_frame, cmap='hot', aspect='auto')
    ax1.set_title(f'Crimaldi Fixed - Frame {middle_idx}')
    ax1.set_xlabel(f'Width: {middle_frame.shape[1]}')
    ax1.set_ylabel(f'Height: {middle_frame.shape[0]}')
    plt.colorbar(im1, ax=ax1)
    
    # Log scale
    log_frame = np.log10(middle_frame + 1e-10)
    im2 = ax2.imshow(log_frame, cmap='hot', aspect='auto')
    ax2.set_title(f'Crimaldi Fixed - Frame {middle_idx} (Log10)')
    ax2.set_xlabel(f'Width: {middle_frame.shape[1]}')
    ax2.set_ylabel(f'Height: {middle_frame.shape[0]}')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('crimaldi_fixed_verification.png', dpi=150)
    print("\nSaved verification plot to crimaldi_fixed_verification.png")
    
    print("\n✓ Crimaldi HDF5 successfully converted to (height, width, frames) format!")
    print(f"  Backup saved at: {backup_path}")

if __name__ == "__main__":
    convert_crimaldi_to_standard_format()
