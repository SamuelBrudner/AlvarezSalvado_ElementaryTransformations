import h5py
import numpy as np

print("Converting smoke plume to Crimaldi format (frames, height, width)...")

with h5py.File('data/smoke_1a_rotated_3d.h5', 'r') as f:
    data = f['/dataset2'][:]
    print(f"Current smoke shape: {data.shape} (height, width, frames)")

# Transpose from (height, width, frames) to (frames, height, width)
data_transposed = np.transpose(data, (2, 0, 1))
print(f"New smoke shape: {data_transposed.shape} (frames, height, width)")

# Save the converted version
output_path = '/home/snb6/palmer_scratch/plume/smoke_1a_crimaldi_format.h5'
with h5py.File(output_path, 'w') as f:
    f.create_dataset('/dataset2', data=data_transposed.astype(np.float32))

# Update symlink
import os
if os.path.exists('data/smoke_1a_rotated_3d.h5'):
    os.unlink('data/smoke_1a_rotated_3d.h5')
os.symlink(output_path, 'data/smoke_1a_rotated_3d.h5')

print("✓ Smoke plume converted to Crimaldi format!")
