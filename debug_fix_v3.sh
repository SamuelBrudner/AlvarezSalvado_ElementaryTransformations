#!/bin/bash
# Debug and fix the issues

echo "=== 1. Check what's actually in the smoke HDF5 file ==="
if [ -f "data/smoke_1a_orig_backgroundsubtracted.h5" ]; then
    conda run --prefix ./dev_env python3 -c "
import h5py
with h5py.File('data/smoke_1a_orig_backgroundsubtracted.h5', 'r') as f:
    print('Datasets in smoke HDF5:', list(f.keys()))
    for key in f.keys():
        print(f'  {key}: shape = {f[key].shape if hasattr(f[key], \"shape\") else \"N/A\"}')"
else
    echo "Smoke HDF5 file not found at expected location"
    echo "Looking for smoke files..."
    find data/ -name "*smoke*.h5" -o -name "*smoke*.hdf5" | head -10
fi

echo -e "\n=== 2. Check Crimaldi HDF5 dimensions ==="
conda run --prefix ./dev_env python3 -c "
import h5py
with h5py.File('data/10302017_10cms_bounded.hdf5', 'r') as f:
    print('Crimaldi dataset2 shape:', f['dataset2'].shape)
    print('This means: height={}, width={}, frames={}'.format(*f['dataset2'].shape))"

echo -e "\n=== 3. Fix load_plume_hdf5.m to handle different dataset names ==="
cat > fix_load_plume_hdf5.patch << 'EOF'
--- Code/load_plume_hdf5.m.orig
+++ Code/load_plume_hdf5.m
@@ -16,7 +16,22 @@
     frame_rate (1,1) double
 end
 
-data = h5read(filename, '/dataset2');
+% Try to find the dataset name dynamically
+info = h5info(filename);
+dataset_name = '';
+for i = 1:length(info.Datasets)
+    name = info.Datasets(i).Name;
+    % Look for dataset1, dataset2, or anything with 'data' in the name
+    if strcmp(name, 'dataset2') || strcmp(name, 'dataset1') || contains(lower(name), 'data')
+        dataset_name = ['/' name];
+        break;
+    end
+end
+
+if isempty(dataset_name)
+    error('No suitable dataset found in %s. Available: %s', filename, strjoin({info.Datasets.Name}, ', '));
+end
+
+data = h5read(filename, dataset_name);
 plume.data = double(data);
 plume.px_per_mm = px_per_mm;
 plume.frame_rate = frame_rate;
EOF

# Apply the patch
cp Code/load_plume_hdf5.m Code/load_plume_hdf5.m.bak
patch -p0 < fix_load_plume_hdf5.patch || echo "Patch failed, applying manually..."

# If patch fails, apply manually
if ! grep -q "h5info" Code/load_plume_hdf5.m; then
    cat > Code/load_plume_hdf5.m << 'EOF'
function plume = load_plume_hdf5(filename, px_per_mm, frame_rate)
%LOAD_PLUME_HDF5 Load plume data from an HDF5 file.
%   PLUME = LOAD_PLUME_HDF5(FILENAME, PX_PER_MM, FRAME_RATE) reads the
%   dataset from the specified HDF5 file. It will try to find dataset2,
%   dataset1, or any dataset containing 'data' in its name. The returned
%   structure contains fields:
%       data       - numeric array (height x width x frames)
%       px_per_mm  - pixels per millimeter
%       frame_rate - frames per second
%
%   Example:
%       plume = load_plume_hdf5('plume.h5', 20, 50);

arguments
    filename (1,:) char
    px_per_mm (1,1) double
    frame_rate (1,1) double
end

% Try to find the dataset name dynamically
info = h5info(filename);
dataset_name = '';
for i = 1:length(info.Datasets)
    name = info.Datasets(i).Name;
    % Look for dataset2, dataset1, or anything with 'data' in the name
    if strcmp(name, 'dataset2') || strcmp(name, 'dataset1') || contains(lower(name), 'data')
        dataset_name = ['/' name];
        fprintf('Found dataset: %s\n', dataset_name);
        break;
    end
end

if isempty(dataset_name)
    error('No suitable dataset found in %s. Available: %s', filename, strjoin({info.Datasets.Name}, ', '));
end

data = h5read(filename, dataset_name);
plume.data = double(data);
plume.px_per_mm = px_per_mm;
plume.frame_rate = frame_rate;
end
EOF
fi

echo -e "\n=== 4. Check the initial positions in navigation model ==="
# The error might be due to initial positions being out of bounds
grep -A5 "gaussian distribution" Code/navigation_model_vec.m
grep -A5 "rand(1,ntrials)" Code/navigation_model_vec.m | grep -E "x\(1,:\)|y\(1,:\)"

echo -e "\n=== 5. Update the Crimaldi config to use correct ntrials ==="
cat > configs/batch_crimaldi.yaml << 'EOF'
# Crimaldi plume configuration for batch processing
environment: Crimaldi
plotting: 0
ntrials: 1  # Should be 1 for batch runs, not 1000
bilateral: false
triallength: 3600
EOF

cat > configs/batch_smoke_hdf5.yaml << 'EOF'
# Smoke plume configuration for batch processing (HDF5)
environment: video
plume_metadata: data/smoke_hdf5_meta.yaml  # Use relative path
plotting: 0
ntrials: 1  # Should be 1 for batch runs
bilateral: false
triallength: 3600
EOF

echo -e "\n=== 6. Update smoke metadata to use correct path ==="
# First check where the actual HDF5 file is
SMOKE_H5=$(find . -name "smoke*.h5" -o -name "smoke*.hdf5" | grep -v ".git" | head -1)
if [ -n "$SMOKE_H5" ]; then
    echo "Found smoke HDF5 at: $SMOKE_H5"
    SMOKE_DIR=$(dirname "$SMOKE_H5")
    SMOKE_FILE=$(basename "$SMOKE_H5")
    
    cat > data/smoke_hdf5_meta.yaml << EOF
# Metadata for smoke HDF5 plume
output_directory: $SMOKE_DIR
output_filename: $SMOKE_FILE
output_h5: $SMOKE_FILE
vid_mm_per_px: 0.1530  # 1/6.536
fps: 60
scaled_to_crim: true
EOF
    echo "Updated smoke_hdf5_meta.yaml"
else
    echo "WARNING: No smoke HDF5 file found!"
fi

echo -e "\n=== Files have been updated. Ready to rerun test_batch_v3 ==="