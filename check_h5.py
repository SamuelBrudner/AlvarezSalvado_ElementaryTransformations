import h5py

filename = 'data/10302017_10cms_bounded.hdf5'
print(f"Checking {filename}...")

with h5py.File(filename, 'r') as f:
    print("\nDatasets in file:")
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    
    f.visititems(print_structure)
    
    print("\nTop-level keys:")
    for key in f.keys():
        print(f"  {key}")
        
    # Try to access the first dataset
    if len(f.keys()) > 0:
        first_key = list(f.keys())[0]
        print(f"\nFirst dataset is: '{first_key}'")
        print(f"Shape: {f[first_key].shape}")
