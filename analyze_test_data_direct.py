#!/usr/bin/env python3
"""Direct analysis of test MAT files."""
import glob
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

print("=== ANALYZING TEST MAT FILES ===")

# Find all test MAT files
mat_files = glob.glob("test_output/**/*.mat", recursive=True)
print(f"Found {len(mat_files)} MAT files")

if not mat_files:
    print("No MAT files found!")
    exit(1)

# Analyze each file
results = []
for mat_file in mat_files:
    print(f"\nAnalyzing: {mat_file}")
    try:
        data = loadmat(mat_file)
        
        # Find trajectory data
        x = y = None
        if 'out' in data:
            if hasattr(data['out'], 'dtype') and 'x' in data['out'].dtype.names:
                x = data['out']['x'][0,0]
                y = data['out']['y'][0,0]
        elif 'result' in data and 'x' in data['result']:
            x = data['result']['x']
            y = data['result']['y']
        elif 'x' in data and 'y' in data:
            x = data['x']
            y = data['y']
        
        if x is not None:
            print(f"  ✓ Found trajectory data: shape {x.shape}")
            results.append({
                'file': mat_file,
                'name': Path(mat_file).stem,
                'x': x,
                'y': y,
                'has_data': True
            })
        else:
            print(f"  ✗ No trajectory data found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Create a simple plot
if results:
    print("\nCreating visualization...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(results):
        if r['has_data']:
            x = r['x']
            y = r['y']
            
            # Plot first trial/trajectory
            if x.ndim == 2:
                ax.plot(x[:, 0], y[:, 0], alpha=0.7, linewidth=2, 
                       color=colors[i], label=r['name'])
            else:
                ax.plot(x, y, alpha=0.7, linewidth=2, 
                       color=colors[i], label=r['name'])
    
    ax.plot(0, 0, 'r*', markersize=15, label='Source')
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_title('Test Trajectories')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('test_trajectories.png', dpi=150)
    print("✓ Saved test_trajectories.png")
    plt.close()
    
    # Create summary
    with open('test_summary.txt', 'w') as f:
        f.write("TEST DATA SUMMARY\n")
        f.write("=================\n\n")
        f.write(f"Total files: {len(mat_files)}\n")
        f.write(f"Files with data: {sum(r['has_data'] for r in results)}\n\n")
        
        for r in results:
            if r['has_data']:
                f.write(f"{r['name']}: ")
                if r['x'].ndim == 2:
                    f.write(f"{r['x'].shape[1]} trials, {r['x'].shape[0]} timesteps\n")
                else:
                    f.write(f"1 trial, {len(r['x'])} timesteps\n")
    
    print("✓ Saved test_summary.txt")
else:
    print("No trajectory data found in any files!")

print("\nDone!")
