#!/usr/bin/env python3
"""Comprehensive analysis of test MAT files with statistics."""
import glob
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

print("=== COMPREHENSIVE TEST DATA ANALYSIS ===")

# Find all test MAT files
mat_files = glob.glob("test_output/**/*.mat", recursive=True)
print(f"Found {len(mat_files)} MAT files")

def try_extract_data(data, mat_file):
    """Try various methods to extract trajectory data."""
    x = y = None
    method = "unknown"
    
    # Method 1: 'out' structure
    if 'out' in data:
        try:
            if hasattr(data['out'], 'dtype') and 'x' in data['out'].dtype.names:
                x = data['out']['x'][0,0]
                y = data['out']['y'][0,0]
                method = "out structure"
        except:
            pass
    
    # Method 2: 'result' dictionary
    if x is None and 'result' in data:
        try:
            if isinstance(data['result'], dict) and 'x' in data['result']:
                x = data['result']['x']
                y = data['result']['y']
                method = "result dict"
        except:
            pass
    
    # Method 3: Direct x,y
    if x is None and 'x' in data and 'y' in data:
        x = data['x']
        y = data['y']
        method = "direct x,y"
    
    # Method 4: Look for any struct with x,y
    if x is None:
        for key, val in data.items():
            if key.startswith('__'):
                continue
            try:
                if hasattr(val, 'dtype') and val.dtype.names and 'x' in val.dtype.names:
                    x = val['x'][0,0] if val.ndim > 0 else val['x']
                    y = val['y'][0,0] if val.ndim > 0 else val['y']
                    method = f"{key} structure"
                    break
            except:
                pass
    
    return x, y, method

# Analyze each file
results = []
for mat_file in mat_files:
    print(f"\nAnalyzing: {mat_file}")
    result = {
        'file': mat_file,
        'name': Path(mat_file).stem,
        'dir': Path(mat_file).parent.name,
        'has_data': False
    }
    
    try:
        data = loadmat(mat_file)
        
        # Show available keys
        keys = [k for k in data.keys() if not k.startswith('__')]
        print(f"  Keys: {keys[:5]}...")  # Show first 5 keys
        
        # Try to extract trajectory data
        x, y, method = try_extract_data(data, mat_file)
        
        if x is not None and y is not None:
            # Ensure arrays
            x = np.asarray(x)
            y = np.asarray(y)
            
            print(f"  ✓ Found trajectory data via {method}: shape {x.shape}")
            
            result.update({
                'x': x,
                'y': y,
                'has_data': True,
                'method': method,
                'shape': x.shape,
                'n_trials': x.shape[1] if x.ndim > 1 else 1,
                'n_timesteps': x.shape[0] if x.ndim > 0 else len(x)
            })
            
            # Calculate basic statistics
            if x.size > 0:
                result['x_range'] = (float(np.min(x)), float(np.max(x)))
                result['y_range'] = (float(np.min(y)), float(np.max(y)))
                result['total_distance'] = float(np.sum(np.sqrt(np.diff(x.flat)**2 + np.diff(y.flat)**2)))
        else:
            print(f"  ✗ No trajectory data found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result['error'] = str(e)
    
    results.append(result)

# Summary statistics
successful = [r for r in results if r['has_data']]
failed = [r for r in results if not r['has_data']]

print(f"\n=== SUMMARY ===")
print(f"Successfully loaded: {len(successful)}/{len(mat_files)} files")
print(f"Failed: {len(failed)} files")

if failed:
    print("\nFailed files:")
    for r in failed:
        print(f"  - {r['name']}: {r.get('error', 'No trajectory data')}")

# Create comprehensive visualization
if successful:
    print("\nCreating comprehensive visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 1. All trajectories overlaid
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    colors = plt.cm.tab10(np.linspace(0, 1, len(successful)))
    
    for i, r in enumerate(successful):
        x = r['x']
        y = r['y']
        
        # Plot based on data size
        if x.size < 10:  # Very short trajectory
            ax1.plot(x.flat, y.flat, 'o-', alpha=0.7, linewidth=2, 
                    markersize=8, color=colors[i], label=r['name'])
        else:
            # Plot first trial/trajectory
            if x.ndim == 2 and x.shape[1] > 0:
                ax1.plot(x[:, 0], y[:, 0], alpha=0.7, linewidth=1.5, 
                        color=colors[i], label=r['name'])
            else:
                ax1.plot(x.flat[:1000], y.flat[:1000], alpha=0.7, linewidth=1.5, 
                        color=colors[i], label=r['name'])
    
    ax1.plot(0, 0, 'r*', markersize=20, label='Source', zorder=10)
    ax1.set_xlabel('X (cm)')
    ax1.set_ylabel('Y (cm)')
    ax1.set_title('All Test Trajectories')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax1.set_aspect('equal')
    
    # 2. Test statistics bar chart
    ax2 = fig.add_subplot(gs[0, 2])
    test_names = [r['name'][:10] for r in successful]  # Truncate long names
    timesteps = [r['n_timesteps'] for r in successful]
    
    bars = ax2.bar(range(len(test_names)), timesteps, color=colors)
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.set_ylabel('Timesteps')
    ax2.set_title('Trajectory Lengths')
    ax2.set_yscale('log')  # Log scale for better visibility
    
    # 3. Distance traveled
    ax3 = fig.add_subplot(gs[1, 2])
    distances = [r.get('total_distance', 0) for r in successful]
    
    bars = ax3.bar(range(len(test_names)), distances, color=colors)
    ax3.set_xticks(range(len(test_names)))
    ax3.set_xticklabels(test_names, rotation=45, ha='right')
    ax3.set_ylabel('Total Distance (cm)')
    ax3.set_title('Distance Traveled')
    
    # 4. Individual trajectory panels
    n_panels = min(6, len(successful))
    for i in range(n_panels):
        ax = fig.add_subplot(gs[2, i % 3])
        r = successful[i]
        x = r['x']
        y = r['y']
        
        if x.size < 10:
            ax.plot(x.flat, y.flat, 'bo-', markersize=8, linewidth=2)
        else:
            if x.ndim == 2:
                ax.plot(x[:, 0], y[:, 0], 'b-', alpha=0.8, linewidth=1.5)
            else:
                ax.plot(x, y, 'b-', alpha=0.8, linewidth=1.5)
        
        ax.plot(0, 0, 'r*', markersize=15)
        ax.set_title(f"{r['name']}\n({r['n_timesteps']} steps)")
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('test_analysis_comprehensive.png', dpi=150, bbox_inches='tight')
    print("✓ Saved test_analysis_comprehensive.png")
    plt.close()
    
    # Create detailed report
    with open('test_analysis_report.txt', 'w') as f:
        f.write("COMPREHENSIVE TEST DATA ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total MAT files found: {len(mat_files)}\n")
        f.write(f"Successfully loaded: {len(successful)}\n")
        f.write(f"Failed to load: {len(failed)}\n\n")
        
        f.write("SUCCESSFUL TESTS:\n")
        f.write("-" * 40 + "\n")
        for r in successful:
            f.write(f"\n{r['name']}:\n")
            f.write(f"  File: {r['file']}\n")
            f.write(f"  Method: {r['method']}\n")
            f.write(f"  Shape: {r['shape']}\n")
            f.write(f"  Trials: {r['n_trials']}\n")
            f.write(f"  Timesteps: {r['n_timesteps']}\n")
            f.write(f"  X range: {r['x_range'][0]:.2f} to {r['x_range'][1]:.2f}\n")
            f.write(f"  Y range: {r['y_range'][0]:.2f} to {r['y_range'][1]:.2f}\n")
            f.write(f"  Total distance: {r['total_distance']:.2f} cm\n")
        
        if failed:
            f.write("\n\nFAILED TESTS:\n")
            f.write("-" * 40 + "\n")
            for r in failed:
                f.write(f"\n{r['name']}:\n")
                f.write(f"  File: {r['file']}\n")
                f.write(f"  Error: {r.get('error', 'No trajectory data found')}\n")
    
    print("✓ Saved test_analysis_report.txt")

print("\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("  - test_trajectories.png (simple overlay)")
print("  - test_summary.txt (basic summary)")
print("  - test_analysis_comprehensive.png (detailed visualization)")
print("  - test_analysis_report.txt (detailed report)")