#!/usr/bin/env python3
"""
view_results.py - View and analyze navigation model results

Usage: python view_results.py [results_file.mat]
Default: results/nav_results_0000.mat

Creates visualizations and analysis of navigation model output.
Requires: numpy, scipy, matplotlib (install via conda environment)
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
import os

def load_results(filename='results/nav_results_0000.mat'):
    """Load MATLAB results file"""
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        sys.exit(1)
    
    # Load the .mat file
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return data['out']

def analyze_results(out):
    """Analyze and print key metrics"""
    print("\n=== Navigation Model Results ===")
    print(f"Environment: {out.environment}")
    
    # Handle both single agent and multiple agents
    if out.x.ndim == 1:
        n_agents = 1
        n_samples = len(out.x)
    else:
        n_samples, n_agents = out.x.shape
    
    print(f"Number of agents: {n_agents}")
    print(f"Trajectory length: {n_samples} samples")
    print(f"Duration: {n_samples/15:.1f} seconds (at 15 Hz)")
    
    # Success metrics
    if hasattr(out, 'successrate'):
        print(f"\nSuccess rate: {out.successrate*100:.1f}%")
        
    if hasattr(out, 'latency'):
        # Handle single value or array
        if np.isscalar(out.latency):
            latencies = [out.latency] if not np.isnan(out.latency) else []
        else:
            latencies = out.latency[~np.isnan(out.latency)]
        
        if len(latencies) > 0:
            print(f"Successful agents: {len(latencies)}/{n_agents}")
            print(f"Mean time to target: {np.mean(latencies):.1f} seconds")
            print(f"Fastest: {np.min(latencies):.1f}s, Slowest: {np.max(latencies):.1f}s")
    
    # Trajectory statistics
    if out.x.ndim == 1:
        # Single agent
        distances = np.sum(np.sqrt(np.diff(out.x)**2 + np.diff(out.y)**2))
        print(f"\nDistance traveled: {distances:.1f} cm")
    else:
        # Multiple agents
        distances = []
        for i in range(n_agents):
            d = np.sum(np.sqrt(np.diff(out.x[:, i])**2 + np.diff(out.y[:, i])**2))
            distances.append(d)
        print(f"\nMean distance traveled: {np.mean(distances):.1f} cm")
        print(f"Range: [{np.min(distances):.1f}, {np.max(distances):.1f}] cm")

def plot_trajectories(out):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. All trajectories
    ax = axes[0, 0]
    if out.x.ndim == 1:
        ax.plot(out.x, out.y, 'k-', alpha=0.5)
    else:
        for i in range(out.x.shape[1]):
            ax.plot(out.x[:, i], out.y[:, i], 'gray', alpha=0.3)
    
    # Mark target and success zone
    ax.plot(0, 0, 'r*', markersize=15)
    circle = plt.Circle((0, 0), 2, fill=False, color='red', linestyle='--')
    ax.add_patch(circle)
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')
    ax.set_title('Agent Trajectories')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Starting positions
    ax = axes[0, 1]
    if out.x.ndim == 1:
        start_x, start_y = out.x[0], out.y[0]
        ax.scatter(start_x, start_y, c='blue', s=50)
    else:
        start_x = out.x[0, :]
        start_y = out.y[0, :]
        
        if hasattr(out, 'latency'):
            # Color by success
            if np.isscalar(out.latency):
                colors = ['green' if not np.isnan(out.latency) else 'red']
            else:
                colors = ['green' if not np.isnan(l) else 'red' for l in out.latency]
            ax.scatter(start_x, start_y, c=colors, s=50, alpha=0.6)
            ax.legend(['Successful', 'Failed'])
        else:
            ax.scatter(start_x, start_y, c='blue', s=50, alpha=0.6)
    
    ax.set_xlabel('Starting X (cm)')
    ax.set_ylabel('Starting Y (cm)')
    ax.set_title('Initial Positions')
    ax.grid(True, alpha=0.3)
    
    # 3. Example trajectory with odor
    ax = axes[1, 0]
    idx = 0  # First agent
    if out.x.ndim == 1:
        x_traj, y_traj = out.x, out.y
    else:
        x_traj, y_traj = out.x[:, idx], out.y[:, idx]
    
    ax.plot(x_traj, y_traj, 'k-', linewidth=1.5)
    ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start')
    ax.plot(x_traj[-1], y_traj[-1], 'ro', markersize=8, label='End')
    
    # Show odor encounters if available
    if hasattr(out, 'odor'):
        odor_data = out.odor[:, idx] if out.odor.ndim > 1 else out.odor
        odor_idx = np.where(odor_data > 0.01)[0]
        if len(odor_idx) > 0:
            ax.scatter(x_traj[odor_idx], y_traj[odor_idx], 
                      c='magenta', s=20, alpha=0.5, label='Odor')
    
    ax.set_xlabel('X position (cm)')
    ax.set_ylabel('Y position (cm)')
    ax.set_title(f'Example Trajectory (Agent {idx+1})')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # 4. Success rate visualization or time series
    ax = axes[1, 1]
    if hasattr(out, 'odor') and hasattr(out, 'ON'):
        time = np.arange(len(out.odor)) / 15  # Convert to seconds
        odor_data = out.odor[:, idx] if out.odor.ndim > 1 else out.odor
        on_data = out.ON[:, idx] if out.ON.ndim > 1 else out.ON
        
        ax.plot(time, odor_data, 'k-', label='Odor', alpha=0.7)
        ax.plot(time, on_data, 'm-', label='ON response', linewidth=2)
        if hasattr(out, 'OFF'):
            off_data = out.OFF[:, idx] if out.OFF.ndim > 1 else out.OFF
            ax.plot(time, off_data, 'c-', label='OFF response', linewidth=2)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Response')
        ax.set_title('Sensory Responses')
        ax.legend()
        ax.grid(True, alpha=0.3)
    elif hasattr(out, 'latency') and not np.isscalar(out.latency):
        # Latency histogram
        valid_latencies = out.latency[~np.isnan(out.latency)]
        if len(valid_latencies) > 0:
            ax.hist(valid_latencies, bins=15, alpha=0.7, color='blue')
            ax.axvline(np.mean(valid_latencies), color='red', 
                      linestyle='--', label=f'Mean: {np.mean(valid_latencies):.1f}s')
            ax.set_xlabel('Time to reach target (seconds)')
            ax.set_ylabel('Number of agents')
            ax.set_title('Success Time Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    # Get filename from command line or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'results/nav_results_0000.mat'
    
    print(f"Loading: {filename}")
    
    # Load and analyze
    out = load_results(filename)
    analyze_results(out)
    
    # Create plots
    fig = plot_trajectories(out)
    
    # Save plots
    plot_file = filename.replace('.mat', '_plots.png')
    fig.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlots saved to: {plot_file}")
    
    plt.show()

if __name__ == '__main__':
    main()