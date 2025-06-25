#!/usr/bin/env python3
"""
view_results.py - View and analyze navigation model results

Usage: python view_results.py [results_file.mat]
Default: results/nav_results_0000.mat

Creates visualizations and analysis of navigation model output.
Requires: numpy, scipy, matplotlib, loguru (install via conda environment)
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from loguru import logger

# Configure loguru for structured logging to logs/ directory
def setup_logging():
    """Configure loguru logging to output to logs/ directory with timestamps"""
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Remove default handler and add our custom one
    logger.remove()
    
    # Add structured logging to logs/ directory with timestamp-based files
    logger.add(
        "logs/view_results_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        retention="30 days",
        compression="gz"
    )
    
    # Also add console output for backwards compatibility
    logger.add(
        sys.stdout,
        format="{message}",
        level="INFO",
        filter=lambda record: record["extra"].get("console", True)
    )

# Initialize logging
setup_logging()

try:
    import h5py
except ImportError:
    logger.warning("h5py not installed. Cannot read MATLAB v7.3 files.")
    h5py = None

def load_results(filename='results/nav_results_0000.mat'):
    """Load MATLAB results file (handles both v5 and v7.3 formats)"""
    logger.info(f"Attempting to load results file: {filename}")
    
    if not os.path.exists(filename):
        logger.error(f"Results file not found: {filename}")
        sys.exit(1)
    
    # Try loading with scipy first (v5 format)
    try:
        logger.debug("Attempting to load with scipy.io (MATLAB v5 format)")
        data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
        logger.info("Successfully loaded MATLAB v5 format file")
        return data['out']
    except NotImplementedError:
        # Fall back to h5py for v7.3 files
        logger.info("Detected MATLAB v7.3 file, using h5py...")
        
        if h5py is None:
            logger.error("h5py is required for MATLAB v7.3 files but not installed")
            sys.exit(1)
            
        import h5py
        
        # Custom object to mimic MATLAB struct
        class Struct:
            pass
        
        out = Struct()
        
        try:
            with h5py.File(filename, 'r') as f:
                logger.debug("Opening HDF5 file and reading 'out' structure")
                # Get the 'out' structure
                out_group = f['out']
                
                # Load each field
                for key in out_group.keys():
                    data = out_group[key]
                    if isinstance(data, h5py.Dataset):
                        # Load the data
                        value = data[()]
                        
                        # Handle different data types
                        if value.dtype.type is np.bytes_:
                            # String data - decode it
                            if value.shape == ():
                                value = value.decode('utf-8')
                            else:
                                value = ''.join(chr(c) for c in value)
                        elif len(value.shape) > 1:
                            # Multi-dimensional array - transpose for MATLAB compatibility
                            value = value.T
                        
                        setattr(out, key, value)
                        logger.debug(f"Loaded field '{key}' with shape {getattr(value, 'shape', 'scalar')}")
            
            logger.info("Successfully loaded MATLAB v7.3 format file")
            return out
            
        except Exception as e:
            logger.error(f"Failed to load HDF5 file: {e}")
            sys.exit(1)

def analyze_results(out):
    """Analyze and print key metrics"""
    logger.info("Starting results analysis", extra={"console": False})
    
    # These messages should appear on console for user interface compatibility
    logger.info("\n=== Navigation Model Results ===")
    logger.info(f"Environment: {out.environment}")
    
    # Handle both single agent and multiple agents
    if out.x.ndim == 1:
        n_agents = 1
        n_samples = len(out.x)
    else:
        n_samples, n_agents = out.x.shape
    
    logger.info(f"Number of agents: {n_agents}")
    logger.info(f"Trajectory length: {n_samples} samples")
    logger.info(f"Duration: {n_samples/15:.1f} seconds (at 15 Hz)")
    
    # Log detailed analysis to file only
    logger.info(f"Analysis started for {n_agents} agents with {n_samples} trajectory samples", extra={"console": False})
    
    # Success metrics
    if hasattr(out, 'successrate'):
        logger.info(f"\nSuccess rate: {out.successrate*100:.1f}%")
        logger.info(f"Success rate calculated: {out.successrate*100:.1f}%", extra={"console": False})
        
    if hasattr(out, 'latency'):
        # Handle single value or array
        if np.isscalar(out.latency):
            latencies = [out.latency] if not np.isnan(out.latency) else []
        else:
            latencies = out.latency[~np.isnan(out.latency)]
        
        if len(latencies) > 0:
            logger.info(f"Successful agents: {len(latencies)}/{n_agents}")
            logger.info(f"Mean time to target: {np.mean(latencies):.1f} seconds")
            logger.info(f"Fastest: {np.min(latencies):.1f}s, Slowest: {np.max(latencies):.1f}s")
            
            # Detailed logging
            logger.info(f"Latency analysis: {len(latencies)} successful out of {n_agents} agents", extra={"console": False})
            logger.info(f"Latency statistics - Mean: {np.mean(latencies):.2f}s, Std: {np.std(latencies):.2f}s", extra={"console": False})
    
    # Trajectory statistics
    if out.x.ndim == 1:
        # Single agent
        distances = np.sum(np.sqrt(np.diff(out.x)**2 + np.diff(out.y)**2))
        logger.info(f"\nDistance traveled: {distances:.1f} cm")
        logger.info(f"Single agent trajectory distance: {distances:.2f} cm", extra={"console": False})
    else:
        # Multiple agents
        distances = []
        for i in range(n_agents):
            d = np.sum(np.sqrt(np.diff(out.x[:, i])**2 + np.diff(out.y[:, i])**2))
            distances.append(d)
        logger.info(f"\nMean distance traveled: {np.mean(distances):.1f} cm")
        logger.info(f"Range: [{np.min(distances):.1f}, {np.max(distances):.1f}] cm")
        
        # Detailed logging
        logger.info(f"Distance statistics for {len(distances)} agents - Mean: {np.mean(distances):.2f}, Std: {np.std(distances):.2f}", extra={"console": False})

def plot_trajectories(out):
    """Create visualization plots"""
    logger.info("Generating trajectory visualizations", extra={"console": False})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    logger.debug("Created 2x2 subplot figure for trajectory plots", extra={"console": False})
    
    # 1. All trajectories
    ax = axes[0, 0]
    if out.x.ndim == 1:
        ax.plot(out.x, out.y, 'k-', alpha=0.5)
        logger.debug("Plotted single agent trajectory", extra={"console": False})
    else:
        for i in range(out.x.shape[1]):
            ax.plot(out.x[:, i], out.y[:, i], 'gray', alpha=0.3)
        logger.debug(f"Plotted {out.x.shape[1]} agent trajectories", extra={"console": False})
    
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
        logger.debug(f"Plotted single agent start position: ({start_x:.2f}, {start_y:.2f})", extra={"console": False})
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
            
            success_count = sum(1 for c in colors if c == 'green')
            logger.debug(f"Plotted starting positions: {success_count} successful, {len(colors)-success_count} failed", extra={"console": False})
        else:
            ax.scatter(start_x, start_y, c='blue', s=50, alpha=0.6)
            logger.debug(f"Plotted {len(start_x)} starting positions", extra={"console": False})
    
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
            logger.debug(f"Found {len(odor_idx)} odor encounter points for example trajectory", extra={"console": False})
    
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
        logger.debug("Generated sensory response time series plot", extra={"console": False})
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
            logger.debug(f"Generated latency histogram with {len(valid_latencies)} successful agents", extra={"console": False})
    
    plt.tight_layout()
    logger.info("Completed trajectory visualization generation", extra={"console": False})
    return fig

def main():
    """Main execution function"""
    logger.info("Starting view_results.py execution", extra={"console": False})
    
    # Get filename from command line or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        logger.info(f"Using filename from command line: {filename}", extra={"console": False})
    else:
        filename = 'results/nav_results_0000.mat'
        logger.info(f"Using default filename: {filename}", extra={"console": False})
    
    logger.info(f"Loading: {filename}")
    
    # Load and analyze
    try:
        out = load_results(filename)
        logger.info("Results loaded successfully, starting analysis", extra={"console": False})
        
        analyze_results(out)
        
        # Create plots
        fig = plot_trajectories(out)
        
        # Save plots
        plot_file = filename.replace('.mat', '_plots.png')
        fig.savefig(plot_file, dpi=150, bbox_inches='tight')
        logger.info(f"\nPlots saved to: {plot_file}")
        logger.info(f"Plots successfully saved to: {plot_file}", extra={"console": False})
        
        plt.show()
        logger.info("Visualization display completed", extra={"console": False})
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        logger.exception("Full exception details:", extra={"console": False})
        sys.exit(1)

if __name__ == '__main__':
    main()