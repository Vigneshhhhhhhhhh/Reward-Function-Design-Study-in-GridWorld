import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from config import RESULTS_DIR, PLOT_DPI

FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")

def _save_fig(fig, filename):
    """
    Helper function to save a matplotlib figure object as both PNG and PDF.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        The base filename without extension.
    """
    base_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(f"{base_path}.png", dpi=PLOT_DPI, bbox_inches='tight')
    fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
    plt.close(fig)

def convergence_plot(grid_size=(10, 10)):
    """
    Generates a line chart displaying Bellman error vs iterations.
    Plots the mean across all seeds with ±1 standard deviation shading for
    each reward condition.
    
    Parameters
    ----------
    grid_size : tuple of int, optional
        The grid dimension to plot convergence for. Defaults to (10, 10).
    """
    raw_files = glob.glob(os.path.join(RESULTS_DIR, "raw", f"*_{grid_size[0]}x{grid_size[1]}_*.json"))
    
    data = {}
    max_len = 0
    
    for f in raw_files:
        with open(f, 'r') as file:
            res = json.load(file)
            cond = res['condition']
            if cond not in data:
                data[cond] = []
            curve = res.get('bellman_error_curve', [])
            data[cond].append(curve)
            max_len = max(max_len, len(curve))
            
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cond, curves in data.items():
        # Pad shorter curves with their final value to ensure matrix logic works
        padded_curves = []
        for c in curves:
            if not c:
                padded_curves.append([0.0] * max_len)
            else:
                padded = c + [c[-1]] * (max_len - len(c))
                padded_curves.append(padded)
                
        matrix = np.array(padded_curves)
        mean_curve = np.mean(matrix, axis=0)
        std_curve = np.std(matrix, axis=0)
        
        iters = np.arange(len(mean_curve))
        line = ax.plot(iters, mean_curve, label=cond, linewidth=2)[0]
        ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, color=line.get_color(), alpha=0.2)
        
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_xlabel('Iterations', fontsize=12)
    ax.set_ylabel('Max Bellman Error (Log Scale)', fontsize=12)
    ax.set_title(f'Value Iteration Convergence Rates ({grid_size[0]}x{grid_size[1]} Grid)', fontsize=14)
    ax.legend()
    
    _save_fig(fig, f"convergence_plot_{grid_size[0]}x{grid_size[1]}")

def policy_heatmap(env, V, policy, title, filename):
    """
    Generates a visual heatmap of the value function V(s) overlaid with
    optimal policy arrows. Colors the goal green and obstacles black.
    
    Parameters
    ----------
    env : GridWorld
        The environment instance defining states and obstacles.
    V : dict
        A dictionary mapping states to their scalar value.
    policy : dict
        A dictionary mapping states to the best integer action.
    title : str
        The title string for the plot.
    filename : str
        The base filename to save the outputs as.
    """
    grid = np.zeros((env.rows, env.cols))
    
    # Track min and max value for robust coloration
    vals = []
    for s in env.states:
        if s not in env.obstacles and s != env.goal_state:
            vals.append(V[s])
            
    v_min, v_max = min(vals) if vals else 0, max(vals) if vals else 1
    
    for s in env.states:
        grid[s] = V[s]
        
    for obs in env.obstacles:
        grid[obs] = np.nan
        
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(grid, annot=False, cmap="viridis", cbar=True, ax=ax,
                 cbar_kws={'label': 'Value $V(s)$'}, vmin=v_min, vmax=v_max)
                 
    # Black out obstacles
    ax.set_facecolor('black')
    
    # Draw policy arrows
    # 0: Up, 1: Right, 2: Down, 3: Left
    action_dx = {0: 0, 1: 0.4, 2: 0, 3: -0.4}
    action_dy = {0: -0.4, 1: 0, 2: 0.4, 3: 0}
    
    for s in env.states:
        if s in env.obstacles or s == env.goal_state:
            continue
            
        r, c = s
        a = policy.get(s, 0)
        
        # Calculate cell center
        cx, cy = c + 0.5, r + 0.5
        dx, dy = action_dx[a], action_dy[a]
        
        # Draw arrow pointing in the policy direction
        ax.arrow(cx - dx*0.5, cy - dy*0.5, dx, dy, 
                 head_width=0.2, head_length=0.2, fc='white', ec='white', 
                 alpha=0.8, length_includes_head=True)
                 
    # Mark Goal
    gr, gc = env.goal_state
    rect = plt.Rectangle((gc, gr), 1, 1, fill=True, color='lime', alpha=0.5)
    ax.add_patch(rect)
    ax.text(gc + 0.5, gr + 0.5, 'G', color='black', ha='center', va='center', fontweight='bold', fontsize=16)
    
    # Mark Start (Optional but helpful)
    sr, sc = env.start_state
    ax.text(sc + 0.5, sr + 0.5, 'S', color='red', ha='center', va='center', fontweight='bold', fontsize=16)

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    _save_fig(fig, filename)

def optimality_bar_chart():
    """
    Generates a bar chart comparing the mean optimality score (optimal path
    length divided by actual policy path length) across all reward conditions
    and grid sizes. Reads intermediate statistics directly from summary.json.
    """
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_path):
        return
        
    with open(summary_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    df = df[df['status'] == 'success'] # Only include successful runs
    df['grid_size_str'] = df['grid_size'].apply(lambda x: f"{x[0]}x{x[1]}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(data=df, x="condition", y="optimality_score", hue="grid_size_str", 
                capsize=.1, err_kws={'linewidth': 1}, ax=ax)
                
    ax.set_title("Mean Optimality Score Across Reward Conditions & Grid Sizes", fontsize=16)
    ax.set_xlabel("Reward Condition", fontsize=14)
    ax.set_ylabel("Optimality Score (Optimal_Len / Agent_Len)", fontsize=14)
    ax.set_ylim([0, 1.05])
    plt.legend(title='Grid Size')
    
    _save_fig(fig, "optimality_bar_chart")

def value_evolution_gif(run_id, env):
    """
    Compiles an animated GIF visualization showing how the value function V(s)
    evolves spatially over iterations using stored snapshots.
    
    Parameters
    ----------
    run_id : str
        The specific unique run ID used to fetch the raw JSON telemetry.
    env : GridWorld
        The environment context matching the run's topology.
    """
    file_path = os.path.join(RESULTS_DIR, "raw", f"{run_id}.json")
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
        
    with open(file_path, 'r') as f:
        res = json.load(f)
        
    snapshots = res.get('value_function_snapshots', [])
    if not snapshots: return
    
    temp_dir = os.path.join(RESULTS_DIR, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    frames = []
    
    # Find global min and max for consistent color scaling
    all_vals = []
    for snap in snapshots:
        all_vals.extend([v for k, v in snap.items() if tuple(map(int, k.split(','))) not in env.obstacles])
        
    v_min, v_max = (min(all_vals), max(all_vals)) if all_vals else (0, 1)
    if v_min == v_max: v_max += 0.1
    
    for i, snap in enumerate(snapshots):
        grid = np.zeros((env.rows, env.cols))
        
        for k_str, v in snap.items():
            r, c = map(int, k_str.split(','))
            grid[r, c] = v
            
        for obs in env.obstacles:
            grid[obs] = np.nan
            
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(grid, annot=False, cmap="viridis", cbar=True, ax=ax,
                     vmax=v_max, vmin=v_min)
                     
        ax.set_facecolor('black')
        ax.set_title(f"Value Evolution - {res['condition']} (Iter: {i*10})", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Mark goal
        gr, gc = env.goal_state
        rect = plt.Rectangle((gc, gr), 1, 1, fill=True, color='lime', alpha=0.5)
        ax.add_patch(rect)
        ax.text(gc + 0.5, gr + 0.5, 'G', color='black', ha='center', va='center', fontweight='bold', fontsize=16)
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        frames.append(frame_path)
        
    gif_path = os.path.join(FIGURES_DIR, f"value_evolution_{res['condition']}.gif")
    with imageio.get_writer(gif_path, mode='I', duration=200, loop=0) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
            
    # Cleanup temp frames
    for f in frames:
        try:
            os.remove(f)
        except OSError:
            pass
            
    shutil.rmtree(temp_dir, ignore_errors=True)
