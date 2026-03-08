import os
import glob
import json
from environment.gridworld import GridWorld
from algorithms.value_iteration import ValueIteration
from rewards.reward_functions import DenseReward, SparseReward, ShapedReward, DeceptiveReward, NegativeStep
from visualization.plotter import convergence_plot, optimality_bar_chart, policy_heatmap, value_evolution_gif
from config import RESULTS_DIR, GAMMA

def run_visualization_suite(grid_sizes=None, save_figs=True):
    """
    Orchestrates the entire plotting process for the reward study.
    Generates convergence line plots, optimality bar charts, and 
    policy heatmaps + evolution animated GIFs for a representative 10x10 environment.
    """
    if not save_figs:
        print("Skipping visualizations (--save_figs not specified)")
        return
        
    if grid_sizes is None:
        grid_sizes = [(5, 5), (10, 10), (20, 20)]
        
    print("Generating Convergence Plots...")
    for size in grid_sizes:
        convergence_plot(grid_size=size)
        
    print("Generating Optimality Bar Chart...")
    optimality_bar_chart()
    
    print("Generating Policy Heatmaps and Value Evolution GIFs...")
    # Generate ONE representative GridWorld to plot heatmaps on (e.g. 10x10 Seed 0)
    grid_size = (10, 10)
    seed = 0
    env = GridWorld(size=grid_size, obstacle_density=0.1, slip_prob=0.1, random_seed=seed)
    
    # Needs to match runner decoy logic exactly to display right
    decoy_state = (grid_size[0] // 2, grid_size[1] // 2)
    if decoy_state == env.goal_state or decoy_state == env.start_state or decoy_state in env.obstacles:
        for s in env.states:
            if s != env.goal_state and s != env.start_state and s not in env.obstacles:
                decoy_state = s
                break
                
    reward_conditions = {
        "DenseReward": DenseReward(env.goal_state),
        "SparseReward": SparseReward(env.goal_state),
        "ShapedReward": ShapedReward(env.goal_state, gamma=GAMMA),
        "DeceptiveReward": DeceptiveReward(env.goal_state, decoy_state=decoy_state),
        "NegativeStep": NegativeStep(env.goal_state)
    }

    # Instead of pulling out JSON and deserializing V, it's safer and easier to just re-run 1 baseline
    for cond_name, reward_fn in reward_conditions.items():
        print(f"  -> Processing {cond_name} heatmaps/GIFs...")
        
        # We find the file from earlier experiment matching this exact scenario
        run_id = f"{cond_name}_{grid_size[0]}x{grid_size[1]}_seed{seed}"
        
        # Run VI just to get standard output policy + V for the static heatmap
        vi = ValueIteration()
        V, policy, _, _ = vi.solve(env, reward_fn, gamma=GAMMA)
        
        policy_heatmap(env, V, policy, f"{cond_name} Value & Policy", f"policy_heatmap_{cond_name}")
        value_evolution_gif(run_id, env)
        
    print("Visualizations complete! Check results/figures/")

if __name__ == "__main__":
    run_visualization_suite()
