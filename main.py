import os
import sys
import argparse

# Ensure modules can be imported if run from elsewhere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.runner import run_experiments
from visualization.generate_figs import run_visualization_suite
from analysis.analyzer import run_analysis

def parse_args():
    parser = argparse.ArgumentParser(description="Run the Reinforcement Learning Reward Function Study.")
    parser.add_argument("--grid_size", type=str, default="all",
                        help="Grid size to run. Options: '5', '10', '20', or 'all'. Set to 'all' for standard baseline.")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of random seeds per condition.")
    parser.add_argument("--save_figs", action="store_true",
                        help="Flag to save visualization plots and GIFs.")
    parser.add_argument("--export_latex", action="store_true",
                        help="Flag to export the summary table to LaTeX.")
    parser.add_argument("--quick", action="store_true",
                        help="Run a fast 5x5 test over 5 seeds. Overrides other settings.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Process arguments
    if args.quick:
        grid_sizes = [(5, 5)]
        num_seeds = 5
        print(">>> QUICK RUN ENABLED: Validating with 5x5 grid and 5 seeds. <<<")
    else:
        num_seeds = args.seeds
        if args.grid_size == "all":
            grid_sizes = [(5, 5), (10, 10), (20, 20)]
        else:
            s = int(args.grid_size)
            grid_sizes = [(s, s)]

    print("======================================================")
    print(" Starting Reward Function Design Study in GridWorld")
    print("======================================================")
    
    print(f"\n--- PHASE 1: Running Experiments (Grids: {grid_sizes}, Seeds: {num_seeds}) ---")
    run_experiments(grid_sizes=grid_sizes, num_seeds=num_seeds)
    
    print("\n--- PHASE 2: Generating Visualizations ---")
    run_visualization_suite(grid_sizes=grid_sizes, save_figs=args.save_figs)
    
    print("\n--- PHASE 3: Analyzing Results ---")
    run_analysis(export_latex=args.export_latex)
    
    print("\n======================================================")
    print(" Study Completed Successfully! Check the results/ folder.")
    print("======================================================")
