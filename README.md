# Reward Function Design Study in GridWorld

This repository contains a tabular reinforcement learning study investigating the impact of various reward function designs on the convergence speed and policy optimality of Value Iteration in a stochastic GridWorld environment.

## Project Structure
- `environment/`: Core GridWorld MDP implementation.
- `algorithms/`: Value Iteration and Policy Iteration dynamic programming solvers.
- `rewards/`: Contains 5 distinct reward structures (Dense, Sparse, Shaped, Deceptive, NegativeStep).
- `experiments/`: Experiment runner tracking metrics and collecting telemetry.
- `visualization/`: Automated graph, heatmap, and GIF generation using Matplotlib/Seaborn.
- `analysis/`: Automated statistical analysis, findings generation, and LaTeX table exporting.

## Installation
Ensure you have Python 3.10+ installed. Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running Experiments

You can orchestrate the full end-to-end pipeline (experiments, visualization, and analysis) from the CLI:

```bash
python main.py --grid_size all --seeds 30 --save_figs --export_latex
```

### CLI Arguments
- `--grid_size`: Grid dimension to run. Options: '5', '10', '20', or 'all' (default: 'all').
- `--seeds`: Number of random seeds to evaluate per condition (default: 30).
- `--save_figs`: Flag to generate and save visualization plots and heatmaps/GIFs.
- `--export_latex`: Flag to export the summary analysis table to LaTeX format.
- `--quick`: Flag to run a fast 5x5 validation test over just 5 seeds. Overrides other constraints.

## Reproducing Paper Figures
To exactly reproduce the figures, datasets, and statistics from the baseline study:
```bash
python main.py --grid_size all --seeds 30 --save_figs --export_latex
```
Once complete, check the `results/` folder for CSV/JSON artifacts, and `results/figures/` for high-dpi plots and animated GIFs!
