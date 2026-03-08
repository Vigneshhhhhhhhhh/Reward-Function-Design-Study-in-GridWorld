# config.py

GRID_SIZE = (6, 6)
START_STATE = (0, 0)
GOAL_STATE = (5, 5)

# A small map with some obstacles in the middle
OBSTACLES = [(2, 2), (2, 3), (3, 2), (3, 3), (1, 4), (4, 1)]

# MDP parameters
SLIP_PROBABILITY = 0.1
GAMMA = 0.99
THETA = 1e-6 # Convergence threshold
MAX_ITERATIONS = 10000

# Visualization settings
PLOT_DPI = 300
RESULTS_DIR = "results"
