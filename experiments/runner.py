import os
import json
import time
import numpy as np
from tqdm import tqdm
from collections import deque

from environment.gridworld import GridWorld
from algorithms.value_iteration import ValueIteration
from rewards.reward_functions import DenseReward, SparseReward, ShapedReward, DeceptiveReward, NegativeStep
from config import RESULTS_DIR, GAMMA, MAX_ITERATIONS

class ExperimentRunner:
    """
    Orchestrates the running of Value Iteration across multiple grid
    sizes, reward conditions, and random seeds. Auto-saves results.
    """
    def __init__(self, grid_sizes=None, num_seeds=30):
        if grid_sizes is None:
            self.grid_sizes = [(5, 5), (10, 10), (20, 20)]
        else:
            self.grid_sizes = grid_sizes
        self.num_seeds = num_seeds
        self.base_out_dir = os.path.join(RESULTS_DIR, "raw")
        os.makedirs(self.base_out_dir, exist_ok=True)
        
    def _bfs_shortest_path(self, env):
        """
        Find optimal path length using BFS (ignoring slip probability).
        Provides a baseline for the absolute shortest deterministic path.
        
        Parameters
        ----------
        env : GridWorld
            The instantiated grid environment.
            
        Returns
        -------
        int or float
            The shortest path length to the goal, or float('inf') if unreachable.
        """
        if env.start_state == env.goal_state:
            return 0
            
        queue = deque([(env.start_state, 0)])
        visited = {env.start_state}
        
        while queue:
            curr, dist = queue.popleft()
            
            for a in env.actions:
                # Deterministic next state for absolute shortest path
                # Use private method to avoid stochasticity here
                nxt = env._get_next_state(curr, a)
                
                if nxt == env.goal_state:
                    return dist + 1
                    
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
                    
        return float('inf') # Unreachable goal
        
    def _evaluate_policy_path_length(self, env, policy, max_steps=1000):
        """
        Evaluate paths taken by a fully deterministic optimal policy for comparison.
        
        Parameters
        ----------
        env : GridWorld
            The grid environment to evaluate within.
        policy : dict
            The mapping of states to actions output by an algorithm.
        max_steps : int, optional
            Timeout for endless loops. Defaults to 1000.
            
        Returns
        -------
        int
            The number of steps taken to reach the goal.
        """
        curr = env.start_state
        steps = 0
        
        while steps < max_steps:
            if curr == env.goal_state:
                return steps
                
            action = policy.get(curr, 0)
            next_state = env._get_next_state(curr, action)
            
            # If agent loops and gets stuck
            if next_state == curr and curr not in env.obstacles: 
                # might be stuck against an obstacle due to bad policy
                stuck = True
                for a in env.actions:
                    if env._get_next_state(curr, a) != curr:
                        stuck = False
                        break
                if stuck: return max_steps
                
            curr = next_state
            steps += 1
            
        return max_steps

    def run_all(self):
        """
        Executes iterations for all combinations of grid sizes, reward
        conditions, and random seeds. Dumps telemetric readouts as JSONs.
        """
        total_runs = len(self.grid_sizes) * self.num_seeds * 5 # 5 reward conditions
        progress_bar = tqdm(total=total_runs, desc="Running Experiments")
        
        results = []
        
        for size in self.grid_sizes:
            for seed in range(self.num_seeds):
                # We need to guarantee the environment is reachable.
                # Generate environment and check BFS
                while True:
                    env = GridWorld(size=size, obstacle_density=0.1, slip_prob=0.1, random_seed=seed)
                    optimal_length = self._bfs_shortest_path(env)
                    if optimal_length != float('inf'):
                        break
                    # If unreachable, increment seed internally to find a valid grid
                    seed += 1000 
                
                # Find valid decoy for DeceptiveReward
                decoy_state = (size[0] // 2, size[1] // 2)
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
                
                for cond_name, reward_fn in reward_conditions.items():
                    run_id = f"{cond_name}_{size[0]}x{size[1]}_seed{seed}"
                    
                    try:
                        solver = ValueIteration()
                        V, policy, bellman_errors, tracking = solver.solve(
                            env, reward_fn, gamma=GAMMA
                        )
                        
                        policy_length = self._evaluate_policy_path_length(env, policy)
                        optimality_score = optimal_length / policy_length if policy_length > 0 else 0
                        
                        # Convert history keys from tuples to strings for JSON
                        serializable_v_history = []
                        for snap in solver.v_history:
                            serializable_snap = {f"{k[0]},{k[1]}": v for k, v in snap.items()}
                            serializable_v_history.append(serializable_snap)
                            
                        result = {
                            "run_id": run_id,
                            "condition": cond_name,
                            "grid_size": list(size),
                            "seed": seed,
                            "iterations_to_converge": tracking["iterations"],
                            "convergence_time": tracking["convergence_time"],
                            "final_error": tracking["final_error"],
                            "optimal_path_length": optimal_length,
                            "policy_path_length": policy_length,
                            "optimality_score": optimality_score,
                            "bellman_error_curve": bellman_errors,
                            "value_function_snapshots": serializable_v_history,
                            "status": "success" if tracking["iterations"] < MAX_ITERATIONS else "timeout"
                        }
                        
                    except Exception as e:
                        result = {
                            "run_id": run_id,
                            "condition": cond_name,
                            "grid_size": list(size),
                            "seed": seed,
                            "status": "failed",
                            "error": str(e)
                        }
                        
                    # Save individual JSON
                    out_path = os.path.join(self.base_out_dir, f"{run_id}.json")
                    with open(out_path, 'w') as f:
                        json.dump(result, f)
                        
                    results.append(result)
                    progress_bar.update(1)
                    progress_bar.set_postfix({"Grid": size, "Cond": cond_name})
                    
        progress_bar.close()
        
        # Save summary
        summary_path = os.path.join(RESULTS_DIR, "summary.json")
        summary_data = [
            {k: v for k, v in r.items() if k not in ["bellman_error_curve", "value_function_snapshots"]}
            for r in results
        ]
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
            
        print(f"All experiments completed! Raw data in {self.base_out_dir}")
        print(f"Summary saved to {summary_path}")

def run_experiments(grid_sizes=None, num_seeds=30):
    """
    Initializes and triggers the ExperimentRunner.
    
    Parameters
    ----------
    grid_sizes : list of tuple of int, optional
        The grids to run. Defaults to standard baseline.
    num_seeds : int, optional
        Runs per condition per grid size. Defaults to 30.
    """
    runner = ExperimentRunner(grid_sizes=grid_sizes, num_seeds=num_seeds)
    runner.run_all()
