# pyre-ignore-all-errors
# pyright: reportGeneralTypeIssues=false, reportOperatorIssue=false
# mypy: ignore-errors
import time
from typing import Dict, List, Tuple, Any
import numpy as np  # type: ignore
from config import GAMMA, THETA, MAX_ITERATIONS  # type: ignore

class ValueIteration:
    """
    Computes the optimal policy and value function for an MDP using
    the Value Iteration algorithm.
    """
    def __init__(self):
        self.v_history: List[Dict[Tuple[int, int], float]] = []
        
    def solve(self, env: Any, reward_fn: Any, gamma: float = 0.99, theta: float = 1e-6) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int], List[float], Dict[str, Any]]:
        """
        Solves the given environment using Value Iteration.
        
        Parameters
        ----------
        env : GridWorld
            The MDP environment to solve.
        reward_fn : BaseReward
            The reward condition function instance.
        gamma : float, optional
            The discount factor. Defaults to 0.99.
        theta : float, optional
            The convergence threshold for Bellman error. Defaults to 1e-6.
            
        Returns
        -------
        V : dict
            The optimal value function mapping states to scalar values.
        policy : dict
            The deterministic optimal policy mapping states to actions.
        bellman_errors : list of float
            The history of max Bellman errors tracked per iteration.
        tracking_info : dict
            Various convergence metadata: iterations, CPU time, final error.
        """
        start_time = time.time()
        V: Dict[Tuple[int, int], float] = {s: 0.0 for s in env.states}
        policy: Dict[Tuple[int, int], int] = {s: 0 for s in env.states}
        bellman_errors: List[float] = []
        self.v_history = []
        
        iterations: int = 0
        while iterations < MAX_ITERATIONS:  # type: ignore
            delta: float = 0.0
            new_V: Dict[Tuple[int, int], float] = {}
            if iterations % 10 == 0:
                self.v_history.append(V.copy())
                
            for s in env.states:
                if env.is_terminal(s):
                    new_V[s] = 0.0
                    continue
                    
                action_values: List[float] = []
                for a in env.actions:
                    val: float = 0.0
                    for prob, next_s in env.get_transitions(s, a):
                        r = reward_fn.get_reward(s, a, next_s)  # type: ignore
                        val += prob * (r + gamma * V[next_s])  # type: ignore
                    action_values.append(val)
                
                best_val = max(action_values)  # type: ignore
                new_V[s] = best_val  # type: ignore
                delta = max(delta, abs(best_val - V[s]))  # type: ignore
                
            V = new_V
            bellman_errors.append(delta)  # type: ignore
            iterations += 1  # type: ignore
            if delta < theta:  # type: ignore
                break
                
        # Final snapshot if not just saved
        if (iterations - 1) % 10 != 0:  # type: ignore
            self.v_history.append(V.copy())
            
        # Extract optimal policy
        for s in env.states:
            if env.is_terminal(s):
                policy[s] = 0
                continue
                
            action_values: List[float] = []
            for a in env.actions:
                val: float = 0.0
                for prob, next_s in env.get_transitions(s, a):
                    r = reward_fn.get_reward(s, a, next_s)  # type: ignore
                    val += prob * (r + gamma * V[next_s])  # type: ignore
                action_values.append(val)
                
            policy[s] = int(np.argmax(action_values))  # type: ignore
            
        convergence_time = time.time() - start_time
        final_error = bellman_errors[-1] if bellman_errors else 0.0
        
        tracking_info = {
            "iterations": iterations,
            "convergence_time": convergence_time,
            "final_error": final_error
        }
        
        return V, policy, bellman_errors, tracking_info

class PolicyIteration:
    """
    Computes the optimal policy and value function for an MDP using
    the Policy Iteration algorithm.
    """
    def __init__(self):
        self.v_history: List[Dict[Tuple[int, int], float]] = []
        
    def solve(self, env: Any, reward_fn: Any, gamma: float = 0.99, theta: float = 1e-6) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int], List[float], Dict[str, Any]]:
        """
        Solves the given environment using Policy Iteration.
        
        Parameters
        ----------
        env : GridWorld
            The MDP environment to solve.
        reward_fn : BaseReward
            The reward condition function instance.
        gamma : float, optional
            The discount factor. Defaults to 0.99.
        theta : float, optional
            The convergence threshold for policy evaluation. Defaults to 1e-6.
            
        Returns
        -------
        V : dict
            The optimal value function mapping states to scalar values.
        policy : dict
            The deterministic optimal policy mapping states to actions.
        evaluation_errors : list of float
            The history of max Bellman errors tracked during evaluation phases.
        tracking_info : dict
            Convergence metadata including improvement steps, eval loops, CPU time.
        """
        start_time = time.time()
        V: Dict[Tuple[int, int], float] = {s: 0.0 for s in env.states}
        # Initialize random policy
        policy: Dict[Tuple[int, int], int] = {s: int(np.random.choice(env.actions)) for s in env.states}
        self.v_history = []
        
        evaluation_errors: List[float] = []
        policy_switches_history: List[int] = []
        improvement_steps: int = 0
        total_eval_iters: int = 0
        
        while True:
            if improvement_steps % 10 == 0:  # type: ignore
                self.v_history.append(V.copy())
                
            # Policy Evaluation
            while True:
                delta: float = 0.0
                new_V: Dict[Tuple[int, int], float] = {}
                for s in env.states:
                    if env.is_terminal(s):
                        new_V[s] = 0.0
                        continue
                        
                    a = policy[s]  # type: ignore
                    val: float = 0.0
                    for prob, next_s in env.get_transitions(s, a):
                        r = reward_fn.get_reward(s, a, next_s)  # type: ignore
                        val += prob * (r + gamma * V[next_s])  # type: ignore
                        
                    new_V[s] = val  # type: ignore
                    delta = max(delta, abs(val - V[s]))  # type: ignore
                
                V = new_V
                evaluation_errors.append(delta)  # type: ignore
                total_eval_iters += 1  # type: ignore
                if delta < theta:  # type: ignore
                    break
                    
            # Policy Improvement
            policy_stable = True
            policy_switches: int = 0
            for s in env.states:
                if env.is_terminal(s):
                    continue
                    
                old_action = policy[s]  # type: ignore
                action_values: List[float] = []
                for a in env.actions:
                    val: float = 0.0
                    for prob, next_s in env.get_transitions(s, a):
                        r = reward_fn.get_reward(s, a, next_s)  # type: ignore
                        val += prob * (r + gamma * V[next_s])  # type: ignore
                    action_values.append(val)
                    
                best_action = int(np.argmax(action_values))  # type: ignore
                policy[s] = best_action  # type: ignore
                
                if old_action != best_action:
                    policy_stable = False
                    policy_switches += 1  # type: ignore
                    
            policy_switches_history.append(policy_switches)  # type: ignore
            improvement_steps += 1  # type: ignore
            
            if policy_stable:
                break
                
        # Final snapshot
        if (improvement_steps - 1) % 10 != 0:  # type: ignore
            self.v_history.append(V.copy())
        
        convergence_time = time.time() - start_time
        
        tracking_info = {
            "improvement_steps": improvement_steps,
            "policy_switches_history": policy_switches_history,
            "total_eval_iters": total_eval_iters,
            "convergence_time": convergence_time
        }
        
        return V, policy, evaluation_errors, tracking_info
