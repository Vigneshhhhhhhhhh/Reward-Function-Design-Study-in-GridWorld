import numpy as np  # type: ignore
from typing import Tuple, List, Set, Dict, Optional, Any, Union

class GridWorld:
    """
    A GridWorld Markov Decision Process (MDP) environment.
    
    This environment consists of a 2D grid where an agent can move in four
    directions (Up, Right, Down, Left). It supports random obstacles,
    stochastic transitions (slipping), and customizable start/goal states.
    
    Parameters
    ----------
    size : int or tuple of int, optional
        The dimensions of the grid (rows, cols). If an int is provided,
        creates a square grid of size x size. Default is (10, 10).
    obstacle_density : float, optional
        The fraction of the grid to cover with random obstacles.
        Must be between 0.0 and 1.0. Default is 0.1.
    slip_prob : float, optional
        The probability that the agent ignores its intended action and
        takes a random action uniformly instead. Default is 0.1.
    start_state : tuple of int, optional
        The (row, col) coordinate of the starting state. Defaults to (0, 0).
    goal_state : tuple of int, optional
        The (row, col) coordinate of the goal state. Defaults to
        (rows - 1, cols - 1).
    random_seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        
    Attributes
    ----------
    rows : int
        Number of rows in the grid.
    cols : int
        Number of columns in the grid.
    states : list of tuple of int
        List of all valid (row, col) states in the environment.
    actions : list of int
        List of available actions [0, 1, 2, 3].
    obstacles : set of tuple of int
        Set of (row, col) coordinates representing obstacles.
    current_state : tuple of int
        The agent's current location in the grid.
    """
    def __init__(self, size: Union[int, Tuple[int, int]] = (10, 10), 
                 obstacle_density: float = 0.1, slip_prob: float = 0.1, 
                 start_state: Optional[Tuple[int, int]] = None, 
                 goal_state: Optional[Tuple[int, int]] = None, 
                 random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if isinstance(size, int):
            self.rows, self.cols = size, size
        else:
            self.rows, self.cols = size
            
        self.slip_prob: float = slip_prob
        self.actions: List[int] = [0, 1, 2, 3]  # 0: Up, 1: Right, 2: Down, 3: Left
        
        # Generate grid and list of states
        self.states: List[Tuple[int, int]] = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        
        # Start and Goal states
        self.start_state: Tuple[int, int] = start_state if start_state else (0, 0)
        self.goal_state: Tuple[int, int] = goal_state if goal_state else (self.rows - 1, self.cols - 1)
        
        # Place random obstacles
        num_obstacles = int(self.rows * self.cols * obstacle_density)
        self.obstacles: Set[Tuple[int, int]] = set()
        
        available_cells = [s for s in self.states if s != self.start_state and s != self.goal_state]
        
        if num_obstacles > 0 and len(available_cells) >= num_obstacles:
            obs_indices = np.random.choice(len(available_cells), num_obstacles, replace=False)
            self.obstacles = {available_cells[i] for i in obs_indices}
            
        self.current_state: Tuple[int, int] = self.start_state

    def reset(self) -> Tuple[int, int]:
        """
        Resets the environment to the starting state.
        
        Returns
        -------
        tuple of int
            The initial state of the environment.
        """
        self.current_state = self.start_state
        return self.current_state
        
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        Checks if the given state is a terminal state.
        
        Parameters
        ----------
        state : tuple of int
            The state to check.
            
        Returns
        -------
        bool
            True if the state is the goal state, False otherwise.
        """
        return state == self.goal_state
        
    def step(self, action: int) -> Tuple[Tuple[int, int], bool]:
        """
        Takes a step in the environment using the given action.
        
        Parameters
        ----------
        action : int
            The action to take (0: Up, 1: Right, 2: Down, 3: Left).
            
        Returns
        -------
        next_state : tuple of int
            The state the agent transitioned into.
        done : bool
            Whether the episode has terminated.
        """
        transitions = self.get_transitions(self.current_state, action)
        
        # Choose next state based on probabilities
        probs = [p for p, s in transitions]
        states_idx = list(range(len(transitions)))
        
        next_idx = np.random.choice(states_idx, p=probs)
        next_state = transitions[next_idx][1]
        
        self.current_state = next_state
        done = self.is_terminal(next_state)
        
        return next_state, done

    def get_transitions(self, state: Tuple[int, int], action: int) -> List[Tuple[float, Tuple[int, int]]]:
        """
        Calculates the transition probabilities for taking an action in a state.
        
        Parameters
        ----------
        state : tuple of int
            The state the agent is currently in.
        action : int
            The intended action to take.
            
        Returns
        -------
        list of tuple
            A list of (probability, next_state) tuples representing all
            possible transitions from the given state-action pair.
            Transition is stochastic: probability `slip_prob` to take a random action.
        """
        if self.is_terminal(state):
            return [(1.0, state)]
            
        transitions: Dict[Tuple[int, int], float] = {}
        for a in self.actions:
            # Chance to take intended action is 1 - slip_prob, plus slip_prob * (1/4)
            prob = (1.0 - self.slip_prob) if a == action else 0.0
            prob += self.slip_prob / len(self.actions)
            
            if prob > 0:
                s_prime = self._get_next_state(state, a)
                transitions[s_prime] = transitions.get(s_prime, 0.0) + prob
                
        return [(p, s) for s, p in transitions.items()]
        
    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Computes the deterministic next state for a specific action,
        ignoring bounds and obstacle checks initially.
        
        Parameters
        ----------
        state : tuple of int
            The starting state.
        action : int
            The action to apply.
            
        Returns
        -------
        tuple of int
            The resulting state after the applied action. Returns the
            original state if the movement is blocked by bounds or an obstacle.
        """
        dr, dc = 0, 0
        if action == 0: dr, dc = -1, 0  # Up
        elif action == 1: dr, dc = 0, 1 # Right
        elif action == 2: dr, dc = 1, 0 # Down
        elif action == 3: dr, dc = 0, -1# Left
        
        r, c = state
        nr, nc = r + dr, c + dc
        
        # Cannot move outside bounds or into obstacles
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols or (nr, nc) in self.obstacles:
            return state
        return (nr, nc)
