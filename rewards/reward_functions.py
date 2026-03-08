from typing import Tuple
from config import GAMMA  # type: ignore

class BaseReward:
    """
    Abstract base class for all reward functions.
    """
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        """
        Computes the reward for a given transition.
        
        Parameters
        ----------
        state : tuple of int
            The state the agent started in.
        action : int
            The action taken by the agent.
        next_state : tuple of int
            The state the agent transitioned to.
            
        Returns
        -------
        float
            The scalar reward signal.
        """
        raise NotImplementedError
        
    def describe(self) -> str:
        """
        Returns a human-readable description of the reward function.
        
        Returns
        -------
        str
            A short description of the function.
        """
        raise NotImplementedError

class DenseReward(BaseReward):
    """
     Dense reward condition providing a signal strictly proportional 
    to the agent's Manhattan distance closeness to the goal.
    
    Parameters
    ----------
    goal_state : tuple of int
        The coordinates of the target goal state.
    """
    def __init__(self, goal_state: Tuple[int, int]):
        self.goal_state = goal_state
        
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        if next_state == self.goal_state:
            return 1.0
        
        r, c = next_state
        gr, gc = self.goal_state
        dist = abs(r - gr) + abs(c - gc)
        
        return 1.0 / (1.0 + dist)
        
    def describe(self) -> str:
        return "DenseReward: +1 proportional to closeness to goal (Manhattan distance)."

class SparseReward(BaseReward):
    """
    Sparse reward condition that issues a +1 only on reaching the goal state
    and 0 everywhere else.
    
    Parameters
    ----------
    goal_state : tuple of int
        The coordinates of the target goal state.
    """
    def __init__(self, goal_state: Tuple[int, int]):
        self.goal_state = goal_state
        
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        return 1.0 if next_state == self.goal_state else 0.0
        
    def describe(self) -> str:
        return "SparseReward: +1 only on reaching goal, 0 elsewhere."

class ShapedReward(BaseReward):
    """
    Potential-based reward shaping condition (Ng et al. 1999). 
    Base reward of +1 on goal, plus the difference in potential 
    phi between the next state and current state.
    
    Parameters
    ----------
    goal_state : tuple of int
        The coordinates of the target goal state.
    gamma : float, optional
        The discount factor. Defaults to the config GAMMA.
    """
    def __init__(self, goal_state: Tuple[int, int], gamma: float = GAMMA):
        self.goal_state = goal_state
        self.gamma = gamma
        
    def _phi(self, s: Tuple[int, int]) -> float:
        if s == self.goal_state:
            return 0.0
        r, c = s
        gr, gc = self.goal_state
        return -float(abs(r - gr) + abs(c - gc))
        
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        base_reward = 1.0 if next_state == self.goal_state else 0.0
        
        # Potential-based shaping F(s,a,s') = gamma * Phi(s') - Phi(s)
        f = self.gamma * self._phi(next_state) - self._phi(state)
        
        return base_reward + f
        
    def describe(self) -> str:
        return "ShapedReward: Potential-based shaping using Ng et al. 1999 formula."

class DeceptiveReward(BaseReward):
    """
    Deceptive reward condition that offers a high false reward at a decoy
    location to mislead the agent, alongside the real goal reward.
    
    Parameters
    ----------
    goal_state : tuple of int
        The coordinates of the target goal state.
    decoy_state : tuple of int
        The coordinates of the false trap state.
    """
    def __init__(self, goal_state: Tuple[int, int], decoy_state: Tuple[int, int]):
        self.goal_state = goal_state
        self.decoy_state = decoy_state
        
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        if next_state == self.goal_state:
            return 1.0
        elif next_state == self.decoy_state:
            return 0.5
        return 0.0
        
    def describe(self) -> str:
        return "DeceptiveReward: False +0.5 reward at a decoy location, +1 at true goal."

class NegativeStep(BaseReward):
    """
    Negative step penalty reward condition. Penalizes the agent with a small
    negative cost on every step to encourage the shortest path to the goal.
    
    Parameters
    ----------
    goal_state : tuple of int
        The coordinates of the target goal state.
    """
    def __init__(self, goal_state: Tuple[int, int]):
        self.goal_state = goal_state
        
    def get_reward(self, state: Tuple[int, int], action: int, next_state: Tuple[int, int]) -> float:
        if next_state == self.goal_state:
            return 1.0
        return -0.01
        
    def describe(self) -> str:
        return "NegativeStep: -0.01 per step + +1 at goal."
