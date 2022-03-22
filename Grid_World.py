import numpy as np
import itertools

np.set_printoptions(precision=3, linewidth=180)

class GridWorld:

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, side=4):
        self.side = side
        # -------------------------
        # Define integer states, actions, and final states as specified in the problem description
    
        self.actions = np.array([self.UP, self.DOWN, self.LEFT, self.RIGHT])
        self.states = np.arange(self.side * self.side)
        self.finals = np.array([0, 15])

        self.actions_repr = np.array(['↑', '↓', '←', '→'])

    def _is_terminal(self, s):
        # -------------------------
        # Return True if s is a terminal state and False otherwise
        out = False
        if s in self.finals:
            out = True
        return out   
        
    def _next_state(self, s, a):
        # -------------------------
        # Returns the next state of the environment if action a were taken
        # while in state s

        next_state = s
        
        l_borders = np.array(range(0,self.side*(self.side-1)+1,self.side))
        r_borders = np.array(range(self.side-1,self.side**2,self.side))
        
        up_borders =np.array(range(0,self.side))
        down_borders = np.array(range((self.side**2)-self.side,self.side**2))
        

        
        if a == 0:#up
            if next_state not in up_borders:
                next_state = next_state-self.side
            
        if a==1: #down
            if next_state not in down_borders:
                next_state = next_state+self.side
        
        if a==2: #left
            if next_state not in l_borders:
                next_state = next_state-1
        
        if a==3:#right            
            if next_state not in r_borders:
                next_state = next_state+1
        
        self.s = next_state
    
        return self.s

    def _reward(self, s, s_next, a):
        # -------------------------
        # Return the reward for the given transition as specified
        # in the problem description

        reward = -1
        if s_next in self.finals:
            reward = 0
        
        return reward
    
    def reset(self):
        # -------------------------
        # Set the internal state of the environment to be sampled uniformly
        # at random from the set of non-terminal states and return the state
        
        self.s = np.random.choice(np.arange(self.side*self.side)[1:-1])
        return self.s
    
    def step(self, a):
        # -------------------------
        # Advances the environment one step using action a and returns s, r, T
        # where s is the next state, r is the reward, and T is a boolean saying
        # whether the episode is done or not
        
        next_state = self._next_state(self.s,a)
        reward = self._reward(s = self.s,a=a,s_next=next_state)
        episode = self._is_terminal(s=next_state)
        
        return next_state, reward, episode

    def print_policy(self, policy):
        P = np.array(policy).reshape(self.side, self.side)
        print(self.actions_repr[P])
  
    def print_values(self, values):
        V = np.array(values).reshape(self.side, self.side)
        print(V)
        
