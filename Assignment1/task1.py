"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
def kl(p,q):
    if p == 0:
        return -math.log(1-q)
    elif p == 1:
        return -math.log(q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

# You can use this space to define any helper functions that you need
def find_q(u, pa, t):
    error = 10.0
    left = float(pa)
    right = 1.0
    c = 0.0
    g = (lambda y : (kl(pa, y) - (math.log(t) / u)))
    while right - left > 5e-2:
        middle = (left+right)/2.0
        if g(middle) < 0:     
            left = middle
        else:
            right = middle
    return left
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.t = 0
        self.ucb_t = np.zeros(num_arms) #np.random.random(size = num_arms) 
        self.u_t = np.zeros(num_arms)
        self.heads_t = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.t < self.num_arms:
            return self.t
        else:
            return np.argmax(self.ucb_t)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.heads_t[arm_index] += reward
        self.u_t[arm_index] += 1
        self.t += 1
        for idx in range(self.num_arms):
            self.ucb_t[idx] = ((self.heads_t[idx])/(self.u_t[idx]+1e-15)) + np.sqrt(2*np.log(self.t)/(self.u_t[idx]+1e-15))
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.t = 0
        self.u_t = np.zeros(num_arms)
        self.heads_t = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.t < self.num_arms:
            return self.t
        klucb_t = np.zeros(self.num_arms)
        for idx in range(self.num_arms):
            klucb_t[idx] = find_q(self.u_t[idx], self.heads_t[idx]/self.u_t[idx], self.t)
        return np.argmax(klucb_t)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.t += 1
        self.heads_t[arm_index] += reward
        self.u_t[arm_index] += 1
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.heads_t = np.zeros(num_arms)
        self.tails_t = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        beta_samples = np.zeros(self.num_arms)
        for idx in range(self.num_arms):
            beta_samples[idx] = np.random.beta(self.heads_t[idx]+1, self.tails_t[idx]+1)
        return np.argmax(beta_samples)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.heads_t[arm_index] += reward
        self.tails_t[arm_index] += 1 - reward
        # END EDITING HERE 
