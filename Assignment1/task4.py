"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
import math
# You can use this space to define any helper functions that you need
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.t = 0

        self.heads1_t = np.zeros(num_arms)
        self.tails1_t = np.zeros(num_arms)
        self.heads2_t = np.zeros(num_arms)
        self.tails2_t = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE 
        if self.t < 2*self.num_arms:
            return self.t % self.num_arms
        else:   
            beta_samples1 = np.zeros(self.num_arms)
            beta_samples2 = np.zeros(self.num_arms)

            for idx in range(self.num_arms):
                beta_samples1[idx] = np.random.beta(self.heads1_t[idx]+1, self.tails1_t[idx]+1)
                beta_samples2[idx] = np.random.beta(self.heads2_t[idx]+1, self.tails2_t[idx]+1)

            return np.argmax(beta_samples1 + beta_samples2)
        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        self.t += 1
        if set_pulled == 0:  
                self.heads1_t[arm_index] += reward
                self.tails1_t[arm_index] += 1 - reward
        else: 
                self.heads2_t[arm_index] += reward
                self.tails2_t[arm_index] += 1 - reward
            # END EDITING HERE

