import gym
from gym.spaces import Tuple, Discrete, Box
import torch
import torch.nn.functional as F

class EpisodicEnv(gym.Env):
    def __init__(self, states):
        super(EpisodicEnv, self).__init__()
        self.states = states
        self.action_space = Discrete(len(states))
        self.observation_space = Discrete(len(states))

        self.current_state = None
        self.transition_matrix = None  # Same as policy
        self.reset()


    def reset(self, seed=None, **kwargs):
        self.current_state = self.states[-1]

        # create a transition matrix for RL
        self.transition_matrix = torch.rand(len(self.states), len(self.states))
        self.transition_matrix = F.softmax(self.transition_matrix, dim=1)

        return self.observation_space.sample()

    def step(self, action):
        # TODO: Use generator to generate new state

        # Select a new state based on the transition matrix probabilities
        new_state = torch.multinomial(self.transition_matrix[action], 1).item()
        reward = 0
        terminated = False
        truncated = False
        info = dict()

        return new_state, reward, terminated, truncated, info
