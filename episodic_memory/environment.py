import gymnasium as gym
from gymnasium.spaces import Tuple, Discrete, Box
import torch
import pandas as pd
from sklearn.manifold import MDS


class EpisodicEnv(gym.Env):
    def __init__(self, states: pd.DataFrame,
                 semantic_similarities: pd.DataFrame,
                 spatial_similarities: pd.DataFrame,
                 k=0, m=0, n=0, o=0):
        super(EpisodicEnv, self).__init__()
        self.states = states
        self.n_states = len(states)
        self.action_space = Discrete(len(states))
        self.observation_space = Discrete(len(states))

        self.rewards = None

        self.k = k
        self.m = m
        self.n = n
        self.o = o

        self.semantic_similarities = semantic_similarities
        self.spatial_similarities = spatial_similarities

        # apply multidimensional scaling to semantic and spatial similarities
        mds = MDS(n_components=1, dissimilarity='precomputed')

        self.semantic_1d = pd.DataFrame(mds.fit_transform(self.semantic_similarities))
        self.semantic_1d.index = self.semantic_similarities.index

        mds = MDS(n_components=1, dissimilarity='precomputed')
        self.spatial_1d = pd.DataFrame(mds.fit_transform(self.spatial_similarities))
        self.spatial_1d.index = self.spatial_similarities.index

        self.states_transformed = states.copy()
        self.states_transformed["word"] = self.semantic_1d.loc[self.states["word"]].values
        self.states_transformed["location"] = self.spatial_1d.loc[self.states["location"]].values

        self.current_state = None
        self.transition_matrix = None  # Same as policy
        self.reset()

    def reset(self, seed=None, **kwargs):
        self.current_state = None

        # create a transition matrix for RL
        self.transition_matrix = create_transition_matrix(self.states,
                                                          self.semantic_similarities,
                                                          self.spatial_similarities,
                                                          k=self.k, m=self.m, n=self.n, o=self.o)

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

    def set_reward(self, goal_states: pd.DataFrame):
        self.rewards = torch.zeros(len(self.states))
        for i, state in self.states.iterrows():
            self.rewards[i] = self._compute_reward(state, goal_states)

    def _compute_reward(self, state, goal_states: pd.DataFrame):
        """Computes reward for a given state."""
        # loop over rows in novel episode
        reward = 0
        for i, row in goal_states.iterrows():
            # compute distance between row word and state word
            sim = self.semantic_similarities.loc[state["word"]][row["word"]]
            sim += self.spatial_similarities.loc[state["location"]][row["location"]]

            # compute reward
            reward += sim

        return reward


def create_transition_matrix(states, semantic_similarities, spatial_similarities, k, m, n, o):
    """
    Creates an access matrix from a generator matrix.
             T = stochastic matrix
    """
    def compute_v(state_i, state_j):
        word_i = state_i["word"]
        word_j = state_j["word"]
        time_i = state_i["time"]
        time_j = state_j["time"]
        location_i = state_i["location"]
        location_j = state_j["location"]
        episode_i = state_i["episode"]
        episode_j = state_j["episode"]

        semantic_sim = semantic_similarities.loc[word_i][word_j]  # similarity
        temporal_sim = (1 - abs(time_i - time_j))
        spatial_sim = spatial_similarities.loc[location_i][location_j]

        # Model
        delta = 0 if episode_i == episode_j else 1
        V = k ** delta * semantic_sim ** m * temporal_sim ** n * spatial_sim ** o

        return V

    O = torch.zeros((len(states), len(states)))

    for i, state_i in states.iterrows():
        for j, state_j in states.iterrows():
            if i != j:
                O[i, j] = compute_v(state_i, state_j)
    for i in range(len(states)):
        O[i, i] = -O[i, ].sum()

    T = torch.zeros((len(states), len(states)))
    n = - torch.diag(O)

    for i in range(len(states)):
        for j in range(len(states)):
            if i != j:
                T[i, j] = O[i, j] / n[i]
            else:
                T[i, j] = 0

    return T
