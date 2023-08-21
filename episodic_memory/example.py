from environment import EpisodicEnv
from generator import Generator
from propagator import Propagator
from simulator import Simulator
import torch

import numpy as np
import pandas as pd

df_words_similarities = pd.read_csv('data/words_similarities.csv', index_col=0)
df_words_similarities[df_words_similarities < 0] = 0
df_words_similarities[df_words_similarities > 1] = 1

df_locations_similarities = pd.read_csv('data/locations_similarities.csv', index_col=0)

df_env_states = pd.read_csv('data/environment_states.csv', index_col=0)
df_env_novel_states = pd.read_csv('data/environment_novel_states.csv', index_col=0)
# Select 5 rows
df_env_novel_states = df_env_novel_states.sample(5)

# Scale time to be between 0 and 1
df_env_states['time'] = df_env_states['time'] / df_env_states['time'].max()

# Mutate locations column to have integers
locations = df_env_states['location'].unique()
locations_dict = {location: i for i, location in enumerate(locations)}
inv_locations_dict = {i: location for i, location in enumerate(locations)}

df_env_states['location'] = df_env_states['location'].map(locations_dict)

env = EpisodicEnv(states=df_env_states, semantic_similarities=df_words_similarities,
                  k=1, m=1, n=1)

generator = Generator(env)

propagator = Propagator(generator)
# propagator.min_auto_cf(rho_init=1)

simulator = Simulator(propagator)

states = simulator.sample_states(num_states=10)

print(states)
