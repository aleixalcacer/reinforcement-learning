import torch
import torch.nn as nn
import torch.nn.functional as F


# Using a neural network to learn our policy parameters
class PolicyNetwork(nn.Module):

    # Takes in observations and outputs actions
    def __init__(self, observation_space, action_space):
        super(PolicyNetwork, self).__init__()
        hidden_size = 256
        self.input_layer = nn.Linear(observation_space, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, action_space)

    # forward pass
    def forward(self, x):
        # input states
        x = self.input_layer(x)

        # relu activation
        x = F.relu(x)

        # hidden layer
        x = self.hidden_layer(x)

        # relu activation
        x = F.relu(x)

        # actions
        actions = self.output_layer(x)
        actions = F.relu(actions)

        # get softmax for a probability distribution
        action_probs = F.softmax(actions, dim=-1)

        return action_probs
