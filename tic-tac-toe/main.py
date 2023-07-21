from tictactoe import TicTacToeEnv, HumanAgent, RandomAgent
from td import TDAgent
import tqdm
import numpy as np

env = TicTacToeEnv()

from gym.utils.env_checker import check_env

check_env(env)

env.reset()

p1 = TDAgent(1, epsilon=0.01)
p2 = TDAgent(-1, epsilon=0.01)

train_bar = tqdm.tqdm(range(50_000))
rewards = []
for i in train_bar:
    done = False
    env.reset()
    while not done:
        p1_action = p1.act(env.board)
        _, p1_reward, done, _, _ = env.step(p1_action)
        if done:
            break
        p2_action = p2.act(env.board)
        _, p2_reward, done, _, _ = env.step(p2_action)
        if done:
            break
    if p1_reward == 1:
        reward = 1
    elif p2_reward == 1:
        reward = -1
    else:
        reward = 0
    rewards.append(reward)

    p1.backpropagate()
    p2.backpropagate()
    p1.reset()
    p2.reset()

# plot rewards with a moving average of 20

avg_rewards = np.convolve(rewards, np.ones(20), 'valid') / 20

print(len(avg_rewards))
import matplotlib.pyplot as plt

plt.plot(avg_rewards)
plt.show()




p2 = HumanAgent(1)

while True:
    done = False
    env.reset()
    while not done:
        p1_action = p1.act(env.board)
        _, reward, done, _, _ = env.step(p1_action)
        if done:
            break
        p2_action = p2.act(env.board)
        _, reward, done, _, _ = env.step(p2_action)
        if done:
            break
