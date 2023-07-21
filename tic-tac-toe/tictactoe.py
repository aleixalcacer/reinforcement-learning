import gym
from gym import spaces
import numpy as np
from board import BoardSpace, Board

class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(9)
        self.observation_space = BoardSpace()

        self.board = None
        self.symbol = None
        self.done = None
        self.reset()

    def reset(self, seed=None, **kwargs):
        self.board = Board()
        self.symbol = 1
        self.done = False
        super().reset(seed=seed, **kwargs)
        return self.board, dict()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be between 0 and 8.")

        self.board = self.board.next_state(action, self.symbol)

        if self.board.is_end():
            self.done = True
            winner = self.board.get_winner()
            info = dict(mark=winner)
            reward = 0 if winner is None else winner * self.symbol
        else:
            reward = 0
            info = dict(mark=None)

        self.symbol *= -1

        return self.board, reward, self.done, False, info



class HumanAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, board):
        print(board.data.reshape(3, 3))
        action = int(input("Enter action: "))
        return action

    def reset(self):
        pass


class RandomAgent(object):
    def __init__(self, mark):
        self.mark = mark

    def act(self, board):
        return np.random.choice(np.where(board.data == 0)[0])

    def reset(self):
        pass
