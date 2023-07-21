import gym
from gym import spaces
import numpy as np


def get_all_states_rec(current_state, current_symbol, all_states):
    for i in range(9):
        if current_state.data[i] == 0:
            new_state = current_state.next_state(i, current_symbol)
            new_hash = new_state.hash()
            if new_hash not in all_states:
                is_end = new_state.is_end()
                all_states[new_hash] = (new_state, is_end)
                if not is_end:
                    get_all_states_rec(new_state, -current_symbol, all_states)


def get_all_states():
    current_symbol = 1
    current_state = Board()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_rec(current_state, current_symbol, all_states)
    return all_states


def check_board_status(board):
    data = board.data.reshape(3, 3)

    # Check rows
    for row in data:
        if sum(row) == 3:
            return 1
        elif sum(row) == -3:
            return -1

    # Check columns
    for col in zip(*data):
        if sum(col) == 3:
            return 1
        elif sum(col) == -3:
            return -1

    # Check diagonals
    if data[0][0] + data[1][1] + data[2][2] == 3:
        return 1
    elif data[0][0] + data[1][1] + data[2][2] == -3:
        return -1
    elif data[0][2] + data[1][1] + data[2][0] == 3:
        return 1
    elif data[0][2] + data[1][1] + data[2][0] == -3:
        return -1

    # Check if board is full
    if 0 not in data:
        return 0
    # Game is still in progress
    else:
        return None

class Board(object):
    def __init__(self):
        self.data = np.zeros(9, dtype=np.int8)
        self.hash_val = None
        self.end = None
        self.winner = None

    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.data):
                self.hash_val = self.hash_val * 3 + i + 1
        return self.hash_val

    def is_end(self):
        return False if check_board_status(self) is None else True

    def get_winner(self):
        if self.winner is None:
            self.winner = check_board_status(self) if self.is_end() else 0
        return self.winner

    def next_state(self, action, symbol):
        if self.data[action] != 0:
            raise ValueError(f"Invalid action: {action}. Space already occupied.")
        new_state = Board()
        new_state.data = np.copy(self.data)
        new_state.data[action] = symbol
        return new_state

    def print_board(self):
        for row in self.data:
            print(row)

    def __eq__(self, other):
        return self.hash() == other.hash()

    def __ne__(self, other):
        return self.hash() != other.hash()

    def __hash__(self):
        return self.hash()


class BoardSpace(spaces.Space):
    def __init__(self, seed=None):
        super(BoardSpace).__init__()
        self.observation_space = get_all_states()
        self.n = len(self.observation_space)
        self.seed = seed

    def sample(self, mask=None):
        return np.random.choice(self.observation_space)

    def contains(self, x):
        return x.hash() in self.observation_space

    def seed(self, seed=None):
        self.seed = seed
        return self.seed
