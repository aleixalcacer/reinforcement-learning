from board import Board, get_all_states
import numpy as np

class TDAgent(object):
    def __init__(self, symbol, learning_rate=0.1, epsilon=0.1, gamma=0.9):
        self.symbol = symbol
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.policy = dict()
        all_states = get_all_states()
        for hash, (board, is_end) in all_states.items():
            if is_end:
                if board.get_winner() == self.symbol:
                    self.policy[hash] = 1
                else:
                    self.policy[hash] = 0
            else:
                self.policy[hash] = 0.5

        self.states = [Board()]
        self.greedy = [False]

    def reset(self):
        self.states = []
        self.greedy = []

    def act(self, board):
        if np.random.rand() < self.epsilon:
            max_action = np.random.choice(np.where(board.data.flatten() == 0)[0])
            greedy = False
        else:
            max_value = -np.inf
            max_action = None
            for action in np.where(board.data == 0)[0]:
                next_board = board.next_state(action, self.symbol)
                value = self.policy[next_board.hash()]
                if value > max_value:
                    max_value = value
                    max_action = action
            greedy = True

        next_board = board.next_state(max_action, self.symbol)

        self.states.append(next_board)
        self.greedy.append(greedy)

        return max_action


    def backpropagate(self):
        states = [board.hash() for board in self.states]

        for i in reversed(range(len(states) - 1)):
            td_error = self.greedy[i + 1] * (
                    self.policy[states[i + 1]] - self.policy[states[i]]
            )
            self.policy[states[i]] += self.learning_rate * td_error
