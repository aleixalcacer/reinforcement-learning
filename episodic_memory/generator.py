import torch
from environment import EpisodicEnv


class Generator(object):
    def __init__(self, environment: EpisodicEnv, jump_rate=15.0, symmetrize=False):
        self.environment = environment
        self.n_states = self.environment.n_states
        self.jump_rate = jump_rate
        self.symmetrize = symmetrize

        self.states = None
        self.transitions = None
        self.T = self.environment.transition_matrix
        self.O = stochastic_to_generator(self.T, self.jump_rate)

        eigenvalues, eigenvectors = torch.linalg.eig(self.O)
        eigenvectors = eigenvectors.detach()

        self.G = torch.real(eigenvectors)
        self.V = torch.real(eigenvalues)
        self.W = torch.real(eigenvectors.inverse())


    def spectral_matrix(self):
        """
        Returns the spectral matrix of the generator.
        """
        return torch.einsum('ij, jk->jik', self.G, self.W)


def check_generator(Q: torch.Tensor):
    """
    Checks if a matrix is a generator matrix.
    """
    # check matrix row sums are 0
    row_sums = Q.sum(dim=1)
    if not torch.allclose(row_sums, torch.zeros(len(row_sums))):
        raise ValueError("Row sums must be 0")

    # check diagonal is non-positive
    if not torch.all(Q.diag() <= torch.zeros(len(Q.diag()))):
        raise ValueError("Diagonal must be non-positive")

    # check non-diagonal is non-negative
    if not torch.all(Q[~torch.eye(Q.shape[0], dtype=torch.bool)] >=
                     torch.zeros(len(Q[~torch.eye(Q.shape[0], dtype=torch.bool)]))):
        raise ValueError("Non-diagonal must be non-negative")

    return True


def stochastic_to_generator(transition_matrix: torch.Tensor, jump_rate=15.0):
    """
    Converts a stochastic matrix to a generator matrix.
    OUTPUT: Q = generator matrix
    """
    Q = jump_rate * (transition_matrix - torch.eye(len(transition_matrix)))
    return Q

def generator_to_weighted(Q: torch.Tensor):
    """
    Converts a generator matrix to a weighted adjacency matrix.
    OUTPUT: W = weighted adjacency matrix
    """
    W = Q.clone()
    W[W < 0] = 0
    return W





