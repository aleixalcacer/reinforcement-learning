import torch
from torch.distributions import Categorical


class Simulator(object):
    def __init__(self, propagator, no_dwell: bool = True, mass: int = 1):
        self.no_dwell = no_dwell
        # check mass is an integer
        if not isinstance(mass, int):
            raise ValueError("Mass must be an integer")
        self.mass = mass
        self.propagator = propagator

    def sample_state(self, rho, prev_states=None):
        """
        Samples a single state from the propagator.
        """

        if (self.no_dwell and self.mass != 0) and prev_states is not None:
            # Sample away from previous states
            except_states = prev_states.type(torch.int64)
            except_states = except_states[-min(self.mass, len(except_states)):]  #   "memory" of states
            rho[except_states] = 0.
            rho = rho / rho.sum()

        m = Categorical(rho)
        state = m.sample()
        log_prob = m.log_prob(state)
        if self.propagator.generator.environment.rewards is not None:
            reward = self.propagator.generator.environment.rewards[state]
        else:
            reward = 0

        return state, log_prob, reward

    def norm_density(self, V, beta=1.0, type="l1"):
        """
        FUNCTION: normalize to [0,1].
        INPUTS: V = values to normalize ("negative energies")
                beta = "inverse temperature" scaling
                type = type of normalization, L1, boltzmann
        """
        V[torch.isinf(V)] = 0.0
        # shift into positive range
        # (alpha>1, sometimes results in negative Y values presumably due to precision issues)
        if (V < 0).any():
            V = V - V.min()
        if type == "l1":
            P = V / V.sum()
        else:
            raise ValueError("Unknown normalization requested.")
        return P

    @staticmethod
    def check_state_distribution(rho):
        """
        Checks that the state distribution is valid.
        """
        if (rho < 0).any():
            rho[rho < 0] = 0
            print(Warning(f"State distribution must be non-negative, cropping to 0"))
        if (rho > 1).any():
            rho[rho > 1] = 1
            print(ValueError("State distribution must be less than or equal to 1"))

        # Check that the state distribution sums to 1
        # if not torch.allclose(rho.sum(), torch.ones(1)):
        #     raise ValueError(f"State distribution must sum to 1 (not {rho.sum()})")

        rho = rho / rho.sum()

        return rho

    def process_rho(self, state, n_states):
        rho_inter = torch.zeros(n_states)
        rho_inter[state] = 1.0
        return rho_inter

    def evolve(self, rho_start):
        """
        Evolves the state distribution according to the propagator.
        """
        # Evolve the state distribution
        rho_stop = rho_start @ self.propagator.P
        rho_stop = self.check_state_distribution(rho_stop)
        rho_stop = self.norm_density(rho_stop, type='l1')  # L1 normalize

        return rho_stop

    def sample_states(self, num_states=100, rho_start=None):
        """
        Samples states from the propagator.
        """
        if rho_start is None:
            rho_start = torch.ones(len(self.propagator.generator.environment.states))
            rho_start = rho_start / rho_start.sum()

        # Sample states
        state, log_prob, reward = self.sample_state(rho_start)
        states = [state]
        log_probs = [log_prob]
        rewards = [reward]

        # rho_inter is 1 in state else 0
        rho_inter = self.process_rho(state, len(rho_start))

        for i in range(1, num_states):
            rho_stop = self.evolve(rho_inter)
            state, log_prob, reward = self.sample_state(rho_stop, prev_states=torch.Tensor(states))
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            rho_inter = self.process_rho(state, len(rho_start))

        return states, log_probs, rewards

