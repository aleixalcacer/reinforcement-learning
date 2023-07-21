class Propagator(object):
    def __init__(self, generator, sigma=1.0, tau=1.0):
        self.generator = generator

        # Check sigma is grater than 0 and convert to float
        if sigma <= 0:
            raise ValueError("Sigma must be greater than 0")
        self.sigma = float(sigma)

        # Check sigma is grater than 0 and convert to float
        if tau <= 0:
            raise ValueError("Tau must be greater than 0")
        self.tau = float(tau)

