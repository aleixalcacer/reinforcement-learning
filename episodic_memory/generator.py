class Generator(object):
    def __init__(self, jump_rate=15.0, symmetrize=False):
        self.jump_rate = jump_rate
        self.symmetrize = symmetrize

        self.states = None
        self.transitions = None
        self.transition_matrix = None



