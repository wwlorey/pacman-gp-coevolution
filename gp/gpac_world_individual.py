class GPacWorldIndividual:
    def __init__(self, world, game_state, pacman_conts, ghost_conts, fitness=0):
        """Initializes the GPacWorldIndividual class."""
        self.world = world
        self.game_state = game_state
        self.pacman_conts = pacman_conts
        self.ghost_conts = ghost_conts
        self.fitness = fitness
