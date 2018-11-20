# BONUS2 (changed pacman_cont to pacman_conts to signify multiple controllers)
class GPacWorldIndividual:
    def __init__(self, world, game_state, pacman_conts, ghosts_cont, fitness=0):
        """Initializes the GPacWorldIndividual class."""
        self.world = world
        self.game_state = game_state
        self.pacman_conts = pacman_conts
        self.ghosts_cont = ghosts_cont
        self.fitness = fitness
