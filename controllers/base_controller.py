class BaseController:
    def __init__(self, config):
        """Initializes the BaseController class."""
        self.config = config


    def check_valid_location(self, coord, game_state):
        """Returns True if the given coordinate indicates a valid location
        (i.e. a location that either a ghost or pacman can move to.

        Returns False otherwise.
        """
        return not coord in game_state.wall_coords and coord.x >= 0 and coord.y >= 0 and coord.x < int(self.config.settings['width']) and coord.y < int(self.config.settings['height'])
