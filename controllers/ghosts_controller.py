import controllers.base_controller as base_controller_class
import controllers.direction as d
import copy
import random
import world.coordinate as coord_class


class GhostsController(base_controller_class.BaseController):
    def __init__(self, config):
        """Initializes the GhostsController class."""
        self.config = config

        super(base_controller_class.BaseController, self).__init__()

        self.POSSIBLE_MOVES = [d.Direction.UP, d.Direction.DOWN, d.Direction.LEFT,
            d.Direction.RIGHT]


    def get_move(self, ghost_id, game_state):
        """Produces a move based on game_state.

        Note: for assignment 2b, the move is randomized.
        """
        while True:
            # Choose a random direction
            direction_to_try = random.choice(self.POSSIBLE_MOVES)

            # Determine if this direction is valid
            new_coord = coord_class.Coordinate(game_state.ghost_coords[ghost_id].x, game_state.ghost_coords[ghost_id].y)
            if direction_to_try == d.Direction.UP:
                new_coord.y += 1

            elif direction_to_try == d.Direction.DOWN:
                new_coord.y -= 1

            elif direction_to_try == d.Direction.LEFT:
                new_coord.x -= 1

            elif direction_to_try == d.Direction.RIGHT:
                new_coord.x += 1
            
            if self.check_valid_location(new_coord, game_state):
                break # We have found a valid direction

        return direction_to_try

