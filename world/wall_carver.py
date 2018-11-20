import controllers.direction as d
import copy
import random
import world.coordinate as coord_class


class WallCarver:
    def __init__(self, x, y, config):
        """Initializes the WallCarver class."""
        self.config = config
        self.max_travel_dist = int(self.config.settings['max wall carver travel distance'])
        self.min_travel_dist = int(self.config.settings['min wall carver travel distance'])
        self.max_x = int(self.config.settings['width'])
        self.max_y = int(self.config.settings['height'])

        self.POSSIBLE_MOVES = [d.Direction.UP, d.Direction.DOWN, d.Direction.LEFT, d.Direction.RIGHT]

        self.coord = coord_class.Coordinate(x, y)

        self.get_travel_distance()
        self.get_direction()

        self.seen_coords = set([])

        self.marked_for_death = False


    def __eq__(self, other):
        return self.travel_distance == other.travel_distance and self.coord == other.coord and self.direction == other.direction


    def get_travel_distance(self):
        """Randomly generates a new travel distance."""
        self.travel_distance = random.randint(self.min_travel_dist, self.max_travel_dist)


    def get_direction(self):
        """Randomly generates a new travel direction."""
        self.direction = random.choices(self.POSSIBLE_MOVES)[0]


    def move(self):
        """Moves the WallCarver, generating new travel parameters if the travel
        distance is reached.
        """
        if self.travel_distance <= 0:
            self.get_travel_distance()
            self.get_direction()

        new_coord = coord_class.Coordinate(self.coord.x, self.coord.y)

        if self.direction == d.Direction.UP:
            new_coord.y += 1

        elif self.direction == d.Direction.DOWN:
            new_coord.y -= 1

        elif self.direction == d.Direction.LEFT:
            new_coord.x -= 1

        elif self.direction == d.Direction.RIGHT:
            new_coord.x += 1
        
        if new_coord.x >= 0 and new_coord.y >= 0 and new_coord.x < self.max_x and new_coord.y < self.max_y:
            self.coord = coord_class.Coordinate(new_coord.x, new_coord.y)

        self.travel_distance -= 1


    def coord_already_carved(self, wall_carver_list):
        """Returns True if this WallCarver's coord has already been carved (which 
        is extrapolated from wall_carver_list). Returns False otherwise.
        """
        # Remove this WallCarver from wall_carver_list
        wall_carver_list = [wall_carver for wall_carver in wall_carver_list if wall_carver != self]
        
        for other_wall_carver in wall_carver_list:
            if self.coord in other_wall_carver.seen_coords:
                return True

        return False


    def mark_for_death(self):
        """Marks a WallCarver for death."""
        self.marked_for_death = True
