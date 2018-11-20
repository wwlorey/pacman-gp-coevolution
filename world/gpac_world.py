import controllers.direction as d
import copy
import gp.world_file as world_file_class
import random
import world.coordinate as coord_class
import world.gpac_chars as chars
import world.wall_carver as wall_carver_class


class GPacWorld:
    def __init__(self, config, initial_instance=False):
        """Initializes the GPacWorld class.
        
        Where config is a Config object for the GPac problem and initial_instance
        determines if a random world should be generated yet.
        """
        self.config = config

        # Load configuration file settings
        self.width = int(self.config.settings['width'])
        self.height = int(self.config.settings['height'])
        self.pill_density = float(self.config.settings['pill density']) / 100
        self.wall_density = float(self.config.settings['wall density']) / 100
        self.num_pacmen = int(self.config.settings['num pacmen'])
        self.num_ghosts = int(self.config.settings['num ghosts'])
        self.fruit_spawn_prob = float(self.config.settings['fruit spawn probability'])
        self.fruit_score = int(self.config.settings['fruit score'])
        self.time_multiplier = int(self.config.settings['time multiplier'])

        # Create initial world attributes
        self.pacman_coords= [coord_class.Coordinate(0, self.height - 1) for _ in range(self.num_pacmen)]
        self.ghost_coords = [coord_class.Coordinate(self.width - 1, 0) for _ in range(self.num_ghosts)]
        self.wall_coords = set([])
        self.pill_coords = set([])
        self.fruit_coord = set([])
        self.time_remaining = self.time_multiplier * self.width * self.height
        self.total_time = self.time_remaining
        self.num_pills_consumed = 0
        self.num_fruit_consumed = 0
        self.score = 0

        self.prev_pacman_coords = []
        for pacman_coord in self.pacman_coords:
            self.prev_pacman_coords.append(coord_class.Coordinate(pacman_coord.x, pacman_coord.y))

        # BONUS2 (include a taboo list of pacmen to help with indexing)
        self.dead_pacmen = set([])

        self.prev_ghost_coords = []
        for ghost_coord in self.ghost_coords:
            self.prev_ghost_coords.append(coord_class.Coordinate(ghost_coord.x, ghost_coord.y))
        
        # Create helper set of all coordinates
        self.all_coords = set([])

        for x in range(self.width):
            for y in range(self.height):
                self.all_coords.add(coord_class.Coordinate(x, y))

        # Place walls and pills in the world
        # Only do this if the world is not being created for the first time
        # Subsequent runs of the GP will ensure random board generation
        if not initial_instance:
            self.generate_world()

        # Create & write to world file
        self.world_file = world_file_class.WorldFile(self.config)
        self.world_file.save_first_snapshot(self.width, self.height, self.pacman_coords,
            self.wall_coords, self.ghost_coords, self.pill_coords, self.time_remaining)


    def generate_world(self):
        """Randomly generates a GPac world (in place) by placing walls and pills
        with densities found in self.config.
        """

        def add_walls():
            """Places (carves out) walls in the world such that every non-wall cell is 
            accessible from every other non-wall cell.
            """

            def assign_unit_starting_coords(wall_carver_list):
                """Assigns the first two WallCarvers in wall_carver_list to have 
                the unit (pacman & ghosts) starting coords, ensuring these
                coords are carved out.
                """
                wall_carver_list[0].coord = self.pacman_coords[0]
                wall_carver_list[1].coord = self.ghost_coords[0]


            # Create WallCarver population
            # Note: num wall carvers should be greater than or equal to two
            wall_carvers = [wall_carver_class.WallCarver(random.randint(0, self.width), random.randint(0, self.height), self.config) for _ in range(int(self.config.settings['num wall carvers']))]
            assign_unit_starting_coords(wall_carvers)

            # Get walls to carve
            walls_to_carve = []
            for coord in self.all_coords:
                walls_to_carve.append(coord_class.Coordinate(coord.x, coord.y))

            past_seen_carved_coords = set([])

            # Calculate wall density
            num_coords = len(walls_to_carve)
            num_walls = num_coords
            wall_density = num_walls / num_coords
            completed_circuit = False

            while wall_density > self.wall_density or not completed_circuit:
                if not len(wall_carvers):
                    # Re-spawn WallCarvers
                    wall_carvers = []

                    for i in range(int(self.config.settings['num respawn wall carvers'])):
                        spawn_coord = random.choices(list(walls_to_carve))[0]
                        wall_carvers.append(wall_carver_class.WallCarver(spawn_coord.x, spawn_coord.y, self.config))

                    completed_circuit = False

                for wall_carver in wall_carvers:
                    if wall_carver.coord_already_carved(wall_carvers) or wall_carver.coord in past_seen_carved_coords:
                        # This coord has already been seen by another WallCarver
                        wall_carver.mark_for_death()

                    elif wall_carver.coord in walls_to_carve:
                        # This is a new coord. Remove the wall
                        walls_to_carve.remove(wall_carver.coord)
                        wall_carver.seen_coords.add(wall_carver.coord)
                        num_walls -= 1

                    wall_carver.move()

                # Update past seen carved coordinates with coordinate sets of agents marked for death
                for marked_wall_carver in [wall_carver for wall_carver in wall_carvers if wall_carver.marked_for_death]:
                    past_seen_carved_coords = past_seen_carved_coords.union(marked_wall_carver.seen_coords)

                # Kill WallCarvers marked for death
                wall_carvers = [wall_carver for wall_carver in wall_carvers if not wall_carver.marked_for_death]

                if not len(wall_carvers):
                    completed_circuit = True

                wall_density = num_walls / num_coords

            self.wall_coords = walls_to_carve


        # Add walls to the world
        add_walls()

        # Add pills to the world
        for c in self.all_coords.difference(set(self.pacman_coords)).difference(self.wall_coords):
            if random.random() < self.pill_density:
                self.pill_coords.add(c)

        # Ensure at least one pill was placed
        if not len(self.pill_coords):
            for c in self.all_coords.difference(set(self.pacman_coords)).difference(self.wall_coords):
                self.pill_coords.add(c)
                break


    def move_pacmen(self, direction, pacman_index):
        """Moves the pacman in self.pacman_coords at pacman_index in direction, where the direction 
        leads pacman to a valid location.

        If a pacman is marked as dead, it will not be moved.
        """
        # BONUS2
        if self.pacman_coords[pacman_index] in self.dead_pacmen:
            # This pacman is dead, don't move it
            return

        if direction == d.Direction.NONE:
            # No action needed
            return

        new_coord = coord_class.Coordinate(self.pacman_coords[pacman_index].x, self.pacman_coords[pacman_index].y)
        
        # Adjust new_coord depending on pacman's desired direction
        if direction == d.Direction.UP:
            new_coord.y += 1

        elif direction == d.Direction.DOWN:
            new_coord.y -= 1

        elif direction == d.Direction.LEFT:
            new_coord.x -= 1

        elif direction == d.Direction.RIGHT:
            new_coord.x += 1
        
        self.prev_pacman_coords[pacman_index] = coord_class.Coordinate(self.pacman_coords[pacman_index].x, self.pacman_coords[pacman_index].y)
        self.pacman_coords[pacman_index] = coord_class.Coordinate(new_coord.x, new_coord.y)


    def move_ghost(self, ghost_id, direction):
        """Moves the given ghost in direction, where direction is leads the
        ghost to a valid location.
        """
        if ghost_id >= len(self.ghost_coords):
            # This ghost does not exist
            return

        new_coord = coord_class.Coordinate(self.ghost_coords[ghost_id].x, self.ghost_coords[ghost_id].y)
        
        # Adjust new_coord depending on pacman's desired direction
        if direction == d.Direction.UP:
            new_coord.y += 1

        elif direction == d.Direction.DOWN:
            new_coord.y -= 1

        elif direction == d.Direction.LEFT:
            new_coord.x -= 1

        elif direction == d.Direction.RIGHT:
            new_coord.x += 1
        
        self.prev_ghost_coords[ghost_id] = coord_class.Coordinate(self.ghost_coords[ghost_id].x, self.ghost_coords[ghost_id].y)
        self.ghost_coords[ghost_id] = coord_class.Coordinate(new_coord.x, new_coord.y)


    def check_game_over(self):
        """Returns True if the game is over, False otherwise.

        The game is over if any of the following is true:
            1. a pacman and a ghost are in the same cell, leaving no other surviving pacman
            2. a pacman collided with a ghost, leaving no other surviving pacman
            3. all pills are gone
            4. time remaining is equal to zero
        """
        # BONUS2
        for pacman_index, pacman_coord in enumerate(self.pacman_coords):
            if pacman_coord in self.ghost_coords:
                self.dead_pacmen.add(self.pacman_coords[pacman_index])
                if len(self.dead_pacmen) == len(self.pacman_coords):
                    return True

            for ghost_index in range(self.num_ghosts):
                if self.prev_pacman_coords[pacman_index] == self.ghost_coords[ghost_index] and self.pacman_coords[pacman_index] == self.prev_ghost_coords[ghost_index]:
                    self.dead_pacmen.add(self.pacman_coords[pacman_index])
                    if len(self.dead_pacmen) == len(self.pacman_coords):
                        return True

        if not len(self.pill_coords):
            return True

        if not self.time_remaining:
            return True

        return False # The game continues


    def visualize(self, wall_coords=None):
        """Prints the world and its contents to the screen, with an option
        to pass in alternate wall coordinates.
        """
        if not wall_coords:
            wall_coords = self.wall_coords

        world = [[chars.GPacChars.EMPTY for _ in range(self.width)] for _ in range(self.height)]
        
        for c in self.pill_coords:
            world[c.x][c.y] = chars.GPacChars.PILL

        for c in self.fruit_coord:
            world[c.x][c.y] = chars.GPacChars.FRUIT

        for c in self.ghost_coords:
            world[c.x][c.y] = chars.GPacChars.GHOST

        for c in wall_coords:
            world[c.x][c.y] = chars.GPacChars.WALL

        for c in self.pacman_coords:
            world[c.x][c.y] = chars.GPacChars.PACMAN

        for row in range(self.width):
            for col in range(self.height - 1, -1, -1):
                print(world[row][col].value, end=' ')

            print('\n')

        print('\n')

    
    def get_adj_coords(self, coord):
        """Returns a list of coordinates adjacent to coord.

        Where the returned coordinate list includes only valid coordinates.
        """
        adj_coords = []

        if not coord.x == 0:
            adj_coords.append(coord_class.Coordinate(coord.x - 1, coord.y))

        if not coord.x == self.width - 1:
            adj_coords.append(coord_class.Coordinate(coord.x + 1, coord.y))
        
        if not coord.y == 0:
            adj_coords.append(coord_class.Coordinate(coord.x, coord.y - 1))
        
        if not coord.y == self.height - 1:
            adj_coords.append(coord_class.Coordinate(coord.x, coord.y + 1))
        
        return adj_coords


    def update_score(self):
        self.score = int((self.num_pills_consumed / (self.num_pills_consumed + len(self.pill_coords))) * 100) + (self.num_fruit_consumed * self.fruit_score)

        if not len(self.pill_coords):
            # No more pills in the world
            self.score += self.time_remaining // self.total_time
        

    def randomly_spawn_fruit(self):
        """Probabilistically spawns a fruit in the world.

        Note that if a fruit already exists, another cannot be spawned.

        A fruit can only spawn in cell that is not occupied by pacman, not
        occupied by a wall, and not occupied by a pill.
        """
        if len(self.fruit_coord):
            # A fruit already exists
            return

        if random.random() <= self.fruit_spawn_prob:
            possible_coords = []
            for coord in list(self.all_coords.difference(set(self.pacman_coords)).difference(self.wall_coords).difference(self.pill_coords)):
                possible_coords.append(coord_class.Coordinate(coord.x, coord.y))

            random.shuffle(possible_coords)

            for possible_fruit_coord in possible_coords:
                self.fruit_coord.add(possible_fruit_coord)
                break

