import world.coordinate as coord_class


class WorldFile:
    def __init__(self, config):
        """Initializes the WorldFile class.

        Where config is a Config object.
        """
        self.config = config
        self.file_str = ''
        self.prev_seen_fruit = coord_class.Coordinate(-1, -1)


    def save_first_snapshot(self, width, height, pacmen, walls, ghosts, pills, remaining_time):
        """Saves the initial snapshot to the world file string."""
        self.save_line(width)
        self.save_line(height)

        for pacman in pacmen:
            self.save_triplet('m', pacman.x, pacman.y)

        for c in walls:
            self.save_triplet('w', c.x, c.y)

        for ghost_id in range(len(ghosts)):
            self.save_triplet(ghost_id + 1, ghosts[ghost_id].x, ghosts[ghost_id].y)

        for c in pills:
            self.save_triplet('p', c.x, c.y)

        self.save_triplet('t', remaining_time, 0)

        
    def save_snapshot(self, pacmen, ghosts, fruit, remaining_time, score):
        """Saves a non-initial snapshot to the world file string."""
        for pacman in pacmen:
            self.save_triplet('m', pacman.x, pacman.y)     

        for ghost_id in range(len(ghosts)):
            self.save_triplet(ghost_id + 1, ghosts[ghost_id].x, ghosts[ghost_id].y)

        for c in fruit:
            # A fruit coordinate should only be written once
            if self.prev_seen_fruit != c:
                self.prev_seen_fruit = c
                self.save_triplet('f', c.x, c.y)

        self.save_triplet('t', remaining_time, score)


    def save_line(self, data):
        """Saves a single line (including newline char) to the world file string."""
        self.file_str += str(data) + '\n'

    
    def save_triplet(self, data1, data2, data3):
        """Saves a space delimited, single line (including newline char) 
        containing the given triplet to the world file string."""
        self.file_str += str(data1) + ' ' + str(data2) + ' ' + str(data3) + '\n'


    def write_to_file(self):
        """Writes self.file_str to the world file."""
        file = open(self.config.settings['world file path'], 'w')
        file.write(self.file_str)
        file.close()

