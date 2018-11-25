import controllers.base_controller as base_controller_class
import controllers.direction as d
import controllers.nodes as nodes_classes
import controllers.tree as tree_class
import copy
import random
import world.coordinate as coord_class


# Assign new node class names
terminals = nodes_classes.TerminalNodes
functions = nodes_classes.FunctionNodes


# Constant declarations
POSSIBLE_MOVES = [d.Direction.NONE, d.Direction.UP, d.Direction.DOWN, 
    d.Direction.LEFT, d.Direction.RIGHT]

ARBITRARY_LARGE_NUMBER = 99999


class PacmanController(base_controller_class.BaseController):
    def __init__(self, config):
        """Initializes the PacmanController class."""
        self.config = config

        super(base_controller_class.BaseController, self).__init__()

        self.max_fp_constant = float(self.config.settings['max fp constant'])

        self.init_state_evaluator()


    def init_state_evaluator(self):
        """Initializes this controller's state evaluator tree using the ramped half-and-half method."""
        target_height_met = False

        def init_state_evaluator_recursive(parent_node, current_depth=1):
            nonlocal target_height_met
            nonlocal allow_premature_end

            if current_depth + 1 == target_height:
                # End this branch
                target_height_met = True
                self.state_evaluator.add_node_left(parent_node, self.get_rand_terminal_node())
                self.state_evaluator.add_node_right(parent_node, self.get_rand_terminal_node())
                return
            
            if allow_premature_end and target_height_met and current_depth < target_height and random.random() < float(self.config.settings['premature end probability']):
                # Prematurely end this branch
                self.state_evaluator.add_node_left(parent_node, self.get_rand_terminal_node())
                self.state_evaluator.add_node_right(parent_node, self.get_rand_terminal_node())
                return

            # Continue constructing the tree
            self.state_evaluator.add_node_left(parent_node, self.get_rand_function_node())
            self.state_evaluator.add_node_right(parent_node, self.get_rand_function_node())

            init_state_evaluator_recursive(self.state_evaluator.get_left_child(parent_node), current_depth + 1)
            init_state_evaluator_recursive(self.state_evaluator.get_right_child(parent_node), current_depth + 1)


        target_height = random.randint(2, int(self.config.settings['max tree generation height']))
        self.state_evaluator = tree_class.Tree(self.config, self.get_rand_function_node())

        if random.random() < float(self.config.settings['ramped half-and-half probability']):
            # Use full initialization method
            allow_premature_end = False
        
        else:
            # Use grow initialization method
            allow_premature_end = True

        init_state_evaluator_recursive(self.state_evaluator.get_root())


    def get_rand_terminal_node(self):
        """Returns a random terminal node."""
        terminal_node = random.choices([node for node in terminals])[0]

        if terminal_node == terminals.FP_CONSTANT:
            # Return a floating point constant
            return random.uniform(0, self.max_fp_constant)

        return terminal_node


    def get_rand_function_node(self):
        """Returns a random function node."""
        return random.choices([node for node in functions])[0]


    def get_move(self, game_state, pacman_index):
        """Checks all possible moves against the state evaluator and 
        determines which move is optimal, returning that move for the pacman at pacman_index
        in game_state.
        """

        def move_pacman(pacman_coord, direction):
            """Alters pacman's new coordinate to indicate movement in direction.
            
            Returns True if the new position is valid, False otherwise.
            """
            if direction == d.Direction.NONE:
                # No action needed
                return True

            # Adjust new_coord depending on pacman's desired direction
            if direction == d.Direction.UP:
                pacman_coord.y += 1

            elif direction == d.Direction.DOWN:
                pacman_coord.y -= 1

            elif direction == d.Direction.LEFT:
                pacman_coord.x -= 1

            elif direction == d.Direction.RIGHT:
                pacman_coord.x += 1
            
            if self.check_valid_location(pacman_coord, game_state):
                return True

            return False


        best_eval_result = -1 * ARBITRARY_LARGE_NUMBER
        best_eval_direction = d.Direction.NONE

        for direction in POSSIBLE_MOVES:
            tmp_pacman_coord = coord_class.Coordinate(game_state.pacman_coords[pacman_index].x, game_state.pacman_coords[pacman_index].y)

            if move_pacman(tmp_pacman_coord, direction):
                eval_result = self.evaluate_state(game_state, tmp_pacman_coord)

                if eval_result > best_eval_result:
                    best_eval_result = eval_result
                    best_eval_direction = direction

        
        return best_eval_direction
         

    def evaluate_state(self, game_state, pacman_coord):
        """Given a current (or potential) game state and a new pacman coordinate, a rating
        is provided from the state evaluator.
        """

        def get_nearest_distance(pacman_coord, object):
            """Returns the distance between the given pacman coordinate
            and the nearest instance of type object.
            """

            def get_distance(coord1, coord2):
                """Returns the Manhattan distance between the given coordinates."""
                if not coord1 or not coord2:
                    return -1

                return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)


            if object == 'ghost':
                coords_to_search = game_state.ghost_coords
            
            elif object == 'pill':
                coords_to_search = game_state.pill_coords

            elif object == 'fruit' and game_state.fruit_coord:
                coords_to_search = game_state.fruit_coord

            elif object == 'pacman' and len(game_state.pacman_coords) > 1:
                coords_to_search = [coord_class.Coordinate(c.x, c.y) for c in game_state.pacman_coords if not c.x == pacman_coord.x and not c.y == pacman_coord.y]

            else:
                coords_to_search = []
                
            min_distance = ARBITRARY_LARGE_NUMBER
            for coord in coords_to_search:
                min_distance = min(min_distance, get_distance(pacman_coord, coord))
            
            return min_distance

        
        def evaluate_state_recursive(node):

            def is_last_function_node(node):
                """Returns True if this node's children are terminal nodes,
                False otherwise.
                """
                if not self.state_evaluator.is_leaf(node):
                    return False
                
                return self.state_evaluator.get_left_child().value in [n for n in terminals]
        

            def get_fp(node):
                """Returns the FP value associated with the given *terminal* node."""
                nonlocal ghost_distance
                nonlocal pill_distance
                nonlocal fruit_distance
                nonlocal num_adj_walls

                nonlocal pacman_distance

                ret = 0

                if node.value == terminals.PACMAN_GHOST_DIST:
                    ret = ghost_distance
                
                elif node.value == terminals.PACMAN_PILL_DIST:
                    ret = pill_distance

                elif node.value == terminals.PACMAN_FRUIT_DIST:
                    ret = fruit_distance

                elif node.value == terminals.NUM_ADJ_WALLS:
                    ret = num_adj_walls
                
                elif node.value == terminals.NEAREST_PACMAN_DIST:
                    ret = pacman_distance
                
                else:
                    ret = node.value
                
                return float(ret)


            def evaluate(operator, operands):
                """Evaluates the given operator and operands, producing a FP value."""
                if operator == functions.RANDOM_FLOAT:
                    return random.uniform(min(operands), max(operands))

                if operator == functions.MULTIPLY:
                    return operands[0] * operands[1]
                
                if operator == functions.DIVIDE:
                    if 0.0 in operands:
                        return 0.0
                    
                    else:
                        return operands[0] / operands[1]
                
                if operator == functions.ADD:
                    return operands[0] + operands[1]
                
                if operator == functions.SUBTRACT:
                    return operands[0] - operands[1]


            if self.state_evaluator.is_leaf(node):
                return get_fp(node)

            if is_last_function_node(node):
                return evaluate(node.value, [node.left(), node.right()])
            
            return evaluate(node.value, [evaluate_state_recursive(self.state_evaluator.get_left_child(node)), evaluate_state_recursive(self.state_evaluator.get_right_child(node))])

        
        ghost_distance = get_nearest_distance(pacman_coord, 'ghost')
        pill_distance = get_nearest_distance(pacman_coord, 'pill')
        fruit_distance = get_nearest_distance(pacman_coord, 'fruit')
        pacman_distance = get_nearest_distance(pacman_coord, 'pacman')
        
        num_adj_walls = game_state.num_adj_walls

        return evaluate_state_recursive(self.state_evaluator.get_root())
        

    def visualize(self, print_output=True):
        """Prints a function representing the state evaluator.
        
        If print_output is True, the output is printed. Otherwise, it 
        is returned as a string.
        """

        def get_symbol(node):
            """Returns a symbol (string) associated with the given node."""
            if node.value == terminals.PACMAN_GHOST_DIST:
                return 'ghost distance'

            if node.value == terminals.PACMAN_PILL_DIST:
                return 'pill distance'

            if node.value == terminals.PACMAN_FRUIT_DIST:
                return 'fruit distance'

            if node.value == terminals.NUM_ADJ_WALLS:
                return 'num adj walls'
            
            if node.value == functions.ADD:
                return '+'
                
            if node.value == functions.SUBTRACT:
                return '-'
                
            if node.value == functions.MULTIPLY:
                return '*'
                
            if node.value == functions.DIVIDE:
                return '/'
                
            if node.value == functions.RANDOM_FLOAT:
                return 'rand'

            return str(node.value)


        def visualize_recursive(node):
            if self.state_evaluator.is_leaf(node):
                return get_symbol(node)

            return '( ' + visualize_recursive(self.state_evaluator.get_left_child(node)) + ' ' + get_symbol(node) + ' ' + visualize_recursive(self.state_evaluator.get_right_child(node)) + ' )'


        output = visualize_recursive(self.state_evaluator.get_root())

        if print_output:
            print(output)
        
        else:
            return output


    def __copy__(self):
        """Performs a deep copy of this object, except for the state evaluator."""
        other = type(self)(self.config)
        super(base_controller_class.BaseController, other).__init__()
        other.config = self.config
        other.max_fp_constant = float(self.config.settings['max fp constant'])
        other.state_evaluator = tree_class.Tree(self.config)
        other.state_evaluator.list[:] = [tree_class.TreeNode(node.index, node.value) if node else None for node in self.state_evaluator]

        return other


    def grow(self, starting_node):
        """Randomly (re)grows a branch on state_evaluator starting at (and including) 
        starting_node up to target_height.
        """

        def grow_recursive(node, relative_depth=1):
            if relative_depth == target_height:
                self.state_evaluator.add_node_left(node, self.get_rand_terminal_node())
                self.state_evaluator.add_node_right(node, self.get_rand_terminal_node())
                return

            self.state_evaluator.add_node_left(node, self.get_rand_function_node())
            self.state_evaluator.add_node_right(node, self.get_rand_function_node())

            grow_recursive(self.state_evaluator.get_left_child(node), relative_depth + 1)
            grow_recursive(self.state_evaluator.get_right_child(node), relative_depth + 1)

        
        target_height = random.randint(int(self.config.settings['min tree mutation height']), int(self.config.settings['max tree mutation height']))

        self.state_evaluator[starting_node.index].value = self.get_rand_function_node()
        grow_recursive(starting_node)


    def get_num_nodes(self):
        """Returns the number of non-None nodes in self.state_evaluator."""
        return len([node for node in self.state_evaluator if node.value])
