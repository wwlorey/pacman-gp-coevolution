import controllers.base_controller as base_controller_class
import controllers.direction as d
import controllers.function_nodes as func_nodes_class
import controllers.ghost_terminal_nodes as term_nodes_class
import controllers.tree as tree_class
import copy
import random
import world.coordinate as coord_class


# Assign new node class names
terminals = term_nodes_class.TerminalNodes
functions = func_nodes_class.FunctionNodes

# Constant declarations
POSSIBLE_MOVES = [d.Direction.UP, d.Direction.DOWN, d.Direction.LEFT,
    d.Direction.RIGHT]

ARBITRARY_LARGE_NUMBER = 99999


class GhostController(base_controller_class.BaseController):
    def __init__(self, config):
        """Initializes the GhostController class."""
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
            
            if allow_premature_end and target_height_met and current_depth < target_height and random.random() < float(self.config.settings['ghost premature end probability']):
                # Prematurely end this branch
                self.state_evaluator.add_node_left(parent_node, self.get_rand_terminal_node())
                self.state_evaluator.add_node_right(parent_node, self.get_rand_terminal_node())
                return

            # Continue constructing the tree
            self.state_evaluator.add_node_left(parent_node, self.get_rand_function_node())
            self.state_evaluator.add_node_right(parent_node, self.get_rand_function_node())

            init_state_evaluator_recursive(self.state_evaluator.get_left_child(parent_node), current_depth + 1)
            init_state_evaluator_recursive(self.state_evaluator.get_right_child(parent_node), current_depth + 1)


        target_height = random.randint(2, int(self.config.settings['ghost max tree generation height']))
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
        terminal_node = random.choice([node for node in terminals])

        if terminal_node == terminals.FP_CONSTANT:
            # Return a floating point constant
            return random.uniform(0, self.max_fp_constant)

        return terminal_node


    def get_rand_function_node(self):
        """Returns a random function node."""
        return random.choice([node for node in functions])


    def get_move(self, game_state, ghost_index):
        """Checks all possible moves against the state evaluator and 
        determines which move is optimal, returning that move for the ghost at ghost_index
        in game_state.
        """

        def move_ghost(ghost_coord, direction):
            """Alters ghost's new coordinate to indicate movement in direction.
            
            Returns True if the new position is valid, False otherwise.
            """
            if direction == d.Direction.NONE:
                # No action needed
                return True

            # Adjust new_coord depending on ghost's desired direction
            if direction == d.Direction.UP:
                ghost_coord.y += 1

            elif direction == d.Direction.DOWN:
                ghost_coord.y -= 1

            elif direction == d.Direction.LEFT:
                ghost_coord.x -= 1

            elif direction == d.Direction.RIGHT:
                ghost_coord.x += 1
            
            if self.check_valid_location(ghost_coord, game_state):
                return True

            return False


        best_eval_result = -1 * ARBITRARY_LARGE_NUMBER
        best_eval_direction = d.Direction.NONE

        for direction in POSSIBLE_MOVES:
            tmp_ghost_coord = coord_class.Coordinate(game_state.ghost_coords[ghost_index].x, game_state.ghost_coords[ghost_index].y)

            if move_ghost(tmp_ghost_coord, direction):
                eval_result = self.evaluate_state(game_state, tmp_ghost_coord)

                if eval_result > best_eval_result:
                    best_eval_result = eval_result
                    best_eval_direction = direction

        
        return best_eval_direction


    def evaluate_state(self, game_state, ghost_coord):
        """Given a current (or potential) game state and a new ghost coordinate, a rating
        is provided from the state evaluator.
        """

        def get_nearest_distance(ghost_coord, object):
            """Returns the distance between the given ghost coordinate
            and the nearest instance of type object.
            """

            def get_distance(coord1, coord2):
                """Returns the Manhattan distance between the given coordinates."""
                if not coord1 or not coord2:
                    return -1

                return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)


            if object == 'ghost':
                coords_to_search = game_state.ghost_coords
                coords_to_search = [coord_class.Coordinate(c.x, c.y) for c in game_state.ghost_coords if not c.x == ghost_coord.x and not c.y == ghost_coord.y]
            
            elif object == 'pacman':
                coords_to_search = game_state.pacman_coords

            else:
                coords_to_search = []
                
            min_distance = ARBITRARY_LARGE_NUMBER
            for coord in coords_to_search:
                min_distance = min(min_distance, get_distance(ghost_coord, coord))
            
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
                nonlocal pacman_distance

                ret = 0

                if node.value == terminals.NEAREST_GHOST_DIST:
                    ret = ghost_distance
                
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

        
        ghost_distance = get_nearest_distance(ghost_coord, 'ghost')
        pacman_distance = get_nearest_distance(ghost_coord, 'pacman')

        return evaluate_state_recursive(self.state_evaluator.get_root())
        

    def visualize(self, print_output=True):
        """Prints a function representing the state evaluator.
        
        If print_output is True, the output is printed. Otherwise, it 
        is returned as a string.
        """

        def get_symbol(node):
            """Returns a symbol (string) associated with the given node."""
            if node.value == terminals.NEAREST_PACMAN_DIST:
                return 'pacman distance'

            if node.value == terminals.NEAREST_GHOST_DIST:
                return 'ghost distance'

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

        
        target_height = random.randint(int(self.config.settings['ghost min tree mutation height']), int(self.config.settings['ghost max tree mutation height']))

        self.state_evaluator[starting_node.index].value = self.get_rand_function_node()
        grow_recursive(starting_node)


    def get_num_nodes(self):
        """Returns the number of non-None nodes in self.state_evaluator."""
        return len([node for node in self.state_evaluator if node.value])
