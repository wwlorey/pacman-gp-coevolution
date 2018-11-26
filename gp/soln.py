import world.coordinate as coord_class


LIST_HEADER_STR = '###################################\n# State Evaluator Tree (list format)\n###################################\n'
EQUATION_HEADER_STR = '###################################\n# State Evaluator Equation\n###################################\n'


class Solution:
    def __init__(self, config):
        """Initializes the Solution class.

        Where config is a Config object.
        """
        self.config = config

    
    def write_to_file(self, conts, unit_type='pacman'):
        """Writes the given solutions (PacmanController and GhostController objects) 
        to the solution files specified in self.config.
        """
        if unit_type == 'pacman':
            file_name = self.config.settings['pacman soln file path']

        else:
            # Default to ghost
            file_name = self.config.settings['ghost soln file path']

        with open(file_name, 'w') as file:
            file.write(LIST_HEADER_STR)
            
            for cont in conts:
                file.write(str([n.value for n in cont.state_evaluator]))
                file.write('\n')

            file.write('\n')

            file.write(EQUATION_HEADER_STR)
            for cont in conts:
                file.write(cont.visualize(print_output=False))
                file.write('\n')
