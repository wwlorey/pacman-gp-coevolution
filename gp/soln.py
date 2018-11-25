import world.coordinate as coord_class


LIST_HEADER_STR = '###################################\n# State Evaluator Tree (list format)\n###################################\n'
EQUATION_HEADER_STR = '###################################\n# State Evaluator Equation\n###################################\n'


class Solution:
    def __init__(self, config):
        """Initializes the Solution class.

        Where config is a Config object.
        """
        self.config = config

    
    def write_to_file(self, individual):
        """Writes the given solutions (PacmanController and GhostController objects) 
        to the solution files specified in self.config.
        """
        file_names = [self.config.settings['pacman soln file path'], self.config.settings['ghost soln file path']]
        cont_list = [individual.pacman_conts, individual.ghost_conts]

        for i, file_name in enumerate(file_names):
            file = open(file_name, 'w')

            file.write(LIST_HEADER_STR)
            
            for cont in cont_list[i]:
                file.write(str([n.value for n in cont.state_evaluator]))
                file.write('\n')

            file.write('\n')

            file.write(EQUATION_HEADER_STR)
            for cont in cont_list[i]:
                file.write(cont.visualize(print_output=False))
                file.write('\n')

            file.close()
