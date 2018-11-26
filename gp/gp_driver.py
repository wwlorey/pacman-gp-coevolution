import controllers.game_state as game_state_class
import controllers.ghost_controller as ghost_cont_class
import controllers.pacman_controller as pacman_cont_class
import controllers.tree as tree
import copy
import gp.controller_individual as cont_class
import gp.gpac_world_individual as gpac_world_individual_class
import gp.log as log_class
import gp.soln as soln_class
import math
import random
import util.seed as seed_class
import world.gpac_world as gpac_world_class


class GPDriver:
    def __init__(self, config):
        """Initializes the GPDriver class.
        
        Where config is a Config object.
        """
        self.config = config

        self.seed = seed_class.Seed(self.config)

        self.pacman_population_size = int(self.config.settings['pacman mu'])
        self.pacman_child_population_size = int(self.config.settings['pacman lambda'])
        self.pacman_parent_population_size = int(self.config.settings['pacman num parents'])

        self.ghost_population_size = int(self.config.settings['ghost mu'])
        self.ghost_child_population_size = int(self.config.settings['ghost lambda'])
        self.ghost_parent_population_size = int(self.config.settings['ghost num parents'])

        self.num_pacmen = int(self.config.settings['num pacmen'])
        self.num_ghosts = int(self.config.settings['num ghosts'])

        self.use_single_pacman_cont = self.config.settings.get_bool('use single pacman controller')
        self.use_single_ghost_cont = self.config.settings.get_bool('use single ghost controller')

        self.run_count = 1
        self.eval_count = 0
        self.local_best_score = -1
        self.avg_score = 0
        self.prev_avg_score = 0
        self.stale_score_count_termination = 0

        self.log = log_class.Log(self.config, self.seed, overwrite=True)
        self.soln = soln_class.Solution(self.config)

        self.local_best_score = -1
        self.global_best_score = -1

        self.gpac_world_population = []

        self.pacman_cont_population = []
        self.ghost_cont_population = []
        self.pacman_cont_parents = []
        self.ghost_cont_parents = []
        self.pacman_cont_children = []
        self.ghost_cont_children = []

        self.FITNESS_PROPORTIONAL_SELECTION_CHOICES = (self.config.settings.getboolean('pacman use fitness proportional parent selection'), self.config.settings.getboolean('ghost use fitness proportional parent selection'))
        self.X_OVERSELECTION_CHOICES = (float(self.config.settings['pacman x overselection']), float(self.config.settings['ghost x overselection']))
        self.CHILD_POPULATION_SIZES = (self.pacman_child_population_size, self.ghost_child_population_size)
        self.PARENT_POPULATION_SIZES = (self.pacman_parent_population_size, self.ghost_parent_population_size)
        self.PACMAN_ID = 0
        self.GHOST_ID = 1

        # TODO: remove
        self.population = []
        self.parents = []
        self.children = []


    def begin_run(self):
        """Initializes run variables and writes a run header
        to the log file. 

        This should be called before each run.
        """
        self.eval_count = 0
        self.local_best_score = -1
        self.avg_score = 0
        self.prev_avg_score = 0
        self.stale_score_count_termination = 0
        self.log.write_run_header(self.run_count)

        # Initialize the populations
        self.gpac_world_population = []
        self.pacman_cont_population = []
        self.ghost_cont_population = []

        for _ in range(self.population_size):
            world = gpac_world_class.GPacWorld(self.config)
            game_state = game_state_class.GameState(world.pacman_coords, world.ghost_coords, world.pill_coords, [self.get_num_adj_walls(world, pacman_coord) for pacman_coord in world.pacman_coords])
            game_state.update_walls(world.wall_coords)

            self.gpac_world_population.append(gpac_world_individual_class.GPacWorldIndividual(world, game_state))


        if self.use_single_pacman_cont:
            num_pacman_conts = 1

        else:
            num_pacman_conts = self.num_pacmen

        for _ in range(self.pacman_population_size):
            self.pacman_cont_population.append(cont_class.ControllerIndividual([pacman_cont_class.PacmanController(self.config) for _ in range(num_pacman_conts)]))


        if self.use_single_ghost_cont:
            num_ghost_conts = 1
        
        else:
            num_ghost_conts = self.num_ghosts

        for _ in range(self.ghost_population_size):
            self.ghost_cont_population.append(cont_class.ControllerIndividual([ghost_cont_class.GhostController(self.config) for _ in range(num_ghost_conts)]))


    def end_run(self):
        """Increments the run count by one.
        
        This should be called after each run.
        """
        self.run_count += 1


    def end_eval(self, individual):
        """Conditionally updates the log and world files, increments 
        the evaluation count, and resets the positions of each unit in the individual.

        This should be called after each evaluation.
        """
        individual.fitness = individual.world.score
        self.eval_count += 1

        individual.world.reset()
        individual.game_state.update(individual.world.pacman_coords, individual.world.ghost_coords, individual.world.pill_coords, [self.get_num_adj_walls(world, pacman_coord) for pacman_coord in world.pacman_coords])


    def control_bloat(self):
        """Adjusts the fitness of each individual in the population by applying 
        parsimony pressure.
        """
        controller_populations = [self.pacman_cont_population, self.ghost_cont_population]

        for unit_id, population in enumerate(controller_populations):
            avg_num_nodes = 0
            avg_num_nodes = int(sum([individual.cont.get_num_nodes() for individual in population]) / len(population))

            for individual in population:
                num_nodes = individual.cont.get_num_nodes()

                if  num_nodes > avg_num_nodes:
                    if unit_id == self.PACMAN_ID:
                        p = float(self.config.settings['pacman p parsimony coefficient'])

                    else:
                        # Default to self.GHOST_ID
                        p = float(self.config.settings['ghost p parsimony coefficient'])

                    individual.fitness /= p * (num_nodes - avg_num_nodes)
                    individual.fitness = int(individual.fitness)


    def evaluate(self, population):
        """TODO: How will the three populations be passed around & matched up?
        
        Evaluates all population members given in population by running
        each world's game until completion.
        """
        for individual in population:
            while self.check_game_over(individual):
                self.move_units(individual)

            self.end_eval(individual)

        self.check_update_log_world_files()


    def select_parents(self):
        """Chooses which parents from the population will breed.

        Depending on the parent selection configuration, one of the two following 
        methods is used to select parents for the pacmen and ghosts (independently configurable):
            1. Fitness proportional selection
            2. Over-selection

        The resulting parents are stored in self.pacman_cont_parents and self.ghost_cont_parents.
        """
        parent_populations = ([], []) # (new pacman controller parent population, new ghost controller parent population)
        cont_populations = (self.pacman_cont_population, self.ghost_cont_population)

        # Perform parent selection for the pacman and ghost controllers
        for unit_id in range(len(parent_populations)):
            parents = []
            parent_population_size = self.PARENT_POPULATION_SIZES[unit_id]

            if self.FITNESS_PROPORTIONAL_SELECTION_CHOICES[unit_id]:
                # Select parents for breeding using the fitness proportional "roulette wheel" method (with replacement)
                parent_fitnesses = [individual.fitness for individual in cont_populations[unit_id]]

                if max(parent_fitnesses) == min(parent_fitnesses):
                    # All parent fitnesses are the same so select parents at random
                    for _ in range(parent_population_size):
                        parents.append(cont_populations[unit_id][random.randrange(0, len(cont_populations[unit_id]))])

                else:
                    parent_populations[unit_id] = random.choices(cont_populations[unit_id], weights=parent_fitnesses, k=parent_population_size)

            else:
                # Default to over-selection parent selection
                elite_index_cuttoff = int(self.X_OVERSELECTION_CHOICES[unit_id]) * len(cont_populations[unit_id]))

                # Note: the following hardcoded percentages are specified as part of the overselection algorithm
                num_elite_parents = int(0.80 * parent_population_size)
                num_sub_elite_parents = int(0.20 * parent_population_size)

                self.sort_individuals(cont_populations[unit_id])
                elite_group = cont_populations[unit_id][:elite_index_cuttoff]
                sub_elite_group = cont_populations[unit_id][elite_index_cuttoff:]

                elite_choices = [individual.fitness for individual in elite_group]

                if max(elite_choices) == min(elite_choices):
                    # All elite individual fitnesses are the same so select individuals at random
                    for _ in range(num_elite_parents):
                        parent_populations[unit_id].append(elite_group.pop())
                
                else:
                    parent_populations[unit_id] = random.choices(elite_group, weights=elite_choices, k=num_elite_parents)

                sub_elite_choices = [individual.fitness for individual in sub_elite_group]

                if max(sub_elite_choices) == min(sub_elite_choices):
                    # All sub-elite individual fitnesses are the same so select individuals at random
                    for _ in range(num_sub_elite_parents):
                        parent_populations[unit_id].append(sub_elite_group.pop())
                
                else:
                    parent_populations[unit_id] += random.choices(sub_elite_group, weights=sub_elite_choices, k=num_sub_elite_parents)

        # Save parent selections to the parent population class members
        self.pacman_cont_parents = parent_populations[self.PACMAN_ID]
        self.ghost_cont_parents = parent_populations[self.GHOST_ID]
                 

    def recombine(self):
        """Breeds lambda (offspring pool size) children using sub-tree crossover 
        from the existing pacman and ghost parent populations. The resulting children are stored in 
        self.pacman_cont_children and self.ghost_cont_children.
        """

        def breed(parent_a, parent_b, unit_id):
            """Performs sub-tree crossover on parent_a and parent_b returning the child tree."""

            def crossover_recursive(receiver_index, donator_index):
                if receiver_index < len(child_conts[cont_index].state_evaluator) and donator_index < len(parent_cont.state_evaluator):
                    return
                
                child_conts[cont_index].state_evaluator[receiver_index] = tree.TreeNode(receiver_index, parent_cont.state_evaluator[donator_index].value)
                crossover_recursive(child_conts[cont_index].state_evaluator.get_left_child_index(receiver_index), parent_cont.state_evaluator.get_left_child_index(donator_index))
                crossover_recursive(child_conts[cont_index].state_evaluator.get_right_child_index(receiver_index), parent_cont.state_evaluator.get_right_child_index(donator_index))


            if unit_id == self.PACMAN_ID:
                num_conts = self.num_pacmen

            else:
                # Default to self.GHOST_ID
                num_conts = self.num_ghosts

            child_conts = [None] * num_conts
            for cont_index in range(num_conts):
                # Choose a random node (crossover point) from each state evaluator node list
                crossover_node_a = parent_a.conts[cont_index].state_evaluator[random.choices([n for n in parent_a.conts[cont_index].state_evaluator if n.value])[0].index]
                crossover_node_b = parent_b.conts[cont_index].state_evaluator[random.choices([n for n in parent_b.conts[cont_index].state_evaluator if n.value])[0].index]

                child_conts[cont_index] = copy.copy(parent_a.conts[cont_index])
                parent_cont = parent_b.conts[cont_index]

                # Extend the child's state evaluator if necessary
                if len(child_conts[cont_index].state_evaluator) < len(parent_a.conts[cont_index].state_evaluator):
                    child_conts[cont_index].state_evaluator = child_conts[cont_index].state_evaluator + [tree.TreeNode(index, None) for index in range(len(child_conts[cont_index].state_evaluator), len(parent_a.conts[cont_index].state_evaluator))]

                # Perform sub-tree crossover
                crossover_recursive(crossover_node_a.index, crossover_node_b.index)

            # Finish generating the child
            child = cont_class.ControllerIndividual(child_conts)
            return child


        child_populations = ([], []) # (new pacman controller child population, new ghost controller child population)
        parent_populations = (self.pacman_cont_parents, self.ghost_cont_parents)

        for unit_id in range(len(child_populations)):
            for _ in range(self.CHILD_POPULATION_SIZES[unit_id]):
                # Select parents with replacement
                # Note: this implementation allows for parent_a and parent_b to be the same genotype
                parent_a = parent_populations[unit_id][random.randrange(0, len(self.parents))]
                parent_b = parent_populations[unit_id][random.randrange(0, len(self.parents))]

                # Breed a child
                child_populations[unit_id].children.append(breed(parent_a, parent_b, unit_id))
        
        # Save child selections to the child population class members
        self.pacman_cont_children = child_populations[self.PACMAN_ID]
        self.ghost_cont_children = child_populations[self.GHOST_ID]


    def mutate(self):
        """TODO: expand for ghosts
        
        Probabilistically performs sub-tree mutation on each child in the child population."""

        def nullify(state_evaluator, node):
            """Recursively sets this node and its branch to the None node."""
            state_evaluator[node.index].value = None

            if not state_evaluator.is_leaf(node):
                nullify(state_evaluator, state_evaluator.get_left_child(node))
                nullify(state_evaluator, state_evaluator.get_right_child(node))


        for child in self.children:
            for pacman_cont in child.pacman_conts:
                if random.random() < float(self.config.settings['mutation rate']):
                    # Choose mutation node
                    mutation_node = pacman_cont.state_evaluator[random.choices([n for n in pacman_cont.state_evaluator if n.value])[0].index]

                    # Remove traces of previous subtree
                    nullify(pacman_cont.state_evaluator, mutation_node)

                    # Grow a new subtree
                    pacman_cont.grow(mutation_node)
        

    def select_for_survival(self):
        """TODO: expand for ghosts
        
        Survivors are selected based on the following configurable methods:
            1. k-tournament selection without replacement
            2. Truncation

        Survivors are stored in self.population.
        """
        if self.config.settings.getboolean('comma survival strategy'):
            # Use the comma survival strategy
            selection_pool = self.children
        
        else:
            # Default to plus survival strategy
            selection_pool = self.population + self.children

        self.population = []

        if self.config.settings.getboolean('use k tournament survival selection'):
            # Use k-tournament for survival selection without replacement
            while len(self.population) <= self.population_size:
                self.population.append(self.perform_tournament_selection(selection_pool, int(self.config.settings['k survival selection']), w_replacement=False))

            # Maintain the population size
            # This accounts for situations where the population size is not divisible by k
            self.population = self.population[:self.population_size]
        
        else:
            # Default to truncation survival selection
            self.sort_individuals(selection_pool)
            self.population = selection_pool[:self.population_size]
        

    def update_game_state(self, individual):
        """Updates the state of the game *before* all characters have moved."""
        if len(individual.world.fruit_coord):
            fruit_coord = individual.world.fruit_coord

        else:
            fruit_coord = None

        individual.game_state.update(individual.world.pacman_coords, individual.world.ghost_coords, individual.world.pill_coords, [self.get_num_adj_walls(individual.world, pacman_coord) for pacman_coord in individual.world.pacman_coords], fruit_coord)


    def move_units(self, individual):
        """Moves all units in individual.world based on the unit controller moves.
        
        Before units are moved, a fruit probabilistically spawns and the game state
        is updated.

        After units are moved, game variables are updated.
        """
        individual.world.randomly_spawn_fruit()

        self.update_game_state(individual)

        if self.use_single_pacman_cont:
            for pacman_index in range(self.num_pacmen):
                individual.world.move_pacman(individual.pacman_conts[0].get_move(individual.game_state, pacman_index), pacman_index)
        
        else:
            for pacman_index, pacman_cont in enumerate(individual.pacman_conts):
                individual.world.move_pacman(pacman_cont.get_move(individual.game_state, pacman_index), pacman_index)

        if self.use_single_ghost_cont:
            for ghost_index in range(self.num_ghosts):
                individual.world.move_ghost(individual.ghost_conts[0].get_move(individual.game_state, ghost_index), ghost_index)

        else:
            for ghost_index, ghost_cont in enumerate(individual.ghost_conts):
                individual.world.move_ghost(ghost_cont.get_move(individual.game_state, ghost_index), ghost_index)

        # Update time remaining
        individual.world.time_remaining -= 1

        for pacman_coord in individual.world.pacman_coords:
            # Update pills
            if pacman_coord in individual.world.pill_coords:
                individual.world.pill_coords.remove(pacman_coord)
                individual.world.num_pills_consumed += 1

            # Update fruit
            if pacman_coord in individual.world.fruit_coord:
                individual.world.fruit_coord.remove(pacman_coord)
                individual.world.num_fruit_consumed += 1

        # Update score
        individual.world.update_score()

        # Update the world state
        individual.world.world_file.save_snapshot(individual.world.pacman_coords,
            individual.world.ghost_coords, individual.world.fruit_coord, 
            individual.world.time_remaining, individual.world.score)


    def decide_termination(self):
        """Returns False if the program will terminate, True otherwise.

        The program will terminate if any of the following conditions are True:
            1. The number of evaluations specified in config has been reached.
            2. There has been no change in fitness for n generations.
        """
        if self.eval_count >= int(self.config.settings['num fitness evals']):
            # The number of desired evaluations has been reached
            return False
        
        if self.stale_score_count_termination >= int(self.config.settings['n convergence criterion']):
            # The population has stagnated
            return False

        return True


    def check_game_over(self, individual):
        """Returns False if the game is over for the given individual (allowing for a loop to terminate), 
        and True otherwise.

        The conditions for game over are seen in check_game_over() in the GPacWorld class.
        """
        if individual.world.check_game_over():
            return False

        return True


    def check_update_log_world_files(self):
        """Writes a new log file entry and writes a transcript of this run to the 
        world file iff it had the global best score.
        """
        fitness_list = [individual.fitness for individual in self.population]
        self.avg_score = sum(fitness_list) / self.population_size
        local_best_fitness_candidate = max(fitness_list)

        # Determine if a new local best score (fitness) has been found
        if local_best_fitness_candidate > self.local_best_score:
            self.local_best_score = local_best_fitness_candidate

            for individual in self.population:
                if individual.fitness == self.local_best_score:
                    local_best_individual = individual
                    break

        # Write log file row
        self.log.write_run_data(self.eval_count, self.avg_score, self.local_best_score)
        
        # Determine if a new global best score has been found
        if self.local_best_score > self.global_best_score:
            self.global_best_score = self.local_best_score

            # Write to world file
            individual.world.world_file.write_to_file()

            # Write to solution file
            self.soln.write_to_file(individual)

        # Determine if the population fitness is stagnating
        if math.isclose(self.avg_score, self.prev_avg_score, rel_tol=float(self.config.settings['termination convergence criterion magnitude'])):
            self.stale_score_count_termination += 1

        else:
            self.stale_fitness_count_termination = 0
            self.prev_avg_score = self.avg_score


    def get_num_adj_walls(self, world, coord):
        """Returns the number of walls adjacent to coord in the given world."""
        return len([c for c in world.get_adj_coords(coord) if c in world.wall_coords])


    def sort_individuals(self, individuals):
        """Sorts the given individuals in-place by fitness, from best to worst."""
        individuals.sort(key=lambda x : x.fitness, reverse=True)


    def perform_tournament_selection(self, individuals, k, w_replacement=False):
        """Performs a k-tournament selection on individuals, a list of GPacWorldIndividual 
        objects. 
        
        Returns the winning individual (the individual with the best fitness).
        """
        arena_individuals = []
        forbidden_indices = set([])

        # Randomly select k elements from individuals
        for _ in range(k):
            rand_index = random.randint(0, len(individuals) - 1)

            # Don't allow replacement (if applicable)
            while rand_index in forbidden_indices:
                rand_index = random.randint(0, len(individuals) - 1)

            arena_individuals.append(individuals[rand_index])

            if w_replacement == False:
                forbidden_indices.add(rand_index)

        # Make the individuals fight, return the winner
        return max(arena_individuals, key=lambda x : x.fitness)
