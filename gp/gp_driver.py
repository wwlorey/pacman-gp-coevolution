import controllers.game_state as game_state_class
import controllers.ghosts_controller as ghosts_cont_class
import controllers.pacman_controller as pacman_cont_class
import controllers.tree as tree
import copy
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

        self.population_size = int(self.config.settings['mu'])
        self.child_population_size = int(self.config.settings['lambda'])
        self.parent_population_size = int(self.config.settings['num parents'])

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

        # Initialize the population
        self.population = []
        for _ in range(self.population_size):
            world = gpac_world_class.GPacWorld(self.config)
            game_state = game_state_class.GameState(world.pacman_coords, world.ghost_coords, world.pill_coords, self.get_num_adj_walls(world, world.pacman_coords[0]))
            
            # BONUS2
            pacman_conts = [pacman_cont_class.PacmanController(self.config) for _ in range(int(self.config.settings['num pacmen']))]
            ghosts_cont = ghosts_cont_class.GhostsController(self.config)
            game_state.update_walls(world.wall_coords)

            # BONUS2
            self.population.append(gpac_world_individual_class.GPacWorldIndividual(world, game_state, pacman_conts, ghosts_cont))


    def end_run(self):
        """Increments the run count by one.
        
        This should be called after each run.
        """
        self.run_count += 1


    def end_eval(self, individual):
        """Conditionally updates the log and world files and increments 
        the evaluation count.

        This should be called after each evaluation.
        """
        individual.fitness = individual.world.score
        self.eval_count += 1


    def control_bloat(self):
        """Adjusts the fitness of each individual in the population by applying 
        parsimony pressure.
        """
        for individual in self.population:
            # BONUS2
            avg_num_nodes = 0

            for pacman_cont in individual.pacman_conts: 
                avg_num_nodes += int(sum([pacman_cont.get_num_nodes() for individual in self.population]) / self.population_size)

            num_nodes = 0

            for pacman_cont in individual.pacman_conts:
                num_nodes += pacman_cont.get_num_nodes()

            if  num_nodes > avg_num_nodes:
                individual.fitness /= int(float(self.config.settings['p parsimony coefficient']) * (num_nodes - avg_num_nodes))
                individual.fitness = int(individual.fitness)


    def evaluate(self, population):
        """Evaluates all population members given in population by running
        each world's game until completion.
        """
        for individual in population:
            while self.check_game_over(individual):
                self.move_units(individual)

            self.end_eval(individual)

        self.check_update_log_world_files()


    def select_parents(self):
        """Chooses which parents from the population will breed.

        Depending on the parent selection configuration, one of the three following 
        methods is used to select parents:
            1. Fitness proportional selection
            2. Over-selection

        The resulting parents are stored in self.parents.
        """
        self.parents = []

        if self.config.settings.getboolean('use fitness proportional parent selection'):
            # Select parents for breeding using the fitness proportional "roulette wheel" method (with replacement)
            parent_fitnesses = [individual.fitness for individual in self.population]

            if max(parent_fitnesses) == min(parent_fitnesses):
                # All parent fitnesses are the same so select parents at random
                for _ in range(self.parent_population_size):
                    self.parents.append(self.population[random.randrange(0, len(self.population))])

            else:
                self.parents = random.choices(self.population, weights=parent_fitnesses, k=self.parent_population_size)

        else:
            # Default to over-selection parent selection
            elite_index_cuttoff = int(float(self.config.settings['x overselection']) * len(self.population))

            # Note: the following hardcoded percentages are specified as part of the overselection algorithm
            num_elite_parents = int(0.80 * self.parent_population_size)
            num_sub_elite_parents = int(0.20 * self.parent_population_size)

            self.sort_individuals(self.population)
            elite_group = self.population[:elite_index_cuttoff]
            sub_elite_group = self.population[elite_index_cuttoff:]

            elite_choices = [individual.fitness for individual in elite_group]

            if max(elite_choices) == min(elite_choices):
                # All elite individual fitnesses are the same so select individuals at random
                for _ in range(num_elite_parents):
                    self.parents.append(elite_group.pop())
            
            else:
                self.parents = random.choices(elite_group, weights=elite_choices, k=num_elite_parents)

            sub_elite_choices = [individual.fitness for individual in sub_elite_group]

            if max(sub_elite_choices) == min(sub_elite_choices):
                # All sub-elite individual fitnesses are the same so select individuals at random
                for _ in range(num_sub_elite_parents):
                    self.parents.append(sub_elite_group.pop())
            
            else:
                self.parents += random.choices(sub_elite_group, weights=sub_elite_choices, k=num_sub_elite_parents)
                 

    def recombine(self):
        """Breeds lambda (offspring pool size) children using sub-tree crossover 
        from the existing parent population. The resulting children are stored in 
        self.children.
        """

        def breed(parent_a, parent_b):
            """Performs sub-tree crossover on parent_a and parent_b returning the child tree."""

            def crossover_recursive(receiver_index, donator_index):
                # BONUS2
                if receiver_index < len(child_pacman_conts[cont_index].state_evaluator) and donator_index < len(parent_pacman_cont.state_evaluator):
                    return
                
                child_pacman_conts[cont_index].state_evaluator[receiver_index] = tree.TreeNode(receiver_index, parent_pacman_cont.state_evaluator[donator_index].value)
                crossover_recursive(child_pacman_conts[cont_index].state_evaluator.get_left_child_index(receiver_index), parent_pacman_cont.state_evaluator.get_left_child_index(donator_index))
                crossover_recursive(child_pacman_conts[cont_index].state_evaluator.get_right_child_index(receiver_index), parent_pacman_cont.state_evaluator.get_right_child_index(donator_index))


            # BONUS2
            child_pacman_conts = [None] * int(self.config.settings['num pacmen'])
            for cont_index in range(int(self.config.settings['num pacmen'])):
                # Choose a random node (crossover point) from each state evaluator node list
                crossover_node_a = parent_a.pacman_conts[cont_index].state_evaluator[random.choices([n for n in parent_a.pacman_conts[cont_index].state_evaluator if n.value])[0].index]
                crossover_node_b = parent_b.pacman_conts[cont_index].state_evaluator[random.choices([n for n in parent_b.pacman_conts[cont_index].state_evaluator if n.value])[0].index]

                child_pacman_conts[cont_index] = copy.copy(parent_a.pacman_conts[cont_index])
                parent_pacman_cont = parent_b.pacman_conts[cont_index]

                # Extend the child's state evaluator if necessary
                if len(child_pacman_conts[cont_index].state_evaluator) < len(parent_a.pacman_conts[cont_index].state_evaluator):
                    child_pacman_conts[cont_index].state_evaluator = child_pacman_cont.state_evaluator + [tree.TreeNode(index, None) for index in range(len(child_pacman_cont.state_evaluator), len(parent_a.pacman_cont.state_evaluator))]

                # Perform sub-tree crossover
                crossover_recursive(crossover_node_a.index, crossover_node_b.index)

            # Finish generating the child
            world = gpac_world_class.GPacWorld(self.config)
            game_state = game_state_class.GameState(world.pacman_coords, world.ghost_coords, world.pill_coords, self.get_num_adj_walls(world, world.pacman_coords[0]))
            
            # BONUS2
            pacman_conts = child_pacman_conts
            
            ghosts_cont = parent_a.ghosts_cont
            game_state.update_walls(world.wall_coords)

            # BONUS2
            child = gpac_world_individual_class.GPacWorldIndividual(world, game_state, pacman_conts, ghosts_cont)
            return child


        self.children = []

        for _ in range(self.child_population_size):
            # Select parents with replacement
            # Note: this implementation allows for parent_a and parent_b to be the same genotype
            parent_a = self.parents[random.randrange(0, len(self.parents))]
            parent_b = self.parents[random.randrange(0, len(self.parents))]

            # Breed a child
            self.children.append(breed(parent_a, parent_b))
        

    def mutate(self):
        """Probabilistically performs sub-tree mutation on each child in the child population."""

        def nullify(state_evaluator, node):
            """Recursively sets this node and its branch to the None node."""
            state_evaluator[node.index].value = None

            if not state_evaluator.is_leaf(node):
                nullify(state_evaluator, state_evaluator.get_left_child(node))
                nullify(state_evaluator, state_evaluator.get_right_child(node))


        for child in self.children:
            # BONUS2
            for pacman_cont in child.pacman_conts:
                if random.random() < float(self.config.settings['mutation rate']):
                    # Choose mutation node
                    mutation_node = pacman_cont.state_evaluator[random.choices([n for n in pacman_cont.state_evaluator if n.value])[0].index]

                    # Remove traces of previous subtree
                    nullify(pacman_cont.state_evaluator, mutation_node)

                    # Grow a new subtree
                    pacman_cont.grow(mutation_node)
        

    def select_for_survival(self):
        """Survivors are selected based on the following configurable methods:
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

        individual.game_state.update(individual.world.pacman_coords, individual.world.ghost_coords, individual.world.pill_coords, self.get_num_adj_walls(individual.world, individual.world.pacman_coords[0]), fruit_coord)


    def move_units(self, individual):
        """Moves all units in individual.world based on the unit controller moves.
        
        Before units are moved, a fruit probabilistically spawns and the game state
        is updated.

        After units are moved, game variables are updated.
        """
        individual.world.randomly_spawn_fruit()

        self.update_game_state(individual)

        # BONUS2
        for pacman_index, pacman_cont in enumerate(individual.pacman_conts):
            individual.world.move_pacmen(pacman_cont.get_move(individual.game_state, pacman_index), pacman_index)

        for ghost_id in range(len(individual.world.ghost_coords)):
            individual.world.move_ghost(ghost_id, individual.ghosts_cont.get_move(ghost_id, individual.game_state))

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
