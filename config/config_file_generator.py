#!/usr/bin/env python3

config_template = '[DEFAULT]\n\
\n\
###################################\n\
# General GP Parameters\n\
###################################\n\
num experiment runs = 30\n\
num fitness evals = 2000\n\
\n\
mu = 20\n\
lambda = 60\n\
comma survival strategy = True\n\
# Default to plus survival strategy if comma is not selected\n\
\n\
control bloat = True\n\
p parsimony coefficient = 2\n\
\n\
n convergence criterion = 15\n\
termination convergence criterion magnitude = 0.01\n\
\n\
###################################\n\
# Parent Selection\n\
###################################\n\
num parents = 10\n\
use fitness proportional parent selection = False\n\
# Default to overselection if FPS is not selected\n\
\n\
x overselection = 0.4\n\
\n\
###################################\n\
# Mutation\n\
###################################\n\
mutation rate = 0.1\n\
min tree mutation height = 2\n\
max tree mutation height = 4\n\
\n\
###################################\n\
# Survival Selection\n\
###################################\n\
use k tournament survival selection = True\n\
# Default to truncation if k-tourny is not selected\n\
\n\
k survival selection = 5\n\
\n\
###################################\n\
# GP Tree Generation\n\
###################################\n\
max tree generation height = 6\n\
premature end probability = 0.2\n\
ramped half-and-half probability = 0.5\n\
\n\
###################################\n\
# World Generation\n\
###################################\n\
width = {width}\n\
height = {height}\n\
pill density = {pill density}\n\
wall density = {wall density}\n\
num pacmen = {num pacmen}\n\
num ghosts = 3\n\
\n\
num wall carvers = 2\n\
num respawn wall carvers = 2\n\
max wall carver travel distance = 10\n\
min wall carver travel distance = 3\n\
\n\
use external seed = False\n\
default seed = 123456789\n\
\n\
###################################\n\
# Score \n\
###################################\n\
fruit spawn probability = {fruit spawn probability}\n\
fruit score = {fruit score}\n\
\n\
###################################\n\
# Timing\n\
###################################\n\
time multiplier = {time multiplier}\n\
\n\
###################################\n\
# State Evaluator\n\
###################################\n\
max fp constant = 1000\n\
\n\
###################################\n\
# Output Files\n\
###################################\n\
log file path = output/{experiment name}_log.txt\n\
soln file path = output/{experiment name}_soln.txt\n\
world file path = output/highest_score_game_sequence_all_time_step_world_file_{experiment name}.txt\n'

config_data = \
[
    (10, 15, 50, 25, 0.05, 10, 2, 'small', 1), 
    (30, 20, 25, 30, 0.05, 10, 2, 'large', 1),
    (10, 15, 50, 25, 0.05, 10, 2, 'BONUS_small', 2), 
    (30, 20, 25, 30, 0.05, 10, 2, 'BONUS_large', 2),
    (10, 15, 50, 25, 0.05, 10, 2, 'BONUS_small_multi_controller', 2), 
    (30, 20, 25, 30, 0.05, 10, 2, 'BONUS_large_multi_controller', 2)
]

FILE_NAME_INDEX = 7

config_data_key = ['width', 'height', 'pill density', 'wall density', 'fruit spawn probability', 'fruit score', 'time multiplier', 'experiment name', 'num pacmen']


# Populate new config files
for tup in config_data:
    config_str = config_template

    for i in range(len(tup)):
        search_str = '{' + config_data_key[i] + '}'

        insertion_index = config_str.find(search_str)

        while insertion_index >= 0:
            config_str = config_str[:insertion_index] + str(tup[i]) + config_str[insertion_index + len(search_str):]

            insertion_index = config_str.find(search_str)

    # Create config file
    file_name = tup[FILE_NAME_INDEX] + '.cfg'

    f = open(file_name, 'w')

    f.write(config_str)
    f.close()
