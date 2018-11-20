#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

log_file_paths = \
    [
        '../output/small_log.txt',
        '../output/large_log.txt',
        '../output/BONUS_small_log.txt',
        '../output/BONUS_small_multi_controller_log.txt'
    ]

for log_file_index in range(len(log_file_paths)):
    with open(log_file_paths[log_file_index], 'r') as log_file:
        # Create a list of lines from the log file, disregarding all config parmeters and empty lines
        log_file = log_file.read().split('\n')
        log_file = [line for line in log_file[log_file.index('Run 1'):] if not line == '']

        # key: evaluation number, value: [average fitness, local best fitness]
        eval_dict = {}

        curr_run = 1

        # Scrape data from the log file
        for line in log_file:
            if not line[0] == 'R':
                # This line has eval and fitness data
                eval_num, avg_fit, best_fit = line.split('\t')

                eval_num = int(eval_num)
                avg_fit = float(avg_fit)
                best_fit = float(best_fit)

                if eval_num in eval_dict:
                    eval_dict[eval_num].append(best_fit)

                else:
                    eval_dict[eval_num] = [best_fit]

        evals = []
        data = []
        for eval_num in sorted(eval_dict.keys()):
            evals.append(eval_num)
            data.append(eval_dict[eval_num])
        
        # Plot the results
        fig, ax = plt.subplots()

        ax.boxplot(data)

        plt.xticks(np.arange(len(evals) + 2), tuple([0] + evals[2::4]))

        ax.set_xticks(ax.get_xticks()[::4])
    
        # Include labels
        plt.xlabel('Evaluations')
        plt.ylabel('Fitness')

        # Save and close the plot
        plt.savefig(log_file_paths[log_file_index][:log_file_paths[log_file_index].find('_log.txt')] + '_graph.png')
        plt.close()
