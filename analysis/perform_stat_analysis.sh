#!/bin/bash

bold=$(tput bold)
normal=$(tput sgr0)

# The depreciated statistical analysis method uses the following commands:
./gen_stat_input_files.py
./gen_stats.py > stats.txt
./gen_tables.py
echo "Statistical analysis (in LaTeX table form): ${bold}stats_tables.txt${normal}"