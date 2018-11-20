#!/bin/bash

# Generate a PDF document
# Do this twice to ensure valid references
pdflatex *.tex
pdflatex *.tex

