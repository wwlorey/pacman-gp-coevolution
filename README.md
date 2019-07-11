#### CS 5401 Evolutionary Computing | Coevolving controllers for pacman and the ghosts using genetic programming (GP)

#################################
# Notes on the BONUS Sections #
#################################

For this assignment, the following bonus assignments were completed:

* BONUS1: Single pacman, multiple ghosts employing different controllers
* BONUS2a: Multiple pacmen employing the same controller, multiple ghosts employing the same controller
* BONUS2b: Multiple pacmen employing different controllers, multiple ghosts employing the same controller
* BONUS2c: Multiple pacmen employing the same controller, multiple ghosts employing different controllers
* BONUS2d: Multiple pacmen employing different controllers, multiple ghosts employing different controllers

To satisfy the above bonus assignments, configuration parameters
were added to programmatically switch between pacman/ghost controller
multiplicity and the number of pacmen used in each world. Configuration files,
output files, and figures each named in accordance to the BONUS* labels above
have been included to satisfy the bonus assignment requirements.

Additionally, code that differs as part of the main assignment implementation is noted 
in the source code using comments (e.g. `# BONUS1`, `# BONUS2, or '# BONUS1, BONUS2` depending
on what bonus the code satisfies).

#### Provided README:

#################################
#	Coding Standards	#
#################################

You are free to use any of the following programming languages for your submission : 
	- C++
	- C#
	- Java
	- Python
Your code must be well formatted and styled according to good coding standards, such as the MST coding standard outlined here : 
http://web.mst.edu/~cpp/cpp_coding_standard_v1_1.pdf
It is required that your code is well documented.

NOTE : Sloppy, undocumented, or otherwise unreadable code will be penalized for not following good coding standards (as laid out in the grading rubric on the course website) : 
http://web.mst.edu/~tauritzd/courses/ec/cs5401fs2018/

#################################
#	Submission Rules	#
#################################

Included in your repository is a script named ”finalize.sh”, which you will use to indicate which version of your code is the one to be graded. When you are ready to submit your final version, run the command ./finalize.sh from your local Git directory, then commit and push your code. This will create a text file, ”readyToSubmit.txt,” that is pre-populated with a known text message for grading purposes. You may commit and push as much as you want, but your submission will be confirmed as ”final” if ”readyToSubmit.txt” exists and is populated with the text generated by ”finalize.sh” at 11:59pm on the due date. If you do not plan to submit before the deadline, then you should NOT run the ”finalize.sh” script until your final submission is ready. If you accidentally run ”finalize.sh” before you are ready to submit, do not commit or push your repo and delete ”readyToSubmit.txt.” Once your final submission is ready, run ”finalize.sh”, commit and push your code, and do not make any further changes to it

Late submissions will be penalized 5% for the first 24 hour period and an additional 10% for every 24 hour period thereafter.

#################################
#       Compiling & Running	#
#################################

Your final submission must include the script "run.sh" which should compile and run your code.
Your run.sh script should generate solutions for each puzzle instance associated with that assignment.

Your script should run on the campus linux machines with the following command : 
	./run.sh
