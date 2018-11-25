from enum import Enum


class TerminalNodes(Enum):
    PACMAN_GHOST_DIST   = 0
    PACMAN_PILL_DIST    = 1
    PACMAN_FRUIT_DIST   = 2
    NUM_ADJ_WALLS       = 3
    FP_CONSTANT         = 4
    NEAREST_PACMAN_DIST = 5


class FunctionNodes(Enum):
    ADD               = 0
    SUBTRACT          = 1
    MULTIPLY          = 2
    DIVIDE            = 3
    RANDOM_FLOAT      = 4

