from enum import Enum


class TerminalNodes(Enum):
    NEAREST_PACMAN_DIST = 0
    NEAREST_GHOST_DIST  = 1
    FP_CONSTANT         = 2
