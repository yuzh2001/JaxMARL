import numpy as np
import jax.numpy as jnp
from flax import struct
import chex
import jax

from enum import IntEnum
from typing import NamedTuple


class StaticObject(IntEnum):
    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3
    POT = 4
    PLATE_PILE = 5
    INGREDIENT_PILE = 10


class DynamicObject(IntEnum):
    EMPTY = 0
    PLATE = 1 << 0
    COOKED = 1 << 1

    # every ingredient has a unique bit
    INGREDIENT = 1 << 2


class Direction(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class Position(NamedTuple):
    x: int
    y: int

    def move(self, direction):
        if direction == Direction.NORTH:
            return Position(self.x, self.y - 1)
        if direction == Direction.SOUTH:
            return Position(self.x, self.y + 1)
        if direction == Direction.EAST:
            return Position(self.x + 1, self.y)
        if direction == Direction.WEST:
            return Position(self.x - 1, self.y)
        raise ValueError(f"Invalid direction {direction}")


class Agent(NamedTuple):
    pos: Position
    dir: Direction
    inventory: int

    def get_fwd_pos(self):
        return self.pos.move(self.dir)
    
    @staticmethod
    def from_position(pos):
        return Agent(pos, Direction.NORTH, 0)


# class OvercookedState(struct.PyTreeNode):
#     """
#     Overcooked state representation.
#     """

#     agent_pos = chex.Array(jnp.array([0, 0]), dtype=jnp.int32)
#     agent_dir_idx = chex.Array(jnp.array([0]), dtype=jnp.int32)
#     grid = chex.Array(jnp.array([0]), dtype=jnp.int32)
#     time = chex.Array(jnp.array([0]), dtype=jnp.int32)


# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array(
    [
        # Pointing right (positive X)
        # (1, 0), # right
        # (0, 1), # down
        # (-1, 0), # left
        # (0, -1), # up
        (0, -1),  # NORTH
        (0, 1),  # SOUTH
        (1, 0),  # EAST
        (-1, 0),  # WEST
    ],
    dtype=jnp.int8,
)
