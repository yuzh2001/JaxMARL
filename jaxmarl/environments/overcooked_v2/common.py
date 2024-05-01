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

    # Agents are only included in the observation grid
    AGENT = 2
    SELF_AGENT = 3

    GOAL = 4
    POT = 5
    PLATE_PILE = 6
    INGREDIENT_PILE = 10


class DynamicObject(IntEnum):
    EMPTY = 0
    PLATE = 1 << 0
    COOKED = 1 << 1

    # every ingredient has a unique bit
    INGREDIENT = 1 << 2


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3


DIR_TO_VEC = jnp.array(
    [
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, 0),
    ],
    dtype=jnp.int8,
)


@struct.dataclass
class Position:
    x: jnp.ndarray
    y: jnp.ndarray

    @staticmethod
    def from_tuple(t):
        x, y = t
        return Position(jnp.array([x]), jnp.array([y]))

    def move(self, direction):
        vec = DIR_TO_VEC[direction]
        return Position(self.x + vec[0], self.y + vec[1])

    @staticmethod
    def move_in_bounds(width, height):
        def _move(pos, direction):
            new_pos = pos.move(direction)
            new_pos.x = jnp.clip(new_pos.x, 0, width - 1)
            new_pos.y = jnp.clip(new_pos.y, 0, height - 1)
            return new_pos

        return _move


@struct.dataclass
class Agent:
    pos: Position
    dir: jnp.ndarray
    inventory: jnp.ndarray

    def get_fwd_pos(self):
        return self.pos.move(self.dir)

    @staticmethod
    def from_position(pos):
        return Agent(pos, jnp.array([Direction.UP]), jnp.zeros((1,)))


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
