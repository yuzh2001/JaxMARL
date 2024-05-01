import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from .common import Direction, Position, StaticItem
from flax import struct
from typing import NameTuple

cramped_room = {
    "height": 4,
    "width": 5,
    "wall_idx": jnp.array([0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 16, 17, 18, 19]),
    "agent_idx": jnp.array([6, 8]),
    "goal_idx": jnp.array([18]),
    "plate_pile_idx": jnp.array([16]),
    "onion_pile_idx": jnp.array([5, 9]),
    "pot_idx": jnp.array([2]),
}
asymm_advantages = {
    "height": 5,
    "width": 9,
    "wall_idx": jnp.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            22,
            26,
            27,
            31,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
        ]
    ),
    "agent_idx": jnp.array([29, 32]),
    "goal_idx": jnp.array([12, 17]),
    "plate_pile_idx": jnp.array([39, 41]),
    "onion_pile_idx": jnp.array([9, 14]),
    "pot_idx": jnp.array([22, 31]),
}
coord_ring = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array(
        [0, 1, 2, 3, 4, 5, 9, 10, 12, 14, 15, 19, 20, 21, 22, 23, 24]
    ),
    "agent_idx": jnp.array([7, 11]),
    "goal_idx": jnp.array([22]),
    "plate_pile_idx": jnp.array([10]),
    "onion_pile_idx": jnp.array([15, 21]),
    "pot_idx": jnp.array([3, 9]),
}
forced_coord = {
    "height": 5,
    "width": 5,
    "wall_idx": jnp.array(
        [0, 1, 2, 3, 4, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 21, 22, 23, 24]
    ),
    "agent_idx": jnp.array([11, 8]),
    "goal_idx": jnp.array([23]),
    "onion_pile_idx": jnp.array([5, 10]),
    "plate_pile_idx": jnp.array([15]),
    "pot_idx": jnp.array([3, 9]),
}

# Example of layout provided as a grid
counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""


@struct.dataclass
class Layout:
    # List of agent positions
    agent_positions: jnp.ndarray

    # width x height grid with static items
    static_objects: jnp.ndarray

    num_ingredients: int


def layout_grid_to_dict(grid):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    P: pot location
    R: recipe of the day indicator
    Ix: Ingredient x pile
    ' ' (space) : empty cell

    Depricated:
    O: onion pile - will be interpreted as I0
    """

    rows = grid.split("\n")

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    static_objects = jnp.zeros((len(rows), len(rows[0])), dtype=jnp.int32)

    char_to_static_item = {
        " ": StaticItem.EMPTY,
        "W": StaticItem.WALL,
        "X": StaticItem.GOAL,
        "B": StaticItem.PLATE_PILE,
        "P": StaticItem.POT,
    }

    for r in range(10):
        char_to_static_item[f"I{r}"] = StaticItem.INGREDIENT_PILE + r

    agent_positions = []

    for r, row in enumerate(rows):
        j = 0
        c = 0
        while j < len(row):
            char = row[j]
            if char == "I" and j + 1 < len(row) and row[j + 1].isdigit():
                # Handle multi-character ingredient identifiers like I0, I1, etc.
                char += row[j + 1]
                j += 1  # Skip the next character as it is part of the current one

            if char == "O":
                char = "I0"

            if char == "A":
                agent_pos = Position(r, j)
                agent_positions.append(agent_pos)

            static_objects[r, c] = char_to_static_item.get(char, StaticItem.EMPTY)
            j += 1
            c += 1

    # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

    layout = Layout(
        agent_positions=jnp.array(agent_positions),
        static_objects=static_objects,
        num_ingredients=10,
    )

    return layout


overcooked_layouts = {
    "cramped_room": FrozenDict(cramped_room),
    "asymm_advantages": FrozenDict(asymm_advantages),
    "coord_ring": FrozenDict(coord_ring),
    "forced_coord": FrozenDict(forced_coord),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid),
}
