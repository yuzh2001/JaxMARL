from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked_v2.common import Direction, Position, StaticObject
from flax import struct
import numpy as np

counter_circuit_grid = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""


@struct.dataclass
class Layout:
    # agent positions list of positions num_agents x 2 (x, y)
    agent_positions: np.ndarray

    # width x height grid with static items
    static_objects: np.ndarray
    width: int
    height: int

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

    static_objects = np.zeros((len(rows), len(rows[0])), dtype=np.int32)

    char_to_static_item = {
        " ": StaticObject.EMPTY,
        "W": StaticObject.WALL,
        "X": StaticObject.GOAL,
        "B": StaticObject.PLATE_PILE,
        "P": StaticObject.POT,
    }

    for r in range(10):
        char_to_static_item[f"I{r}"] = StaticObject.INGREDIENT_PILE + r

    agent_positions = []

    max_width = 0
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
                agent_pos = [c, r]
                agent_positions.append(agent_pos)

            static_objects[r, c] = char_to_static_item.get(char, StaticObject.EMPTY)
            j += 1
            c += 1
        max_width = max(max_width, c)

    # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

    layout = Layout(
        agent_positions=np.array(agent_positions),
        static_objects=static_objects[:, :max_width],
        width=max_width,
        height=len(rows),
        num_ingredients=10,
    )

    return layout


overcooked_layouts = {
    # "cramped_room": FrozenDict(cramped_room),
    # "asymm_advantages": FrozenDict(asymm_advantages),
    # "coord_ring": FrozenDict(coord_ring),
    # "forced_coord": FrozenDict(forced_coord),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid),
}
