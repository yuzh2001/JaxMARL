from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked_v2.common import Direction, Position, StaticObject
from flax import struct
import numpy as np
import jax.numpy as jnp

cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""

cramped_room_v2 = """
WWPWW
I0A AI1
W   R
WBWXW
"""

asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

coord_ring = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""

forced_coord = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""

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
    agent_positions: jnp.ndarray

    # width x height grid with static items
    static_objects: np.ndarray
    width: int
    height: int

    recipe: np.ndarray


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

    row_lens = [len(row) for row in rows]
    static_objects = np.zeros((len(rows), max(row_lens)), dtype=np.int32)

    char_to_static_item = {
        " ": StaticObject.EMPTY,
        "W": StaticObject.WALL,
        "X": StaticObject.GOAL,
        "B": StaticObject.PLATE_PILE,
        "P": StaticObject.POT,
        "R": StaticObject.RECIPE_INDICATOR,
    }

    for r in range(10):
        char_to_static_item[f"I{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

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
        agent_positions=jnp.array(agent_positions),
        static_objects=static_objects[:, :max_width],
        width=max_width,
        height=len(rows),
        recipe=np.array([0, 0, 1]),
    )

    return layout


overcooked_layouts = {
    "cramped_room": layout_grid_to_dict(cramped_room),
    "cramped_room_v2": layout_grid_to_dict(cramped_room_v2),
    "asymm_advantages": layout_grid_to_dict(asymm_advantages),
    "coord_ring": layout_grid_to_dict(coord_ring),
    "forced_coord": layout_grid_to_dict(forced_coord),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid),
}
