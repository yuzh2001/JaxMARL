from jaxmarl.environments.overcooked_v2.common import StaticObject
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

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

two_rooms = """
WI0I1BWBI1I0W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""

two_rooms_simple = """
WWWWWBI1I0W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""


@dataclass
class Layout:
    # agent positions list of positions, tuples (x, y)
    agent_positions: List[Tuple[int, int]]

    # width x height grid with static items
    static_objects: np.ndarray
    width: int
    height: int

    # If none recipe will be randomized on reset
    # If present recipe should be a list of ingredient indices, max 3 ingredients per recipe
    recipe: Optional[List[int]]

    num_ingredients: int


def layout_grid_to_dict(grid, recipe=None):
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


    If `recipe` is provided, it should be a list of ingredient indices, max 3 ingredients per recipe
    If `recipe` is not provided, the recipe will be randomized on reset.
    If the layout does not have a recipe indicator, a fixed `recipe` must be provided.
    """

    rows = grid.split("\n")

    if len(rows[0]) == 0:
        rows = rows[1:]
    if len(rows[-1]) == 0:
        rows = rows[:-1]

    row_lens = [len(row) for row in rows]
    static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

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
    num_ingredients = 0
    includes_recipe_indicator = False
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
                agent_pos = (c, r)
                agent_positions.append(agent_pos)

            obj = char_to_static_item.get(char, StaticObject.EMPTY)
            static_objects[r, c] = obj

            if StaticObject.is_ingredient_pile(obj):
                ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                num_ingredients = max(num_ingredients, ingredient_idx + 1)

            if obj == StaticObject.RECIPE_INDICATOR:
                includes_recipe_indicator = True

            j += 1
            c += 1
        max_width = max(max_width, c)

    # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

    if recipe is not None:
        if not (0 < len(recipe) <= 3):
            raise ValueError("Recipe must be a list of length 1, 2, or 3")
        if [i for i in recipe if i < 0 or i >= num_ingredients]:
            raise ValueError("Invalid ingredient index in recipe")
    elif not includes_recipe_indicator:
        raise ValueError(
            "Layout does not include a recipe indicator, a fixed recipe must be provided"
        )

    layout = Layout(
        agent_positions=agent_positions,
        static_objects=static_objects[:, :max_width],
        width=max_width,
        height=len(rows),
        recipe=recipe,
        num_ingredients=num_ingredients,
    )

    return layout


overcooked_v2_layouts = {
    "cramped_room": layout_grid_to_dict(cramped_room, recipe=[0, 0, 0]),
    "cramped_room_v2": layout_grid_to_dict(cramped_room_v2),
    "asymm_advantages": layout_grid_to_dict(asymm_advantages, recipe=[0, 0, 0]),
    "coord_ring": layout_grid_to_dict(coord_ring, recipe=[0, 0, 0]),
    "forced_coord": layout_grid_to_dict(forced_coord, recipe=[0, 0, 0]),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid, recipe=[0, 0, 0]),
    "two_rooms": layout_grid_to_dict(two_rooms),
    "two_rooms_simple": layout_grid_to_dict(two_rooms_simple),
}
