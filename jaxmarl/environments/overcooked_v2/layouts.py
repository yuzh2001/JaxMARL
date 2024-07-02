from jaxmarl.environments.overcooked_v2.common import StaticObject
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools

cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""

cramped_room_v2 = """
WWPWW
0A A1
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

asymm_advantages_recipes_center = """
WWWWWWWWW
0 WXR01 X
1   P   W
W A PA  W
WWWBWBWWW
"""

asymm_advantages_recipes_right = """
WWWWWWWWW
0 WXW01 X
1   P   R
W A PA  W
WWWBWBWWW
"""

asymm_advantages_recipes_left = """
WWWWWWWWW
0 WXW01 X
1   P   W
R A PA  W
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
W01BWB10W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""

two_rooms_simple = """
WWWWWB10W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""

long_room = """
WWWWWWWWWWWWWWW
B            AP
0             X
WWWWWWWWWWWWWWW
"""

fun_coordination = """
WWWWWWWWW
0   X   2
RA  P  AW
1   B   3
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

    # If recipe is none, recipes will be sampled from the possible_recipes
    # If possible_recipes is none, all possible recipes with the available ingredients will be considered
    possible_recipes: Optional[List[List[int]]]

    @staticmethod
    def _get_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """
        Get all possible recipes given the number of ingredients.
        """
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )

        return [list(recipe) for recipe in unique_recipes]

    def get_possible_recipes(self):
        if self.recipe is not None:
            possible_recipes = [self.recipe]
        elif self.possible_recipes is not None:
            possible_recipes = self.possible_recipes
        else:
            possible_recipes = self._get_possible_recipes(self.num_ingredients)
        return possible_recipes


def layout_grid_to_dict(grid, recipe=None, possible_recipes=None):
    """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
    W: wall
    A: agent
    X: goal
    B: plate (bowl) pile
    P: pot location
    R: recipe of the day indicator
    0-9: Ingredient x pile
    ' ' (space) : empty cell

    Depricated:
    O: onion pile - will be interpreted as ingredient 0


    If `recipe` is provided, it should be a list of ingredient indices, max 3 ingredients per recipe
    If `recipe` is not provided, the recipe will be randomized on reset.
    If the layout does not have a recipe indicator, a fixed `recipe` must be provided.

    If `possible_recipes` is provided, it should be a list of lists of ingredient indices, 3 ingredients per recipe.
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
        char_to_static_item[f"{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

    agent_positions = []

    num_ingredients = 0
    includes_recipe_indicator = False
    for r, row in enumerate(rows):
        c = 0
        while c < len(row):
            char = row[c]

            if char == "O":
                char = "0"

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

            c += 1

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
        static_objects=static_objects,
        width=len(rows[0]),
        height=len(rows),
        recipe=recipe,
        num_ingredients=num_ingredients,
        possible_recipes=possible_recipes,
    )

    return layout


overcooked_v2_layouts = {
    "cramped_room": layout_grid_to_dict(cramped_room, recipe=[0, 0, 0]),
    "cramped_room_v2": layout_grid_to_dict(cramped_room_v2),
    "asymm_advantages": layout_grid_to_dict(asymm_advantages, recipe=[0, 0, 0]),
    "asymm_advantages_recipes_center": layout_grid_to_dict(
        asymm_advantages_recipes_center
    ),
    "asymm_advantages_recipes_right": layout_grid_to_dict(
        asymm_advantages_recipes_right
    ),
    "asymm_advantages_recipes_left": layout_grid_to_dict(asymm_advantages_recipes_left),
    "coord_ring": layout_grid_to_dict(coord_ring, recipe=[0, 0, 0]),
    "forced_coord": layout_grid_to_dict(forced_coord, recipe=[0, 0, 0]),
    "counter_circuit": layout_grid_to_dict(counter_circuit_grid, recipe=[0, 0, 0]),
    "two_rooms": layout_grid_to_dict(two_rooms),
    "two_rooms_simple": layout_grid_to_dict(two_rooms_simple),
    "long_room": layout_grid_to_dict(long_room, recipe=[0, 0, 0]),
    "fun_coordination": layout_grid_to_dict(
        fun_coordination, possible_recipes=[[2, 2, 0], [3, 3, 1]]
    ),
}
