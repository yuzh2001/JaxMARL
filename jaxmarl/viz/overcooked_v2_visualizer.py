import math

import numpy as np

from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering as rendering
import jax

# from jaxmarl.environments.overcooked_v2.common import OBJECT_TO_INDEX, COLOR_TO_INDEX, COLORS
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject
from jaxmarl.environments.overcooked_v2.overcooked import POT_COOK_TIME

# INDEX_TO_COLOR = [k for k,v in COLOR_TO_INDEX.items()]
TILE_PIXELS = 32

COLOR_TO_AGENT_INDEX = {0: 0, 2: 1}  # Hardcoded. Red is first, blue is second

COLORS = {
    "red": np.array([255, 0, 0]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([112, 39, 195]),
    "yellow": np.array([255, 255, 0]),
    "grey": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "black": np.array([25, 25, 25]),
    "orange": np.array([230, 180, 0]),
    "pink": np.array([255, 105, 180]),
    "brown": np.array([139, 69, 19]),
    "cyan": np.array([0, 255, 255]),
    "light_blue": np.array([173, 216, 230]),
}

INGREDIENT_COLORS = [
    COLORS["yellow"],
    COLORS["purple"],
    COLORS["blue"],
    COLORS["orange"],
    COLORS["red"],
    COLORS["pink"],
    COLORS["brown"],
    COLORS["cyan"],
    COLORS["light_blue"],
]


class OvercookedV2Visualizer:
    """
    Manages a window and renders contents of EnvState instances to it.
    """

    tile_cache = {}

    def __init__(self):
        self.window = None

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window("Overcooked V2")

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        self._lazy_init_window()

        img = self._render_state(agent_view_size, state, highlight, tile_size)

        self.window.show_img(img)

    def animate(self, state_seq, agent_view_size, filename="animation.gif"):
        """Animate a gif give a state sequence and save if to file."""
        import imageio

        def get_frame(state):
            frame = OvercookedV2Visualizer._render_state(
                agent_view_size, state, highlight=False
            )
            return frame

        frame_seq = [get_frame(state) for state in state_seq]

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    @classmethod
    def _render_state(
        cls, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS
    ):
        """
        Render the state
        """

        grid = state.grid
        agents = state.agents
        recipe = state.recipe

        def _include_agents(grid, agent):
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir
            return (
                grid.at[pos.y, pos.x].set([StaticObject.AGENT, inventory, direction]),
                None,
            )

        grid, _ = jax.lax.scan(_include_agents, grid, agents)

        indicator_locations = np.argwhere(
            grid[:, :, 0] == StaticObject.RECIPE_INDICATOR
        )
        grid = grid.at[indicator_locations[:, 0], indicator_locations[:, 1], 1].set(
            recipe | DynamicObject.COOKED | DynamicObject.PLATE
        )

        # Render the whole grid
        img = OvercookedV2Visualizer._render_grid(
            grid,
            tile_size,
        )
        return img

    @classmethod
    def _render_cell(cls, cell, img):
        static_object = cell[0]
        ingredients = cell[1]
        extra_info = cell[2]

        match static_object:
            case StaticObject.WALL:
                rendering.fill_coords(
                    img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
                )

                OvercookedV2Visualizer._render_counter(ingredients, img)

            case StaticObject.RECIPE_INDICATOR:
                rendering.fill_coords(
                    img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
                )
                rendering.fill_coords(
                    img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
                )

                OvercookedV2Visualizer._render_counter(ingredients, img)

            case StaticObject.GOAL:
                rendering.fill_coords(
                    img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
                )
                rendering.fill_coords(
                    img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["green"]
                )

            case StaticObject.EMPTY:
                pass

            case StaticObject.AGENT:
                tri_fn = rendering.point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                )
                # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
                direction_reording = [3, 1, 0, 2]
                direction = direction_reording[extra_info]
                tri_fn = rendering.rotate_fn(
                    tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
                )
                rendering.fill_coords(img, tri_fn, COLORS["red"])

                OvercookedV2Visualizer._render_inv(ingredients, img)

            case StaticObject.POT:
                OvercookedV2Visualizer._render_pot(cell, img)

            case StaticObject.PLATE_PILE:
                rendering.fill_coords(
                    img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
                )
                plate_fns = [
                    rendering.point_in_circle(*coord, 0.2)
                    for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]
                ]
                [
                    rendering.fill_coords(img, plate_fn, COLORS["white"])
                    for plate_fn in plate_fns
                ]

            case (
                ingredient_pile
            ) if ingredient_pile >= StaticObject.INGREDIENT_PILE_BASE:
                ingredient_idx = ingredient_pile - StaticObject.INGREDIENT_PILE_BASE

                rendering.fill_coords(
                    img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
                )
                ingredient_fns = [
                    rendering.point_in_circle(*coord, 0.15)
                    for coord in [
                        (0.5, 0.15),
                        (0.3, 0.4),
                        (0.8, 0.35),
                        (0.4, 0.8),
                        (0.75, 0.75),
                    ]
                ]
                [
                    rendering.fill_coords(
                        img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx]
                    )
                    for ingredient_fn in ingredient_fns
                ]

            case _:
                raise ValueError(
                    f"Rendering object at index {static_object} is currently unsupported."
                )

    @classmethod
    def _render_counter(cls, ingredients, img):
        if ingredients & DynamicObject.PLATE:
            plate_fn = rendering.point_in_circle(0.5, 0.5, 0.3)
            rendering.fill_coords(img, plate_fn, COLORS["white"])

        if DynamicObject.is_ingredient(ingredients):
            idx = DynamicObject.get_ingredient_idx(ingredients)
            ingredient_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
            rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

        if ingredients & DynamicObject.COOKED:
            positions = [(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]
            ingredient_indices = DynamicObject.get_ingredient_idx_list(ingredients)

            for idx, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[ingredient_idx]
                pos = positions[min(idx, len(positions) - 1)]
                ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
                rendering.fill_coords(img, ingredient_fn, color)

    @classmethod
    def _render_pot(cls, cell, img):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > 0
        is_cooked = (ingredients & DynamicObject.COOKED) != 0
        is_idle = not is_cooking and not is_cooked
        ingredients = DynamicObject.get_ingredient_idx_list(ingredients)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

        if len(ingredients) > 0:
            ingredient_fns = [
                rendering.point_in_circle(*coord, 0.13)
                for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
            ]
            [
                rendering.fill_coords(
                    img, ingredient_fns[i], INGREDIENT_COLORS[ingredient_idx]
                )
                for i, ingredient_idx in enumerate(ingredients)
            ]

        if len(ingredients) > 0 and is_idle:
            lid_fn = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
            handle_fn = rendering.rotate_fn(
                handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi
            )

        # if is_cooked:
        # TODO: maybe make it more obvious that the dish is cooked

        # Render the pot itself
        pot_fns = [pot_fn, lid_fn, handle_fn]
        [rendering.fill_coords(img, pot_fn, COLORS["black"]) for pot_fn in pot_fns]

        # Render progress bar
        if is_cooking:
            progress_fn = rendering.point_in_rect(
                0.1, 0.9 - (0.9 - 0.1) / POT_COOK_TIME * time_left, 0.83, 0.88
            )
            rendering.fill_coords(img, progress_fn, COLORS["green"])

    @classmethod
    def _render_inv(cls, ingredients, img):
        print("ingredients: ", ingredients)
        if DynamicObject.is_ingredient(ingredients):
            print("is ingredient")
            idx = DynamicObject.get_ingredient_idx(ingredients)
            ingredient_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
            rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])
            return

        if ingredients & DynamicObject.PLATE:
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])
        if ingredients & DynamicObject.COOKED:
            positions = [(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]
            ingredient_indices = DynamicObject.get_ingredient_idx_list(ingredients)

            for idx, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[ingredient_idx]
                pos = positions[min(idx, len(positions) - 1)]
                ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.10)
                rendering.fill_coords(img, ingredient_fn, color)

    @classmethod
    def _render_tile(
        cls,
        obj,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
    ):
        """
        Render a tile and cache the result
        """
        key = (*obj.tolist(), highlight, tile_size)

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
        )

        # Draw the grid lines (top and left edges)
        rendering.fill_coords(
            img, rendering.point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
        )
        rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
        )

        OvercookedV2Visualizer._render_cell(obj, img)

        if highlight:
            rendering.highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    @classmethod
    def _render_grid(
        cls,
        grid,
        tile_size=TILE_PIXELS,
        highlight_mask=None,
    ):
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=grid.shape[:2], dtype=bool)

        # Compute the total grid size in pixels
        width_px = grid.shape[1] * tile_size
        height_px = grid.shape[0] * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        def _set_tile(x, y, tile_img):
            ymin = y * tile_size
            ymax = (y + 1) * tile_size
            xmin = x * tile_size
            xmax = (x + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

        # Render the grid
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                cell = grid[y, x]
                tile_img = OvercookedV2Visualizer._render_tile(
                    cell,
                    highlight=highlight_mask[y, x],
                    tile_size=tile_size,
                )

                _set_tile(x, y, tile_img)

        return img

    def close(self):
        self.window.close()
