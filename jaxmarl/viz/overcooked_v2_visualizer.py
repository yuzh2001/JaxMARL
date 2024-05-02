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
        return self._render_state(agent_view_size, state, highlight, tile_size)

    def animate(self, state_seq, agent_view_size, filename="animation.gif"):
        """Animate a gif give a state sequence and save if to file."""
        import imageio

        padding = agent_view_size - 2  # show

        def get_frame(state):
            grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
            # Render the state
            frame = OvercookedV2Visualizer._render_grid(
                grid,
                tile_size=TILE_PIXELS,
                highlight_mask=None,
                agent_dir_idx=state.agent_dir_idx,
                agent_inv=state.agent_inv,
            )
            return frame

        frame_seq = [get_frame(state) for state in state_seq]

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    def render_grid(self, grid, tile_size=TILE_PIXELS, k_rot90=0, agent_dir_idx=None):
        self._lazy_init_window()

        img = OvercookedV2Visualizer._render_grid(
            grid,
            tile_size,
            highlight_mask=None,
            agent_dir_idx=agent_dir_idx,
        )
        # img = np.transpose(img, axes=(1,0,2))
        if k_rot90 > 0:
            img = np.rot90(img, k=k_rot90)

        self.window.show_img(img)

    def _render_state(
        self, agent_view_size, state, highlight=True, tile_size=TILE_PIXELS
    ):
        """
        Render the state
        """
        self._lazy_init_window()

        # padding = agent_view_size - 2  # show
        # grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        # grid_offset = np.array([1, 1])
        # h, w = grid.shape[:2]

        grid = state.grid
        agents = state.agents

        def _include_agents(grid, agent):
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir
            return (
                grid.at[pos.y, pos.x].set([StaticObject.AGENT, inventory, direction]),
                None,
            )

        grid, _ = jax.lax.scan(_include_agents, grid, agents)

        # Render the whole grid
        img = OvercookedV2Visualizer._render_grid(
            grid,
            tile_size,
        )
        self.window.show_img(img)

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

                if ingredients & DynamicObject.COOKED:
                    plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
                    rendering.fill_coords(img, plate_fn, COLORS["white"])
                    onion_fn = rendering.point_in_circle(0.5, 0.5, 0.13)
                    rendering.fill_coords(img, onion_fn, COLORS["orange"])
                elif ingredients & DynamicObject.PLATE:
                    plate_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
                    rendering.fill_coords(img, plate_fn, COLORS["white"])
                elif ingredients & DynamicObject.INGREDIENT:
                    onion_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
                    rendering.fill_coords(img, onion_fn, COLORS["yellow"])

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
    def _render_pot(cls, cell, img):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > 0
        is_cooked = (ingredients & DynamicObject.COOKED) != 0

        # num_onions = np.max([23 - pot_status, 0])
        # is_cooking = np.array((pot_status < 20) * (pot_status > 0))
        # is_done = np.array(pot_status == 0)
        num_onions = 3

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

        if ingredients > 0 and not is_cooked:
            onion_fns = [
                rendering.point_in_circle(*coord, 0.13)
                for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
            ]
            onion_fns = onion_fns[:num_onions]
            [
                rendering.fill_coords(img, onion_fn, COLORS["yellow"])
                for onion_fn in onion_fns
            ]
            if not is_cooking:
                lid_fn = rendering.rotate_fn(
                    lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi
                )
                handle_fn = rendering.rotate_fn(
                    handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi
                )

        # Render done soup
        if is_cooked:
            soup_fn = rendering.point_in_rect(0.12, 0.88, 0.23, 0.35)
            rendering.fill_coords(img, soup_fn, COLORS["orange"])
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
        if ingredients & DynamicObject.PLATE:
            plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
            rendering.fill_coords(img, plate_fn, COLORS["white"])
        if ingredients & DynamicObject.COOKED:
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.13)
            rendering.fill_coords(img, onion_fn, COLORS["orange"])
        if ingredients == DynamicObject.INGREDIENT:
            onion_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
            rendering.fill_coords(img, onion_fn, COLORS["yellow"])

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
        # # Hash map lookup key for the cache
        # if obj is not None and obj[0] == OBJECT_TO_INDEX["agent"]:
        #     # Get inventory of this specific agent
        #     if agent_inv is not None:
        #         color_idx = obj[1]
        #         agent_inv = agent_inv[COLOR_TO_AGENT_INDEX[color_idx]]
        #         agent_inv = np.array([agent_inv, -1, 0])

        #     if agent_dir_idx is not None:
        #         obj = np.array(obj)

        #         # TODO: Fix this for multiagents. Currently the orientation of other agents is wrong
        #         if len(agent_dir_idx) == 1:
        #             # Hacky way of making agent views orientations consistent with global view
        #             obj[-1] = agent_dir_idx[0]

        # no_object = obj is None or (
        #     obj[0] in [OBJECT_TO_INDEX["empty"], OBJECT_TO_INDEX["unseen"]]
        #     and obj[2] == 0
        # )

        # if not no_object:
        #     if agent_inv is not None and obj[0] == OBJECT_TO_INDEX["agent"]:
        #         key = (*obj, *agent_inv, highlight, tile_size)
        #     else:
        #         key = (*obj, highlight, tile_size)
        # else:
        #     key = (obj, highlight, tile_size)

        # if key in cls.tile_cache:
        #     return cls.tile_cache[key]

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

        # if not no_object:
        OvercookedV2Visualizer._render_cell(obj, img)
        # # render inventory
        # if agent_inv is not None and obj[0] == OBJECT_TO_INDEX["agent"]:
        #     OvercookedV2Visualizer._render_inv(agent_inv, img)

        if highlight:
            rendering.highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, subdivs)

        # Cache the rendered tile
        # cls.tile_cache[key] = img

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
