import math
from jaxmarl.environments.overcooked_v2.utils import compute_view_box
from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering as rendering
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject
from jaxmarl.environments.overcooked_v2.settings import POT_COOK_TIME
import imageio
from functools import partial

TILE_PIXELS = 32

COLORS = {
    "red": jnp.array([255, 0, 0]),
    "green": jnp.array([0, 255, 0]),
    "blue": jnp.array([0, 0, 255]),
    "purple": jnp.array([112, 39, 195]),
    "yellow": jnp.array([255, 255, 0]),
    "grey": jnp.array([100, 100, 100]),
    "white": jnp.array([255, 255, 255]),
    "black": jnp.array([25, 25, 25]),
    "orange": jnp.array([230, 180, 0]),
    "pink": jnp.array([255, 105, 180]),
    "brown": jnp.array([139, 69, 19]),
    "cyan": jnp.array([0, 255, 255]),
    "light_blue": jnp.array([173, 216, 230]),
}

INGREDIENT_COLORS = jnp.array(
    [
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
)


class OvercookedV2Visualizer:
    """
    Manages a window and renders contents of EnvState instances to it.
    """

    tile_cache = {}

    def __init__(self, agent_view_size=None, tile_size=TILE_PIXELS):
        self.window = None

        self.agent_view_size = agent_view_size
        self.tile_size = tile_size

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window("Overcooked V2")

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, state):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        self._lazy_init_window()

        img = self._render_state(state)

        self.window.show_img(img)

    def animate(self, state_seq, filename="animation.gif"):
        """Animate a gif give a state sequence and save if to file."""

        # def get_frame(state):
        #     frame = OvercookedV2Visualizer._render_state(
        #         state, agent_view_size=agent_view_size
        #     )
        #     return frame

        # frame_seq = [get_frame(state) for state in state_seq]

        frame_seq = jax.vmap(self._render_state)(state_seq)

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    @partial(jax.jit, static_argnums=(0,))
    def _render_state(self, state):
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

            new_grid = grid.at[pos.y, pos.x].set(
                [StaticObject.AGENT, inventory, direction]
            )
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, agents)

        recipe_indicator_mask = grid[:, :, 0] == StaticObject.RECIPE_INDICATOR
        new_ingredients_layer = jnp.where(
            recipe_indicator_mask,
            recipe | DynamicObject.COOKED | DynamicObject.PLATE,
            grid[:, :, 1],
        )
        grid = grid.at[:, :, 1].set(new_ingredients_layer)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)
        # if self.agent_view_size:
        #     for x, y in zip(agents.pos.x, agents.pos.y):
        #         x_low, x_high, y_low, y_high = compute_view_box(
        #             x, y, self.agent_view_size, grid.shape[0], grid.shape[1]
        #         )
        #         # highlight_mask[y_low:y_high, x_low:x_high] = True
        #         highlight_mask = highlight_mask.at[y_low:y_high, x_low:x_high].set(True)

        # Render the whole grid
        img = OvercookedV2Visualizer._render_grid(
            grid,
            highlight_mask=highlight_mask,
            tile_size=self.tile_size,
        )
        return img

    @staticmethod
    def _render_dynamic_item(
        ingredients,
        img,
        plate_fn=rendering.point_in_circle(0.5, 0.5, 0.3),
        ingredient_fn=rendering.point_in_circle(0.5, 0.5, 0.15),
        dish_positions=jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]),
    ):
        img_plate = rendering.fill_coords(img, plate_fn, COLORS["white"])
        img = jax.lax.select(ingredients & DynamicObject.PLATE, img_plate, img)

        # if DynamicObject.is_ingredient(ingredients):
        idx = DynamicObject.get_ingredient_idx(ingredients)
        img_ing = rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])
        img = jax.lax.select(DynamicObject.is_ingredient(ingredients), img_ing, img)

        # if ingredients & DynamicObject.COOKED:
        def _render_cooked_ingredient(img, x):
            idx, ingredient_idx = x

            color = INGREDIENT_COLORS[ingredient_idx]
            pos = dish_positions[idx]
            ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
            img_ing = rendering.fill_coords(img, ingredient_fn, color)

            img = jax.lax.select(ingredient_idx != -1, img_ing, img)
            return img, None

        ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        img, _ = jax.lax.scan(
            _render_cooked_ingredient,
            img,
            (jnp.arange(len(ingredient_indices)), ingredient_indices),
        )

        return img

    @staticmethod
    def _render_cell(cell, img):
        static_object = cell[0]

        def _render_empty(cell, img):
            return img

        def _render_wall(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = OvercookedV2Visualizer._render_counter(cell[1], img)

            return img

        def _render_agent(cell, img):
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )
            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            direction_reording = jnp.array([3, 1, 0, 2])
            direction = direction_reording[cell[2]]
            tri_fn = rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
            )
            img = rendering.fill_coords(img, tri_fn, COLORS["red"])

            img = OvercookedV2Visualizer._render_inv(cell[1], img)

            return img

        def _render_agent_self(cell, img):
            raise NotImplementedError()

        def _render_goal(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["green"]
            )

            return img

        def _render_pot(cell, img):
            return OvercookedV2Visualizer._render_pot(cell, img)

        def _render_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV2Visualizer._render_counter(cell[1], img)

            return img

        def _render_plate_pile(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            plate_fns = [
                rendering.point_in_circle(*coord, 0.2)
                for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]
            ]
            for plate_fn in plate_fns:
                img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            return img

        def _render_ingredient_pile(cell, img):
            ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

            img = rendering.fill_coords(
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

            for ingredient_fn in ingredient_fns:
                img = rendering.fill_coords(
                    img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx]
                )

            return img

        render_fns_dict = {
            StaticObject.EMPTY: _render_empty,
            StaticObject.WALL: _render_wall,
            StaticObject.AGENT: _render_agent,
            StaticObject.SELF_AGENT: _render_agent_self,
            StaticObject.GOAL: _render_goal,
            StaticObject.POT: _render_pot,
            StaticObject.RECIPE_INDICATOR: _render_recipe_indicator,
            StaticObject.PLATE_PILE: _render_plate_pile,
        }

        render_fns = [_render_empty] * (max(render_fns_dict.keys()) + 1)
        for key, value in render_fns_dict.items():
            render_fns[key] = value
        render_fns[-1] = _render_ingredient_pile

        branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

        return jax.lax.switch(
            branch_idx,
            render_fns,
            cell,
            img,
        )

    @staticmethod
    def _render_counter(ingredients, img):
        plate_fn = rendering.point_in_circle(0.5, 0.5, 0.3)
        img_plate = rendering.fill_coords(img, plate_fn, COLORS["white"])
        img = jax.lax.select(ingredients & DynamicObject.PLATE, img_plate, img)

        # if DynamicObject.is_ingredient(ingredients):
        idx = DynamicObject.get_ingredient_idx(ingredients)
        ingredient_fn = rendering.point_in_circle(0.5, 0.5, 0.15)
        img_ing = rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])
        img = jax.lax.select(DynamicObject.is_ingredient(ingredients), img_ing, img)

        # if ingredients & DynamicObject.COOKED:
        def _render_cooked_ingredient(img, x):
            idx, ingredient_idx = x

            positions = jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)])

            color = INGREDIENT_COLORS[ingredient_idx]
            # pos = positions[jnp.minimum(idx, len(positions) - 1)]
            pos = positions[idx]
            ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
            img_ing = rendering.fill_coords(img, ingredient_fn, color)

            img = jax.lax.select(ingredient_idx != -1, img_ing, img)
            return img, None

        ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        img, _ = jax.lax.scan(
            _render_cooked_ingredient,
            img,
            (jnp.arange(len(ingredient_indices)), ingredient_indices),
        )

        return img

    @staticmethod
    def _render_pot(cell, img):
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

    @staticmethod
    def _render_inv(ingredients, img):
        # print("ingredients: ", ingredients)
        # if DynamicObject.is_ingredient(ingredients):
        idx = DynamicObject.get_ingredient_idx(ingredients)
        ingredient_fn = rendering.point_in_circle(0.75, 0.75, 0.15)
        img_ing = rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])
        img = jax.lax.select(DynamicObject.is_ingredient(ingredients), img_ing, img)

        # if ingredients & DynamicObject.PLATE:
        plate_fn = rendering.point_in_circle(0.75, 0.75, 0.2)
        img_plate = rendering.fill_coords(img, plate_fn, COLORS["white"])
        img = jax.lax.select(ingredients & DynamicObject.PLATE, img_plate, img)

        # if ingredients & DynamicObject.COOKED:
        positions = [(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]
        ingredient_indices = DynamicObject.get_ingredient_idx_list(ingredients)

        for idx, ingredient_idx in enumerate(ingredient_indices):
            color = INGREDIENT_COLORS[ingredient_idx]
            pos = positions[min(idx, len(positions) - 1)]
            ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.10)
            rendering.fill_coords(img, ingredient_fn, color)

    @staticmethod
    def _render_tile(
        obj,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
    ):
        """
        Render a tile and cache the result
        """
        # key = (*obj.tolist(), highlight, tile_size)

        # if key in OvercookedV2Visualizer.tile_cache:
        #     return OvercookedV2Visualizer.tile_cache[key]

        img = jnp.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=jnp.uint8
        )

        # Draw the grid lines (top and left edges)
        rendering.fill_coords(
            img, rendering.point_in_rect(0, 0.031, 0, 1), jnp.array([100, 100, 100])
        )
        rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 0.031), jnp.array([100, 100, 100])
        )

        OvercookedV2Visualizer._render_cell(obj, img)

        if highlight:
            rendering.highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, subdivs)

        # Cache the rendered tile
        # OvercookedV2Visualizer.tile_cache[key] = img

        return img

    @staticmethod
    def _render_grid(
        grid,
        highlight_mask=None,
        tile_size=TILE_PIXELS,
    ):
        if highlight_mask is None:
            highlight_mask = jnp.zeros(shape=grid.shape[:2], dtype=bool)

        # Compute the total grid size in pixels
        width_px = grid.shape[1] * tile_size
        height_px = grid.shape[0] * tile_size

        img = jnp.zeros(shape=(height_px, width_px, 3), dtype=jnp.uint8)

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
