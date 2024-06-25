from collections import OrderedDict
from enum import IntEnum
from typing import Optional, Union
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from typing import Tuple, Dict
import chex
from flax import struct
from flax.core.frozen_dict import FrozenDict
from jaxmarl.environments.overcooked_v2.common import (
    StaticObject,
    DynamicObject,
    Direction,
    Position,
    Agent,
)
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts, Layout
from jaxmarl.environments.overcooked_v2.utils import compute_view_box, tree_select, get_possible_recipes


URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20
POT_COOK_TIME = 20  # Time it takes to cook a pot


SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 3,
    "POT_START_COOKING": 5,
    "DISH_PICKUP": 5,
    "PLATE_PICKUP": 3,
}


class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5


ACTION_TO_DIRECTION = (
    jnp.full((len(Actions),), -1)
    .at[Actions.right]
    .set(Direction.RIGHT)
    .at[Actions.down]
    .set(Direction.DOWN)
    .at[Actions.left]
    .set(Direction.LEFT)
    .at[Actions.up]
    .set(Direction.UP)
)


class ObservationType(IntEnum):
    LEGACY = 0
    ENCODED = 1


@chex.dataclass
class State:
    agents: Agent

    # width x height x 3
    # First channel: static items
    # Second channel: dynamic items (plates and ingredients)
    # Third channel: extra info
    grid: chex.Array

    time: chex.Array
    terminal: bool

    recipe: int


class OvercookedV2(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        observation_type: ObservationType = ObservationType.LEGACY,
        agent_view_size: Optional[int] = None,
    ):
        """
        Initializes the OvercookedV2 environment.

        Args:
            layout (Layout): The layout configuration for the environment, defaulting to "cramped_room". Either a Layout object or a string key to look up a Layout in overcooked_v2_layouts.
            max_steps (int): The maximum number of steps in the environment.
            observation_type (ObservationType): The type of observation to return, either LEGACY or ENCODED.
            agent_view_size (Optional[int]): The number of blocks the agent can view in each direction, None for full grid.
        """

        if isinstance(layout, str):
            if layout not in overcooked_v2_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, allowed layouts: {overcooked_v2_layouts.keys()}"
                )
            layout = overcooked_v2_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        num_agents = len(layout.agent_positions)

        super().__init__(num_agents=num_agents)

        self.height = layout.height
        self.width = layout.width

        self.layout = layout

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.action_set = jnp.array(list(Actions))

        self.observation_type = observation_type
        self.agent_view_size = agent_view_size
        self.obs_shape = self._get_obs_shape(observation_type)

        self.max_steps = max_steps

        self.possible_recipes = get_possible_recipes(layout.num_ingredients)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(
            indices=jnp.array([actions[f"agent_{i}"] for i in range(self.num_agents)])
        )

        state, reward, shaped_rewards = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        rewards = {f"agent_{i}": reward for i in range(self.num_agents)}
        shaped_rewards = {
            f"agent_{i}": shaped_reward
            for i, shaped_reward in enumerate(shaped_rewards)
        }

        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards},
        )

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        layout = self.layout

        static_objects = layout.static_objects
        grid = jnp.stack(
            [
                static_objects,
                jnp.zeros_like(static_objects),  # ingredient channel
                jnp.zeros_like(static_objects),  # extra info channel
            ],
            axis=-1,
            dtype=jnp.int32,
        )

        num_agents = self.num_agents
        x_positions, y_positions = map(jnp.array, zip(*layout.agent_positions))
        agents = Agent(
            pos=Position(x=x_positions, y=y_positions),
            dir=jnp.full((num_agents,), Direction.UP),
            inventory=jnp.zeros((num_agents,), dtype=jnp.int32),
        )

        if layout.recipe is not None:
            fixed_recipe = jnp.array(layout.recipe)
        else:
            key, subkey = jax.random.split(key)
            # fixed_recipe = jax.random.randint(subkey, (3,), 0, layout.num_ingredients)

            # generate random index for self.possible_recipes
            fixed_recipe_idx = jax.random.randint(subkey, (), 0, len(self.possible_recipes))
            fixed_recipe = self.possible_recipes[fixed_recipe_idx]
            print("fixed_recipe: ", fixed_recipe)

        recipe = DynamicObject.get_recipe_encoding(fixed_recipe)

        print("Recipe: ", fixed_recipe)

        state = State(
            agents=agents,
            grid=grid,
            time=0,
            terminal=False,
            recipe=recipe,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        match self.observation_type:
            case ObservationType.LEGACY:
                all_obs = self.get_obs_legacy(state)
            case ObservationType.ENCODED:
                all_obs = self.get_obs_encoded(state)
            case _:
                raise ValueError("Invalid observation type")

        def _mask_obs(obs, agent):
            view_size = self.agent_view_size
            pos = agent.pos

            padded_obs = jnp.pad(
                obs,
                ((view_size, view_size), (view_size, view_size), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            sliced_obs = jax.lax.dynamic_slice(
                padded_obs,
                (pos.y, pos.x, 0),
                self.obs_shape,
            )

            return sliced_obs

        if self.agent_view_size is not None:
            all_obs = jax.vmap(_mask_obs)(all_obs, state.agents)

        return {f"agent_{i}": obs for i, obs in enumerate(all_obs)}

    def _get_obs_shape(
        self,
        obs_type: ObservationType,
    ) -> Tuple[int]:
        if self.agent_view_size:
            view_size = self.agent_view_size * 2 + 1
            view_width = jnp.minimum(self.width, view_size)
            view_height = jnp.minimum(self.height, view_size)
        else:
            view_width = self.width
            view_height = self.height

        if obs_type == ObservationType.LEGACY:
            num_ingredients = self.layout.num_ingredients
            num_layers = 18 + 4 * num_ingredients
            return (view_height, view_width, num_layers)
        elif obs_type == ObservationType.ENCODED:
            return (view_height, view_width, 3)
        else:
            raise ValueError("Invalid observation type")

    def get_obs_encoded(self, state: State) -> Dict[str, chex.Array]:
        """
        Return a full observation, of size (height x width x 3)

        First channel contains static Items such as walls, pots, goal, plate_pile and ingredient_piles
        Second channel contains dynamic Items such as plates, ingredients and dishes
        Third channel contains agent positions and orientations
        """

        agents = state.agents
        obs = state.grid

        recipe_indicator_mask = obs[:, :, 0] == StaticObject.RECIPE_INDICATOR
        new_ingredients_layer = jnp.where(
            recipe_indicator_mask, state.recipe, obs[:, :, 1]
        )
        obs = obs.at[:, :, 1].set(new_ingredients_layer)

        def _include_agents(grid, agent):
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir
            return (
                grid.at[pos.y, pos.x].set([StaticObject.AGENT, inventory, direction]),
                None,
            )

        obs, _ = jax.lax.scan(_include_agents, obs, agents)

        def _agent_obs(agent):
            pos = agent.pos
            return obs.at[pos.y, pos.x, 0].set(StaticObject.SELF_AGENT)

        return jax.vmap(_agent_obs)(agents)

    def get_obs_legacy(self, state: State) -> Dict[str, chex.Array]:

        width = self.width
        height = self.height
        num_ingredients = self.layout.num_ingredients

        static_objects = state.grid[:, :, 0]

        static_layers = [
            jnp.array(static_objects == StaticObject.WALL, dtype=jnp.uint8),
            jnp.array(static_objects == StaticObject.GOAL, dtype=jnp.uint8),
            jnp.array(static_objects == StaticObject.POT, dtype=jnp.uint8),
            jnp.array(static_objects == StaticObject.RECIPE_INDICATOR, dtype=jnp.uint8),
            jnp.array(static_objects == StaticObject.PLATE_PILE, dtype=jnp.uint8),
        ]
        for i in range(num_ingredients):
            static_layers.append(
                jnp.array(
                    static_objects == StaticObject.INGREDIENT_PILE_BASE + i,
                    dtype=jnp.uint8,
                )
            )

        def _ingridient_layers(ingredients):
            ingredients_layers = []
            for _ in range(num_ingredients):
                ingredients >>= 2
                ingredients_layers.append(jnp.array(ingredients & 0x3, dtype=jnp.uint8))
            return ingredients_layers



        ingredients = state.grid[:, :, 1]

        # recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        # ingredients = jnp.where(recipe_indicator_mask, state.recipe, ingredients)
        # ingredients = ingredients.at[state.agents.pos.y, state.agents.pos.x].set(
        #     state.agents.inventory
        # )

        ingredients_layers = [
            jnp.array(ingredients & DynamicObject.PLATE != 0, dtype=jnp.uint8),
            jnp.array(ingredients & DynamicObject.COOKED != 0, dtype=jnp.uint8),
        ]

        ingredients_layers += _ingridient_layers(ingredients)

        inventory_layers = []
        recipe_layers = []


        tmp_inventory = jnp.zeros_like(ingredients).at[state.agents.pos.y, state.agents.pos.x].set(
            state.agents.inventory
        )
        inventory_layers = _ingridient_layers(tmp_inventory)

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        tmp_recipe = jnp.where(recipe_indicator_mask, state.recipe, 0)
        recipe_layers = _ingridient_layers(tmp_recipe)

        extra_info = state.grid[:, :, 2]
        extra_layers = [
            jnp.array(
                jnp.where(static_objects == StaticObject.POT, extra_info, 0),
                dtype=jnp.uint8,
            ),
        ]

        all_agent_layer = (
            jnp.zeros((height, width, 5), dtype=jnp.uint8)
            .at[state.agents.pos.y, state.agents.pos.x, 0]
            .set(1)
            .at[state.agents.pos.y, state.agents.pos.x, state.agents.dir + 1]
            .set(1)
        )
        all_agent_layers = [all_agent_layer[:, :, i] for i in range(5)]
        agent_self_layers = [
            jnp.zeros((height, width), dtype=jnp.uint8) for _ in range(5)
        ]

        agent_layers = agent_self_layers + all_agent_layers

        # all_agent_layer = jnp.zeros((height, width), dtype=jnp.uint8)
        # print(state.agents.pos)
        # print(state.agents.dir)
        # all_agent_layer = all_agent_layer.at[
        #     state.agents.pos.y, state.agents.pos.x
        # ].set(state.agents.dir)
        # agent_self_layer = jnp.zeros((height, width), dtype=jnp.uint8)

        # agent_layers = [
        #     agent_self_layer,
        #     all_agent_layer,
        # ]

        all_layers = agent_layers + static_layers + ingredients_layers + inventory_layers + recipe_layers + extra_layers

        obs = jnp.stack(
            all_layers,
            axis=-1,
        )

        def _agent_obs(agent):
            pos = agent.pos
            # return obs.at[pos.y, pos.x, 0].set(1)
            direction = agent.dir
            self_layers = (
                jnp.zeros((height, width, 5), dtype=jnp.uint8)
                .at[pos.y, pos.x, 0]
                .set(1)
                .at[pos.y, pos.x, direction + 1]
                .set(1)
            )
            return obs.at[:, :, :5].set(self_layers).at[:, :, 5:10].add(-self_layers)

        return jax.vmap(_agent_obs)(state.agents)

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float]:
        grid = state.grid

        print("actions: ", actions)

        # Move action:
        # 1. move agent to new position (if possible on the grid)
        # 2. resolve collisions
        # 3. prevent swapping
        def _move_wrapper(agent, action):
            direction = ACTION_TO_DIRECTION[action]

            def _move(agent, dir):
                pos = agent.pos
                new_pos = pos.move_in_bounds(dir, self.width, self.height)

                new_pos = tree_select(
                    grid[new_pos.y, new_pos.x, 0] == StaticObject.EMPTY, new_pos, pos
                )

                return agent.replace(pos=new_pos, dir=direction)

            return jax.lax.cond(
                direction != -1,
                _move,
                lambda a, _: a,
                agent,
                direction,
            )

        new_agents = jax.vmap(_move_wrapper)(state.agents, actions)

        # Resolve collisions:
        def _masked_positions(mask):
            return tree_select(mask, state.agents.pos, new_agents.pos)

        def _get_collisions(mask):
            positions = _masked_positions(mask)

            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos.y, pos.x].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1

            collisions = jax.vmap(lambda p: collision_mask[p.y, p.x])(positions)
            return collisions

        initial_mask = jnp.zeros((self.num_agents,), dtype=bool)
        mask = jax.lax.while_loop(
            lambda mask: jnp.any(_get_collisions(mask)),
            lambda mask: mask | _get_collisions(mask),
            initial_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(mask))

        # Prevent swapping:
        def _compute_swapped_agents(original_positions, new_positions):
            original_positions = original_positions.to_array()
            new_positions = new_positions.to_array()

            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)

            swapped_agents = jnp.any(swap_pairs, axis=0)
            return swapped_agents

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))

        # Interact action:
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact

            def _interact(carry, agent):
                grid, reward = carry

                print("interact: ", agent.pos, agent.dir)

                new_grid, new_agent, interact_reward, shaped_reward = (
                    self.process_interact(
                        grid, agent, new_agents.inventory, state.recipe
                    )
                )

                carry = (new_grid, reward + interact_reward)
                return carry, (new_agent, shaped_reward)

            return jax.lax.cond(
                is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent
            )

        carry = (grid, 0.0)
        xs = (new_agents, actions)
        (new_grid, reward), (new_agents, shaped_rewards) = jax.lax.scan(
            _interact_wrapper, carry, xs
        )
        # shaped_rewards = jnp.full_like(shaped_rewards, jnp.sum(shaped_rewards))
        # shaped_rewards = jnp.zeros_like(shaped_rewards)

        # Cook pots:
        def _cook_wrapper(cell):
            is_pot = cell[0] == StaticObject.POT

            def _cook(cell):
                is_cooking = cell[2] > 0
                new_extra = jax.lax.select(is_cooking, cell[2] - 1, cell[2])
                finished_cooking = is_cooking * (new_extra == 0)
                new_ingredients = cell[1] | (finished_cooking * DynamicObject.COOKED)

                return jnp.array([cell[0], new_ingredients, new_extra])

            return jax.lax.cond(is_pot, _cook, lambda x: x, cell)

        new_grid = jax.vmap(jax.vmap(_cook_wrapper))(new_grid)

        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
            ),
            reward,
            shaped_rewards,
        )

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
        all_inventories: jnp.ndarray,
        recipe: int,
    ):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

        inventory = agent.inventory
        fwd_pos = agent.get_fwd_pos()

        shaped_reward = jnp.array(0, dtype=float)

        interact_cell = grid[fwd_pos.y, fwd_pos.x]

        interact_item = interact_cell[0]
        interact_ingredients = interact_cell[1]
        interact_extra = interact_cell[2]
        plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        # Booleans depending on what the object is
        object_is_plate_pile = interact_item == StaticObject.PLATE_PILE
        object_is_ingredient_pile = StaticObject.is_ingredient_pile(interact_item)

        object_is_pile = object_is_plate_pile | object_is_ingredient_pile
        object_is_pot = interact_item == StaticObject.POT
        object_is_goal = interact_item == StaticObject.GOAL
        object_is_wall = interact_item == StaticObject.WALL

        object_has_no_ingredients = interact_ingredients == 0

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        print("inventory_is_ingredient: ", inventory_is_ingredient)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        merged_ingredients = interact_ingredients + inventory
        print("merged_ingredients: ", merged_ingredients)

        pot_is_cooking = object_is_pot * (interact_extra > 0)
        pot_is_cooked = object_is_pot * (
            interact_ingredients & DynamicObject.COOKED != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked

        successful_dish_pickup = pot_is_cooked * inventory_is_plate
        is_dish_pickup_useful = merged_ingredients == plated_recipe
        shaped_reward += (
            successful_dish_pickup
            * is_dish_pickup_useful
            * SHAPED_REWARDS["DISH_PICKUP"]
        )

        successful_pickup = (
            object_is_pile * inventory_is_empty
            + successful_dish_pickup
            + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
        )

        print("successful_pickup: ", successful_pickup)
        print("object_is_pile: ", object_is_pile)
        print("inventory_is_empty: ", inventory_is_empty)

        pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3
        print("pot_full: ", pot_full)

        successful_pot_placement = pot_is_idle * inventory_is_ingredient * ~pot_full
        ingredient_selector = inventory | (inventory << 1)
        is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
            recipe & ingredient_selector
        )
        shaped_reward += (
            successful_pot_placement
            * is_pot_placement_useful
            # * jax.lax.select(
            #     is_pot_placement_useful,
            #     1,
            #     -1,
            # )
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

        successful_drop = (
            object_is_wall * object_has_no_ingredients * ~inventory_is_empty
            + successful_pot_placement
        )
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        pile_ingredient = (
            object_is_plate_pile * DynamicObject.PLATE
            + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
        )
        print("pile_ingredient: ", pile_ingredient)

        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )

        successful_pot_start_cooking = (
            pot_is_idle * ~object_has_no_ingredients * inventory_is_empty
        )
        is_pot_start_cooking_useful = interact_ingredients == recipe
        shaped_reward += (
            successful_pot_start_cooking
            * is_pot_start_cooking_useful
            # * jax.lax.select(
            #     is_pot_start_cooking_useful,
            #     1,
            #     -1,
            # )
            * SHAPED_REWARDS["POT_START_COOKING"]
        )
        new_extra = jax.lax.select(
            successful_pot_start_cooking,
            POT_COOK_TIME,
            interact_extra,
        )
        new_cell = jnp.array([interact_item, new_ingredients, new_extra])

        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

        new_inventory = (
            successful_pickup * (pile_ingredient + merged_ingredients)
            + no_effect * inventory
        )
        print("new_inventory: ", new_inventory)
        new_agent = agent.replace(inventory=new_inventory)

        is_correct_recipe = inventory == plated_recipe
        print("is_correct_recipe: ", is_correct_recipe)
        reward = (
            jnp.array(successful_delivery & is_correct_recipe, dtype=float)
            * DELIVERY_REWARD
        )

        # Plate pickup reward: number of plates in player hands < number ready/cooking/partially full pot
        inventory_is_plate = new_inventory == DynamicObject.PLATE
        successful_plate_pickup = successful_pickup * inventory_is_plate
        num_plates_in_inventory = jnp.sum(all_inventories == DynamicObject.PLATE)
        num_nonempty_pots = jnp.sum(
            (grid[:, :, 0] == StaticObject.POT) & (grid[:, :, 1] != 0)
        )
        is_plate_pickup_useful = num_plates_in_inventory < num_nonempty_pots
        # make sure there are no plates on counters to prevent reward hacking
        no_plates_on_counters = jnp.sum(grid[:, :, 1] == DynamicObject.PLATE) == 0
        shaped_reward += (
            no_plates_on_counters
            * is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )

        return new_grid, new_agent, reward, shaped_reward

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats["return"] > 0

        return _fn

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked V2"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment. Agent_id not used since action_space is uniform for all agents"""
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)

    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        h = self.height
        w = self.width
        agent_view_size = self.agent_view_size
        return spaces.Dict(
            {
                "agent_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
                "agent_dir": spaces.Discrete(4),
                "goal_pos": spaces.Box(0, max(w, h), (2,), dtype=jnp.uint32),
                "maze_map": spaces.Box(
                    0,
                    255,
                    (w + agent_view_size, h + agent_view_size, 3),
                    dtype=jnp.uint32,
                ),
                "time": spaces.Discrete(self.max_steps),
                "terminal": spaces.Discrete(2),
            }
        )

    # def max_steps(self) -> int:
    #     return self.max_steps
