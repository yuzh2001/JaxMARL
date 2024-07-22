from collections import OrderedDict
from enum import Enum
from typing import List, Optional, Union
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
    ACTION_TO_DIRECTION,
    MAX_INGREDIENTS,
    Actions,
    StaticObject,
    DynamicObject,
    Direction,
    Position,
    Agent,
)
from jaxmarl.environments.overcooked_v2.encoding import (
    encode_ingreidients,
    encode_object,
    encoded_ingreidients_num,
    encoded_object_num,
)
from jaxmarl.environments.overcooked_v2.layouts import overcooked_v2_layouts, Layout
from jaxmarl.environments.overcooked_v2.settings import (
    DELIVERY_REWARD,
    INDICATOR_ACTIVATION_COST,
    INDICATOR_ACTIVATION_TIME,
    POT_COOK_TIME,
    SHAPED_REWARDS,
)
from jaxmarl.environments.overcooked_v2.utils import (
    compute_view_box,
    tree_select,
    compute_enclosed_spaces,
)


class ObservationType(str, Enum):
    SPARSE = "sparse"
    EMBEDDED = "embedded"


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
        observation_type: ObservationType = ObservationType.SPARSE,
        agent_view_size: Optional[int] = None,
        random_reset: bool = False,
        random_agent_positions: bool = False,
        start_cooking_interaction: bool = False,
        negative_rewards: bool = False,
        sample_recipe_on_delivery: bool = False,
    ):
        """
        Initializes the OvercookedV2 environment.

        Args:
            layout (Layout): The layout configuration for the environment, defaulting to "cramped_room". Either a Layout object or a string key to look up a Layout in overcooked_v2_layouts.
            max_steps (int): The maximum number of steps in the environment.
            observation_type (ObservationType): The type of observation to return, either LEGACY or ENCODED.
            agent_view_size (Optional[int]): The number of blocks the agent can view in each direction, None for full grid.
            random_reset (bool): Whether to reset the environment with random agent positions, inventories and pot states.
            random_agent_positions (bool): Whether to randomize agent positions. Agents will not be moved outside of their room if they are placed in an enclosed space.
            start_cooking_interaction (bool): If false the pot starts cooking automatically once three ingredients are added, if true the pot starts cooking only after the agent interacts with it.
            negative_rewards (bool): Whether to use negative rewards.
            sample_recipe_on_delivery (bool): Whether to sample a new recipe when a delivery is made. Default is on reset only.
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

        self.possible_recipes = jnp.array(layout.possible_recipes, dtype=jnp.int32)

        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions

        self.start_cooking_interaction = jnp.array(
            start_cooking_interaction, dtype=jnp.bool_
        )
        self.negative_rewards = jnp.array(negative_rewards, dtype=jnp.bool_)
        self.sample_recipe_on_delivery = jnp.array(
            sample_recipe_on_delivery, dtype=jnp.bool_
        )

        self.enclosed_spaces = compute_enclosed_spaces(
            layout.static_objects == StaticObject.EMPTY,
        )

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

        key, subkey = jax.random.split(key)
        recipe = self._sample_recipe(subkey)

        state = State(
            agents=agents,
            grid=grid,
            time=0,
            terminal=False,
            recipe=recipe,
        )

        key, key_randomize = jax.random.split(key)
        if self.random_reset:
            state = self._randomize_state(state, key_randomize)
        elif self.random_agent_positions:
            state = self._randomize_agent_positions(state, key_randomize)

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def _sample_recipe(self, key: chex.PRNGKey) -> int:
        fixed_recipe_idx = jax.random.randint(
            key, (), 0, self.possible_recipes.shape[0]
        )

        fixed_recipe = self.possible_recipes[fixed_recipe_idx]

        return DynamicObject.get_recipe_encoding(fixed_recipe)

    def _randomize_agent_positions(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize agent positions."""
        num_agents = self.num_agents
        agents = state.agents

        def _select_agent_position(taken_mask, x):
            pos, key = x

            allowed_positions = (
                self.enclosed_spaces == self.enclosed_spaces[pos.y, pos.x]
            ) & ~taken_mask
            allowed_positions = allowed_positions.flatten()

            p = allowed_positions / jnp.sum(allowed_positions)
            agent_pos_idx = jax.random.choice(key, allowed_positions.size, (), p=p)
            agent_position = Position(
                x=agent_pos_idx % self.width, y=agent_pos_idx // self.width
            )

            new_taken_mask = taken_mask.at[agent_position.y, agent_position.x].set(True)
            return new_taken_mask, agent_position

        taken_mask = jnp.zeros_like(self.enclosed_spaces, dtype=jnp.bool_)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_agents)
        _, agent_positions = jax.lax.scan(
            _select_agent_position, taken_mask, (agents.pos, keys)
        )

        # print("agent_positions: ", agent_positions)

        key, subkey = jax.random.split(key)
        directions = jax.random.randint(subkey, (num_agents,), 0, len(Direction))

        return state.replace(agents=agents.replace(pos=agent_positions, dir=directions))

    def _randomize_state(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize the state of the environment."""

        key, subkey = jax.random.split(key)
        state = self._randomize_agent_positions(state, subkey)

        num_agents = self.num_agents
        agents = state.agents
        grid = state.grid

        # Agent inventory
        def _sample_inventory(key):
            key_dish, key_ing, key_inv = jax.random.split(key, 3)

            def _sample_dish(key):
                recipe_idx = jax.random.randint(key, (), 0, len(self.possible_recipes))
                recipe = self.possible_recipes[recipe_idx]
                return (
                    DynamicObject.get_recipe_encoding(recipe)
                    | DynamicObject.COOKED
                    | DynamicObject.PLATE
                )

            ingridient_idx = jax.random.randint(
                key_ing, (), 0, self.layout.num_ingredients
            )

            possible_inventory = jnp.array(
                [
                    DynamicObject.EMPTY,
                    DynamicObject.PLATE,
                    DynamicObject.ingredient(ingridient_idx),
                    _sample_dish(key_dish),
                ],
                dtype=jnp.int32,
            )

            inventory = jax.random.choice(
                key_inv, possible_inventory, (), p=jnp.array([0.5, 0.1, 0.25, 0.15])
            )
            return inventory

        key, subkey = jax.random.split(key)
        agent_inventories = jax.vmap(_sample_inventory)(
            jax.random.split(subkey, num_agents)
        )

        def _sample_grid_states_wrapper(cell, key):

            def _sample_pot_states(key):
                key, key_ing, key_num, key_timer = jax.random.split(key, 4)
                raw_ingridients = jax.random.randint(
                    key_ing, (3,), 0, self.layout.num_ingredients
                )
                raw_ingridients = jax.vmap(DynamicObject.ingredient)(raw_ingridients)

                partial_recipe = jax.random.randint(key_num, (), 1, 4)
                mask = jnp.arange(3) < partial_recipe

                pot_ingridients_masked = jnp.sum(raw_ingridients * mask)
                if self.start_cooking_interaction:
                    pot_ingridients_full = pot_ingridients_masked
                else:
                    # without an interaction the pot is always full when cooking
                    pot_ingridients_full = jnp.sum(raw_ingridients)

                pot_timer = jax.random.randint(key_timer, (), 0, POT_COOK_TIME) + 1

                possible_states = jnp.array(
                    [
                        cell,
                        [cell[0], pot_ingridients_masked, 0],
                        [cell[0], pot_ingridients_full, pot_timer],
                        [cell[0], pot_ingridients_full | DynamicObject.COOKED, 0],
                    ]
                )
                # 0 for do nothing, 1 for not started, 2 for started cooking, 3 for finished cooking
                return jax.random.choice(
                    key, possible_states, p=jnp.array([0.4, 0.35, 0.15, 0.1])
                )

            def _sample_counter_state(key):
                key, key_ing, key_dish = jax.random.split(key, 3)

                ingridient_idx = jax.random.randint(
                    key_ing, (), 0, self.layout.num_ingredients
                )
                dish_idx = jax.random.randint(
                    key_dish, (), 0, len(self.possible_recipes)
                )
                dish = (
                    DynamicObject.get_recipe_encoding(self.possible_recipes[dish_idx])
                    | DynamicObject.COOKED
                    | DynamicObject.PLATE
                )

                possible_states = jnp.array(
                    [
                        DynamicObject.EMPTY,
                        DynamicObject.PLATE,
                        DynamicObject.ingredient(ingridient_idx),
                        dish,
                    ]
                )

                ing_layer = jax.random.choice(
                    key, possible_states, p=jnp.array([0.5, 0.1, 0.3, 0.1])
                )
                return cell.at[1].set(ing_layer)

            is_pot = cell[0] == StaticObject.POT
            is_wall = cell[0] == StaticObject.WALL
            branch_idx = 1 * is_pot + 2 * is_wall

            return jax.lax.switch(
                branch_idx,
                [
                    lambda _: cell,
                    lambda key: _sample_pot_states(key),
                    lambda key: _sample_counter_state(key),
                ],
                key,
            )

        key, subkey = jax.random.split(key)
        key_grid = jax.random.split(subkey, (self.height, self.width))
        new_grid = jax.vmap(jax.vmap(_sample_grid_states_wrapper))(grid, key_grid)

        # print("new_grid: ", new_grid)

        return state.replace(
            agents=agents.replace(inventory=agent_inventories),
            grid=new_grid,
        )

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        match self.observation_type:
            case ObservationType.SPARSE:
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

        if obs_type == ObservationType.SPARSE:
            num_ingredients = self.layout.num_ingredients
            num_layers = 18 + 4 * (num_ingredients + 2)
            return (view_height, view_width, num_layers)
        elif obs_type == ObservationType.EMBEDDED:
            return (view_height, view_width, 2)
        else:
            raise ValueError(f"Invalid observation type: {obs_type}")

    def get_encoded_obs_vocab_sizes(self) -> List[int]:
        return [
            encoded_object_num(),
            encoded_ingreidients_num(self.layout.num_ingredients),
        ]

    def get_obs_encoded(self, state: State) -> Dict[str, chex.Array]:
        """
        Return a full observation, of size (height x width x 3)

        First channel contains static Items such as walls, pots, goal, plate_pile and ingredient_piles
        Second channel contains dynamic Items such as plates, ingredients and dishes
        Third channel contains agent positions and orientations
        """

        # TODO: implement this

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

        obs_all = jax.vmap(_agent_obs)(agents)

        def _embed_obs(cell):
            static_obj = cell[0]
            dynamic_obj = cell[1]
            extra_info = cell[2]

            enc_obj = encode_object(static_obj, extra_info)
            enc_ing = encode_ingreidients(dynamic_obj)

            return jnp.array([enc_obj, enc_ing], dtype=jnp.int32)

        return jax.vmap(jax.vmap(jax.vmap(_embed_obs)))(obs_all)

    def get_obs_legacy(self, state: State) -> Dict[str, chex.Array]:

        width = self.width
        height = self.height
        num_ingredients = self.layout.num_ingredients

        static_objects = state.grid[:, :, 0]
        ingredients = state.grid[:, :, 1]
        extra_info = state.grid[:, :, 2]

        static_encoding = jnp.array(
            [
                StaticObject.WALL,
                StaticObject.GOAL,
                StaticObject.POT,
                StaticObject.RECIPE_INDICATOR,
                StaticObject.BUTTON_RECIPE_INDICATOR,
                StaticObject.PLATE_PILE,
            ]
            + [StaticObject.INGREDIENT_PILE_BASE + i for i in range(num_ingredients)]
        )
        static_layers = static_objects[..., None] == static_encoding
        print("static_layers: ", static_layers.shape)

        def _ingridient_layers(ingredients):
            shift = jnp.array([0, 1] + [2 * (i + 1) for i in range(num_ingredients)])
            mask = jnp.array([0x1, 0x1] + [0x3] * num_ingredients)

            layers = ingredients[..., None] >> shift
            layers = layers & mask

            # layers = jax.nn.one_hot(
            #     layers, MAX_INGREDIENTS + 1
            # ).reshape(height, width, -1)
            return layers

        # recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        # ingredients = jnp.where(recipe_indicator_mask, state.recipe, ingredients)
        # ingredients = ingredients.at[state.agents.pos.y, state.agents.pos.x].set(
        #     state.agents.inventory
        # )

        ingredients_layers = _ingridient_layers(ingredients)
        print("ingredients_layers: ", ingredients_layers.shape)

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        button_recipe_indicator_mask = (
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        ) & (extra_info > 0)

        recipe_ingridients = jnp.where(
            recipe_indicator_mask | button_recipe_indicator_mask, state.recipe, 0
        )
        recipe_layers = _ingridient_layers(recipe_ingridients)
        print("recipe_layers: ", recipe_layers.shape)

        extra_info = state.grid[:, :, 2]
        pot_timer_layer = jnp.where(static_objects == StaticObject.POT, extra_info, 0)
        extra_layers = jnp.stack(
            [
                pot_timer_layer,
            ],
            axis=-1,
        )

        def _agent_layers(agent):
            pos = agent.pos
            direction = agent.dir
            inv = agent.inventory

            pos_layers = (
                jnp.zeros((height, width, 1), dtype=jnp.uint8)
                .at[pos.y, pos.x, 0]
                .set(1)
            )
            dir_layers = (
                jnp.zeros((height, width, 4), dtype=jnp.uint8)
                .at[pos.y, pos.x, direction]
                .set(1)
            )
            inv_grid = jnp.zeros_like(ingredients).at[pos.y, pos.x].set(inv)
            inv_layers = _ingridient_layers(inv_grid)

            return jnp.concatenate(
                [
                    pos_layers,
                    dir_layers,
                    inv_layers,
                ],
                axis=-1,
            )

        agent_layers = jax.vmap(_agent_layers)(state.agents)
        all_agent_layers = jnp.sum(agent_layers, axis=0)

        environment_layers = jnp.concatenate(
            [
                static_layers,
                ingredients_layers,
                recipe_layers,
                extra_layers,
            ],
            axis=-1,
        )
        print("environment_layers: ", environment_layers.shape)

        def _agent_obs(agent_layer):
            other_agent_layers = all_agent_layers - agent_layer

            print("agent_layer: ", agent_layer.shape)

            return jnp.concatenate(
                [
                    agent_layer,
                    other_agent_layers,
                    environment_layers,
                ],
                axis=-1,
            )

        return jax.vmap(_agent_obs)(agent_layers)

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float]:
        grid = state.grid

        # print("actions: ", actions)

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
                grid, correct_delivery, reward = carry

                (
                    new_grid,
                    new_agent,
                    new_correct_delivery,
                    interact_reward,
                    shaped_reward,
                ) = self.process_interact(
                    grid, agent, new_agents.inventory, state.recipe
                )

                # jax.debug.print(
                #     "correct_deflivery: {a}, new_correct_delivery: {b}",
                #     a=correct_delivery,
                #     b=new_correct_delivery,
                # )
                carry = (
                    new_grid,
                    correct_delivery | new_correct_delivery,
                    reward + interact_reward,
                )
                return carry, (new_agent, shaped_reward)

            return jax.lax.cond(
                is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent
            )

        carry = (grid, False, 0.0)
        xs = (new_agents, actions)
        (new_grid, new_correct_delivery, reward), (new_agents, shaped_rewards) = (
            jax.lax.scan(_interact_wrapper, carry, xs)
        )
        # shaped_rewards = jnp.full_like(shaped_rewards, jnp.sum(shaped_rewards))
        # shaped_rewards = jnp.zeros_like(shaped_rewards)

        # jax.debug.print("new_correct_delivery: {a}", a=new_correct_delivery)

        # Update extra info:
        def _timestep_wrapper(cell):
            # is_pot = cell[0] == StaticObject.POT
            # is_button_indicator = cell[0] == StaticObject.BUTTON_RECIPE_INDICATOR

            def _cook(cell):
                is_cooking = cell[2] > 0
                new_extra = jax.lax.select(is_cooking, cell[2] - 1, cell[2])
                finished_cooking = is_cooking * (new_extra == 0)
                new_ingredients = cell[1] | (finished_cooking * DynamicObject.COOKED)

                return jnp.array([cell[0], new_ingredients, new_extra])

            def _indicator(cell):
                new_extra = jnp.clip(cell[2] - 1, min=0)
                return cell.at[2].set(new_extra)

            # return jax.lax.cond(is_pot, _cook, lambda x: x, cell)

            branches = (
                jnp.array(
                    [
                        StaticObject.POT,
                        StaticObject.BUTTON_RECIPE_INDICATOR,
                    ]
                )
                == cell[0]
            )

            branch_idx = jax.lax.select(
                jnp.any(branches),
                jnp.argmax(branches) + 1,
                0,
            )

            return jax.lax.switch(
                branch_idx,
                [
                    lambda x: x,
                    _cook,
                    _indicator,
                ],
                cell,
            )

        new_grid = jax.vmap(jax.vmap(_timestep_wrapper))(new_grid)

        sample_new_recipe = new_correct_delivery & self.sample_recipe_on_delivery
        # jax.debug.print(
        #     "sample_new_recipe: {sample_new_recipe}",
        #     sample_new_recipe=sample_new_recipe,
        # )

        key, subkey = jax.random.split(key)
        new_recipe = jax.lax.cond(
            sample_new_recipe,
            lambda _, key: self._sample_recipe(key),
            lambda r, _: r,
            state.recipe,
            subkey,
        )

        # jax.debug.print(
        #     "new_correct_delivery: {a}, old_recipe: {b}, new_recipe: {c}, haha: {d}",
        #     a=new_correct_delivery,
        #     b=state.recipe,
        #     c=new_recipe,
        #     d=new_correct_delivery & self.sample_recipe_on_delivery,
        # )

        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
                recipe=new_recipe,
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
        onject_is_button_recipe_indicator = (
            interact_item == StaticObject.BUTTON_RECIPE_INDICATOR
        )

        object_has_no_ingredients = interact_ingredients == 0

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        # print("inventory_is_ingredient: ", inventory_is_ingredient)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        merged_ingredients = interact_ingredients + inventory
        # print("merged_ingredients: ", merged_ingredients)

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

        successful_indicator_activation = (
            onject_is_button_recipe_indicator
            * inventory_is_empty
            * object_has_no_ingredients
        )

        # print("successful_pickup: ", successful_pickup)
        # print("object_is_pile: ", object_is_pile)
        # print("inventory_is_empty: ", inventory_is_empty)

        pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3
        # print("pot_full: ", pot_full)

        successful_pot_placement = pot_is_idle * inventory_is_ingredient * ~pot_full
        ingredient_selector = inventory | (inventory << 1)
        is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
            recipe & ingredient_selector
        )
        shaped_reward += (
            successful_pot_placement
            * is_pot_placement_useful
            * jax.lax.select(
                is_pot_placement_useful,
                1,
                -1 if self.negative_rewards else 0,
            )
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
        # print("pile_ingredient: ", pile_ingredient)

        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )
        pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3

        successful_pot_start_cooking = (
            pot_is_idle
            * ~object_has_no_ingredients
            * inventory_is_empty
            * self.start_cooking_interaction
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
        auto_cook = pot_is_idle & pot_full_after_drop & ~self.start_cooking_interaction

        use_pot_extra = successful_pot_start_cooking | auto_cook
        new_extra = (
            use_pot_extra * POT_COOK_TIME
            + successful_indicator_activation * INDICATOR_ACTIVATION_TIME
            + ~use_pot_extra * ~successful_indicator_activation * interact_extra
        )

        new_cell = jnp.array([interact_item, new_ingredients, new_extra])

        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

        new_inventory = (
            successful_pickup * (pile_ingredient + merged_ingredients)
            + no_effect * inventory
        )
        # print("new_inventory: ", new_inventory)
        new_agent = agent.replace(inventory=new_inventory)

        is_correct_recipe = inventory == plated_recipe
        # print("is_correct_recipe: ", is_correct_recipe)

        reward = jnp.array(0, dtype=float)

        # Reward for successful delivery of a dish (negative if the dish is incorrect)
        reward += (
            successful_delivery
            * jax.lax.select(
                is_correct_recipe,
                1,
                -1 if self.negative_rewards else 0,
            )
            * DELIVERY_REWARD
        )

        # Cost for activating a button recipe indicator
        reward -= successful_indicator_activation * INDICATOR_ACTIVATION_COST

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

        correct_delivery = successful_delivery & is_correct_recipe
        return new_grid, new_agent, correct_delivery, reward, shaped_reward

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
