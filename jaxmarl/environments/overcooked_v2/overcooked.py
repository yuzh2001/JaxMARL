from collections import OrderedDict
from enum import IntEnum

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
from .layouts import Agent
from .common import StaticObject, DynamicObject, Direction, Position
from typing import NamedTuple

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    OBJECT_INDEX_TO_VEC,
    DIR_TO_VEC,
    make_overcooked_map,
)
from jaxmarl.environments.overcooked.layouts import overcooked_layouts as layouts


class Actions(IntEnum):
    # Turn left, turn right, move forward
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5
    # done = 6


@struct.dataclass
class State:
    agents: chex.Array

    # width x height x 3
    # First channel: static items
    # Second channel: dynamic items (plates and ingredients)
    # Third channel: extra info
    grid: chex.Array

    time: chex.Array
    terminal: bool


URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20
POT_COOK_TIME = 20  # Time it takes to cook a pot of onions


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(
        self,
        layout=layouts["counter_circuit"],
        # random_reset: bool = False,
        max_steps: int = 400,
    ):
        # Sets self.num_agents to 2
        super().__init__(num_agents=2)

        # self.obs_shape = (agent_view_size, agent_view_size, 3)
        # Observations given by 26 channels, most of which are boolean masks
        self.height = layout["height"]
        self.width = layout["width"]
        # self.obs_shape = (420,)
        self.obs_shape = (self.width, self.height, 3)

        self.agent_view_size = (
            5  # Hard coded. Only affects map padding -- not observations.
        )
        self.layout = layout
        self.agents = ["agent_0", "agent_1"]

        self.action_set = jnp.array(list(Actions))

        # self.random_reset = random_reset
        self.max_steps = max_steps

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform single timestep state transition."""

        acts = self.action_set.take(
            indices=jnp.array([actions["agent_0"], actions["agent_1"]])
        )

        state, reward = self.step_agents(key, state, acts)

        state = state.replace(time=state.time + 1)

        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        rewards = {"agent_0": reward, "agent_1": reward}
        dones = {"agent_0": done, "agent_1": done, "__all__": done}

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {},
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
        )

        agents = jax.vmap(Agent.from_position)(layout.agent_positions)

        state = State(
            agents=agents,
            grid=grid,
            time=0,
            terminal=False,
        )

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def get_obs_v2(self, state: State) -> Dict[str, chex.Array]:
        """
        Return a full observation, of size (height x width x 3)

        First channel contains static Items such as walls, pots, goal, plate_pile and ingredient_piles
        Second channel contains dynamic Items such as plates, ingredients and dishes
        Third channel contains agent positions and orientations
        """

        width = self.width
        height = self.height
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = state.maze_map[padding:-padding, padding:-padding, 0]

        def _item_mapping(item):
            is_wall = jnp.array(item == OBJECT_TO_INDEX["wall"])
            is_pot = jnp.array(item == OBJECT_TO_INDEX["pot"])
            is_goal = jnp.array(item == OBJECT_TO_INDEX["goal"])
            is_agent = jnp.array(item == OBJECT_TO_INDEX["agent"])
            is_plate_pile = jnp.array(item == OBJECT_TO_INDEX["plate_pile"])
            is_onion_pile = jnp.array(item == OBJECT_TO_INDEX["onion_pile"])

            static_item = 0
            dynamic_item = 0
            info_item = 0

            return jnp.array(
                [
                    static_item,
                    dynamic_item,
                    info_item,
                ]
            )

        obs = jax.lax.map(_item_mapping, maze_map)

        def _agent_obs(agent_idx):
            agent_pos = state.agent_pos[agent_idx]
            return obs.at[agent_pos[1], agent_pos[0], 2].set(1)

        obs_all = {f"agent_{i}": _agent_obs(i) for i in range(self.num_agents)}
        return obs_all

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float]:

        # Move action:
        # TODO: handle collisions as previously and prevent swapping
        def _move(agent, action):

            return agent.move(action)

        blocked_grid = state.grid[..., 0] == OBJECT_TO_INDEX["wall"]
        new_agents = jax.vmap(_move)(state.agents, actions)

        # Interact action:
        def _interact(carry, agent):
            grid, reward = carry

            new_grid, new_agent, interact_reward = self.process_interact(grid, agent)

            carry = (new_grid, reward + interact_reward)
            return carry, new_agent

        def _no_interact(carry, agent):
            return carry, agent

        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact
            return jax.lax.cond(is_interact, _interact, _no_interact, carry, agent)

        carry = (state.grid, 0.0)
        xs = jnp.stack((new_agents, actions), axis=1)
        (new_grid, reward), new_agents = jax.lax.scan(_interact_wrapper, carry, xs)

        # Cook pots:
        def _cook(cell):
            is_pot = cell.static_item == StaticObject.POT
            is_cooking = is_pot * (cell.extra > 0)
            new_extra = jax.lax.select(is_cooking, cell.extra - 1, cell.extra)
            finished_cooking = is_cooking * (new_extra == 0)
            new_ingredients = cell.ingredients | (
                finished_cooking * DynamicObject.COOKED
            )

            return cell.replace(ingredients=new_ingredients, extra=new_extra)

        new_grid = jax.vmap(_cook)(new_grid)

        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
            ),
            reward,
        )

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
    ):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

        inventory = agent.inventory
        fwd_pos = agent.get_fwd_pos()

        interact_cell = grid[fwd_pos[1], fwd_pos[0]]

        interact_item = interact_cell.static_item
        interact_ingredients = interact_cell.ingredients
        interact_extra = interact_cell.extra

        # Booleans depending on what the object is
        object_is_pile = jnp.logical_or(
            interact_item == StaticObject.PLATE_PILE,
            interact_item > StaticObject.INGREDIENT_PILE,
        )
        object_is_pot = jnp.array(interact_item == StaticObject.POT)
        object_is_goal = jnp.array(interact_item == StaticObject.GOAL)
        object_is_wall = jnp.array(interact_item == StaticObject.WALL)

        object_has_no_ingredients = jnp.array(interact_ingredients == 0)

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = (inventory & DynamicObject.PLATE) == 0
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        pot_is_cooking = object_is_pot * (interact_extra > 0)
        pot_is_cooked = object_is_pot * (
            interact_ingredients & DynamicObject.COOKED != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked

        successful_pickup = (
            object_is_pile * inventory_is_empty
            + pot_is_cooked * inventory_is_plate
            + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
        )

        successful_drop = (
            object_is_wall * object_has_no_ingredients * ~inventory_is_empty
            + pot_is_idle * inventory_is_ingredient
        )
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        # TODO: watch out for duplicate ingredients and overflows
        merged_ingredients = interact_ingredients | inventory
        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )

        new_extra = jax.lax.select(
            pot_is_idle * (interact_ingredients != 0), POT_COOK_TIME, interact_extra
        )
        new_cell = Cell(
            static_item=interact_item,
            ingredients=new_ingredients,
            extra=new_extra,
        )

        new_grid = grid.at[fwd_pos[1], fwd_pos[0]].set(new_cell)
        new_inventory = successful_pickup * merged_ingredients + no_effect * inventory
        new_agent = agent.replace(inventory=new_inventory)
        reward = jnp.array(successful_delivery, dtype=float) * DELIVERY_REWARD

        return new_grid, new_agent, reward

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
        return "Overcooked"

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

    def max_steps(self) -> int:
        return self.max_steps
