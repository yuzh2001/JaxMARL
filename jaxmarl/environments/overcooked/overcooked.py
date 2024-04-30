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
    up = 0
    down = 1
    right = 2
    left = 3
    stay = 4
    interact = 5
    done = 6


@struct.dataclass
class Cell:
    static_item: int
    ingredients: int
    extra: int


@struct.dataclass
class State:
    # agent_pos: chex.Array
    # agent_dir: chex.Array
    # agent_dir_idx: chex.Array
    # agent_inv: chex.Array
    # goal_pos: chex.Array
    # pot_pos: chex.Array
    # wall_map: chex.Array
    # maze_map: chex.Array
    # time: int
    # terminal: bool

    agents: chex.Array

    grid: chex.Array

    time: chex.Array
    terminal: bool


# Pot status indicated by an integer, which ranges from 23 to 0
POT_EMPTY_STATUS = 23  # 22 = 1 onion in pot; 21 = 2 onions in pot; 20 = 3 onions in pot
POT_FULL_STATUS = 20  # 3 onions. Below this status, pot is cooking, and status acts like a countdown timer.
POT_READY_STATUS = 0
MAX_ONIONS_IN_POT = 3  # A pot has at most 3 onions. A soup contains exactly 3 onions.

URGENCY_CUTOFF = 40  # When this many time steps remain, the urgency layer is flipped on
DELIVERY_REWARD = 20


class Overcooked(MultiAgentEnv):
    """Vanilla Overcooked"""

    def __init__(
        self,
        layout=FrozenDict(layouts["cramped_room"]),
        random_reset: bool = False,
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

        self.action_set = jnp.array([
            Actions.up,
            Actions.down,
            Actions.right,
            Actions.left,
            Actions.stay,
            Actions.interact,
        ])

        self.random_reset = random_reset
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
        # obs = self.get_obs_v2(state)

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
        """Reset environment state based on `self.random_reset`

        If True, everything is randomized, including agent inventories and positions, pot states and items on counters
        If False, only resample agent orientations

        In both cases, the environment layout is determined by `self.layout`
        """

        # Whether to fully randomize the start state
        random_reset = self.random_reset
        layout = self.layout

        h = self.height
        w = self.width
        num_agents = self.num_agents

        grid = jnp.array(layout["grid"], dtype=jnp.uint32)

        state = State(
            agents=jnp.array(layout["agents"]),
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
        action: chex.Array,
    ) -> Tuple[State, float]:

        # Update agent position (forward action)
        is_move_action = jnp.logical_and(
            action != Actions.stay, action != Actions.interact
        )
        is_move_action_transposed = jnp.expand_dims(
            is_move_action, 0
        ).transpose()  # Necessary to broadcast correctly

        fwd_pos = jnp.minimum(
            jnp.maximum(
                state.agent_pos
                + is_move_action_transposed * DIR_TO_VEC[jnp.minimum(action, 3)]
                + ~is_move_action_transposed * state.agent_dir,
                0,
            ),
            jnp.array((self.width - 1, self.height - 1), dtype=jnp.uint32),
        )

        # Can't go past wall or goal
        def _wall_or_goal(fwd_position, wall_map, goal_pos):
            fwd_wall = wall_map.at[fwd_position[1], fwd_position[0]].get()
            goal_collision = lambda pos, goal: jnp.logical_and(
                pos[0] == goal[0], pos[1] == goal[1]
            )
            fwd_goal = jax.vmap(goal_collision, in_axes=(None, 0))(
                fwd_position, goal_pos
            )
            # fwd_goal = jnp.logical_and(fwd_position[0] == goal_pos[0], fwd_position[1] == goal_pos[1])
            fwd_goal = jnp.any(fwd_goal)
            return fwd_wall, fwd_goal

        fwd_pos_has_wall, fwd_pos_has_goal = jax.vmap(
            _wall_or_goal, in_axes=(0, None, None)
        )(fwd_pos, state.wall_map, state.goal_pos)

        fwd_pos_blocked = jnp.logical_or(fwd_pos_has_wall, fwd_pos_has_goal).reshape(
            (self.num_agents, 1)
        )

        bounced = jnp.logical_or(fwd_pos_blocked, ~is_move_action_transposed)

        # Agents can't overlap
        # Hardcoded for 2 agents (call them Alice and Bob)
        agent_pos_prev = jnp.array(state.agent_pos)
        fwd_pos = (bounced * state.agent_pos + (~bounced) * fwd_pos).astype(jnp.uint32)
        collision = jnp.all(fwd_pos[0] == fwd_pos[1])

        # No collision = No movement. This matches original Overcooked env.
        alice_pos = jnp.where(
            collision,
            state.agent_pos[0],  # collision and Bob bounced
            fwd_pos[0],
        )
        bob_pos = jnp.where(
            collision,
            state.agent_pos[1],  # collision and Alice bounced
            fwd_pos[1],
        )

        # Prevent swapping places (i.e. passing through each other)
        swap_places = jnp.logical_and(
            jnp.all(fwd_pos[0] == state.agent_pos[1]),
            jnp.all(fwd_pos[1] == state.agent_pos[0]),
        )
        alice_pos = jnp.where(~collision * swap_places, state.agent_pos[0], alice_pos)
        bob_pos = jnp.where(~collision * swap_places, state.agent_pos[1], bob_pos)

        fwd_pos = fwd_pos.at[0].set(alice_pos)
        fwd_pos = fwd_pos.at[1].set(bob_pos)
        agent_pos = fwd_pos.astype(jnp.uint32)

        # Update agent direction
        agent_dir_idx = ~is_move_action * state.agent_dir_idx + is_move_action * action
        agent_dir = DIR_TO_VEC[agent_dir_idx]

        # Handle interacts. Agent 1 first, agent 2 second, no collision handling.
        # This matches the original Overcooked
        fwd_pos = state.agent_pos + state.agent_dir
        maze_map = state.maze_map
        is_interact_action = action == Actions.interact

        # Compute the effect of interact first, then apply it if needed
        candidate_maze_map, alice_inv, alice_reward = self.process_interact(
            maze_map, state.wall_map, fwd_pos[0], state.agent_inv[0]
        )
        alice_interact = is_interact_action[0]
        bob_interact = is_interact_action[1]

        maze_map = jax.lax.select(alice_interact, candidate_maze_map, maze_map)
        alice_inv = jax.lax.select(alice_interact, alice_inv, state.agent_inv[0])
        alice_reward = jax.lax.select(alice_interact, alice_reward, 0.0)

        candidate_maze_map, bob_inv, bob_reward = self.process_interact(
            maze_map, state.wall_map, fwd_pos[1], state.agent_inv[1]
        )
        maze_map = jax.lax.select(bob_interact, candidate_maze_map, maze_map)
        bob_inv = jax.lax.select(bob_interact, bob_inv, state.agent_inv[1])
        bob_reward = jax.lax.select(bob_interact, bob_reward, 0.0)

        agent_inv = jnp.array([alice_inv, bob_inv])

        # Update agent component in maze_map
        def _get_agent_updates(agent_dir_idx, agent_pos, agent_pos_prev, agent_idx):
            agent = jnp.array(
                [
                    OBJECT_TO_INDEX["agent"],
                    COLOR_TO_INDEX["red"] + agent_idx * 2,
                    agent_dir_idx,
                ],
                dtype=jnp.uint8,
            )
            agent_x_prev, agent_y_prev = agent_pos_prev
            agent_x, agent_y = agent_pos
            return agent_x, agent_y, agent_x_prev, agent_y_prev, agent

        vec_update = jax.vmap(_get_agent_updates, in_axes=(0, 0, 0, 0))
        agent_x, agent_y, agent_x_prev, agent_y_prev, agent_vec = vec_update(
            agent_dir_idx, agent_pos, agent_pos_prev, jnp.arange(self.num_agents)
        )
        empty = jnp.array([OBJECT_TO_INDEX["empty"], 0, 0], dtype=jnp.uint8)

        # Compute padding, added automatically by map maker function
        height = self.height
        padding = (state.maze_map.shape[0] - height) // 2

        maze_map = maze_map.at[padding + agent_y_prev, padding + agent_x_prev, :].set(
            empty
        )
        maze_map = maze_map.at[padding + agent_y, padding + agent_x, :].set(agent_vec)

        # Update pot cooking status
        def _cook_pots(pot):
            pot_status = pot[-1]
            is_cooking = jnp.array(pot_status <= POT_FULL_STATUS)
            not_done = jnp.array(pot_status > POT_READY_STATUS)
            pot_status = (
                is_cooking * not_done * (pot_status - 1) + (~is_cooking) * pot_status
            )  # defaults to zero if done
            return pot.at[-1].set(pot_status)

        pot_x = state.pot_pos[:, 0]
        pot_y = state.pot_pos[:, 1]
        pots = maze_map.at[padding + pot_y, padding + pot_x].get()
        pots = jax.vmap(_cook_pots, in_axes=0)(pots)
        maze_map = maze_map.at[padding + pot_y, padding + pot_x, :].set(pots)

        reward = alice_reward + bob_reward

        return (
            state.replace(
                agent_pos=agent_pos,
                agent_dir_idx=agent_dir_idx,
                agent_dir=agent_dir,
                agent_inv=agent_inv,
                maze_map=maze_map,
                terminal=False,
            ),
            reward,
        )

    def process_interact(
        self,
        # maze_map: chex.Array,
        # wall_map: chex.Array,
        # fwd_pos: chex.Array,
        # inventory: chex.Array,
        grid: chex.Array,
        inventory: chex.Array,
        agent: Agent,
    ):
        """Assume agent took interact actions. Result depends on what agent is facing and what it is holding."""

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
        object_is_agent = jnp.array(interact_item == StaticObject.AGENT)
        object_is_wall = jnp.array(interact_item == StaticObject.WALL)

        object_has_no_ingredients = jnp.array(interact_ingredients == 0)

        # Whether the object in front is counter space that the agent can drop on.
        is_counter = jnp.logical_and(
            object_is_wall
        )
        counter_is_empty = is_counter * object_has_no_ingredients

        inventory_is_empty = inventory == 0
        inventory_is_ingredient = (inventory & DynamicObject.PLATE) == 0
        inventory_is_dish = (
            inventory & (DynamicObject.PLATE | DynamicObject.COOKED)
        ) != 0

        pot_is_cooking = object_is_pot * (interact_extra > 0)
        pot_is_cooked = object_is_pot * (
            interact_ingredients & DynamicObject.COOKED != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked

        # Interactions:
        # counter (wall)
        # pot
        # goal
        # agent
        # pile
        # empty

        successful_pickup = inventory_is_empty * (
            object_is_pile + pot_is_cooked + (is_counter * ~counter_is_empty)
        )
        successful_drop = (
            counter_is_empty * ~inventory_is_empty
            + pot_is_idle * inventory_is_ingredient
        )
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        new_ingredients = (
            successful_drop * (interact_ingredients | inventory)
            + no_effect * interact_ingredients
        )

        new_extra = pot_is_cooking * (interact_extra - 1)
        new_cell = Cell(
            static_item=interact_item,
            ingredients=new_ingredients,
            extra=new_extra,
        )

        new_grid = grid.at[fwd_pos[1], fwd_pos[0]].set(new_cell)
        new_inventory = successful_pickup * interact_ingredients + no_effect * inventory
        reward = jnp.array(successful_delivery, dtype=float) * DELIVERY_REWARD

        return new_grid, new_inventory, reward

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
