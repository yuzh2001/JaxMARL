""" 
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs
"""

import jax
import jax.numpy as jnp
from typing import Dict
import chex
from functools import partial
from flax import struct
from typing import Tuple, Optional

from jaxmarl.environments.spaces import Space

@struct.dataclass
class State:
    done: chex.Array
    step: int


class MultiAgentEnv(object):
    """Jittable abstract base class for all JaxMARL Environments."""

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """
        Args:
            num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment.

        Args:
            key (chex.PRNGKey): random key

        Returns:
            Observations (Dict[str, chex.Array]): observations for each agent, keyed by agent name
            State (State): environment state
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        reset_state: Optional[State] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset using `self.reset`.

        Args:
            key (chex.PRNGKey): random key
            state (State): environment state
            actions (Dict[str, chex.Array]): agent actions, keyed by agent name
            reset_state (Optional[State], optional): Optional environment state to reset to on episode completion. Defaults to None.

        Returns:
            Observations (Dict[str, chex.Array]): next observations
            State (State): next environment state
            Rewards (Dict[str, float]): rewards, keyed by agent name
            Dones (Dict[str, bool]): dones, keyed by agent name:
            Info (Dict): info dictionary
        """

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Environment-specific step transition.
        
        Args:
            key (chex.PRNGKey): random key
            state (State): environment state
            actions (Dict[str, chex.Array]): agent actions, keyed by agent name

        Returns:
            Observations (Dict[str, chex.Array]): next observations
            State (State): next environment state
            Rewards (Dict[str, float]): rewards, keyed by agent name
            Dones (Dict[str, bool]): dones, keyed by agent name:
            Info (Dict): info dictionary
        """
        
        raise NotImplementedError

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Applies observation function to state.
        
        Args:
            State (state): Environment state
            
        Returns:
            Observations (Dict[str, chex.Array]): observations keyed by agent names"""
        raise NotImplementedError

    def observation_space(self, agent: str) -> Space:
        """Observation space for a given agent.
        
        Args:
            agent (str): agent name

        Returns:
            space (Space): observation space
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        """Action space for a given agent.
        
        Args:
            agent (str): agent name

        Returns:
            space (Space): action space
        """
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        """Returns the available actions for each agent.
        
        Args:
            state (State): environment state

        Returns:
            available actions (Dict[str, chex.Array]): available actions keyed by agent name
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes

        Format:
            agent_names: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError
