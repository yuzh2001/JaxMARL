import os
from functools import partial
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jax2d.collision import find_axis_of_least_penetration
from jax2d.engine import PhysicsEngine, RigidBody
from jax2d.sim_state import SimState

from jaxmarl.environments import spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.multiwalker import (
    MultiWalkerWorld,
    _extract_joint,
    _extract_polygon,
)
from jaxmarl.environments.multiwalker.base import MW_SimParams, MW_StaticSimParams
from jaxmarl.environments.multiwalker.constants import PACKAGE_LENGTH, SCALE
from jaxmarl.environments.multiwalker.render import make_render_pixels

os.environ["SDL_VIDEODRIVER"] = "dummy"


class MultiWalkerEnv(MultiAgentEnv):
    def __init__(
        self,
        n_walkers: int = 2,
        position_noise=1e-3,
        angle_noise=1e-3,
    ):
        self.n_walkers = n_walkers
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self.screen_dim = (1200, 400)

        self.key = jax.random.PRNGKey(42)
        self.package_scale = self.n_walkers / 1.75
        self.package_length = PACKAGE_LENGTH / SCALE * self.package_scale

        # jax2d参数
        self.static_sim_params = MW_StaticSimParams(
            num_polygons=n_walkers * 5 + 1 + 2,
            num_circles=2,
            num_thrusters=2,
            num_joints=n_walkers * 4,
        )
        self.sim_params = MW_SimParams()

        # jax2d引擎
        self.engine = PhysicsEngine(self.static_sim_params)
        self.step_fn = jax.jit(self.engine.step)

        # 环境
        self.env = MultiWalkerWorld(
            self.sim_params, self.static_sim_params, n_walkers=n_walkers
        )

        # agents
        self.num_agents = n_walkers
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # 观察空间和动作空间
        self.observation_spaces = {
            agent: spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=(21,),
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                -1.0,
                1.0,
                shape=(4,),
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], SimState]:
        state = self.env.reset()
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SimState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[SimState] = None,
    ) -> Tuple[
        Dict[str, chex.Array], SimState, Dict[str, float], Dict[str, bool], Dict
    ]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset randomly."""

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: SimState,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], SimState, Dict[str, float], Dict[str, bool], Dict
    ]:
        print("环境在此步进")
        # 环境步进
        actions_as_array = jnp.array(
            [actions[agent] for agent in self.agents]
        ).flatten()
        actions_as_array = jnp.concatenate(
            [actions_as_array, jnp.zeros(self.static_sim_params.num_thrusters)]
        )
        next_state, _ = self.step_fn(state, self.sim_params, actions_as_array)

        # 获取观测
        observations = self.get_obs(next_state)

        # 计算奖励
        rewards = {agent: next_state.reward for agent in self.agents}
        rewards["__all__"] = next_state.reward

        # 计算终止
        dones = {agent: next_state.done.astype(jnp.bool_) for agent in self.agents}
        dones["__all__"] = next_state.done.astype(jnp.bool_)
        return (
            observations,
            next_state,  # type: ignore
            rewards,
            dones,
            next_state.info,
        )

    def get_obs(self, state: SimState) -> Dict[str, chex.Array]:
        # 通过取出self.env里存放的walker里的对应index，从state里取出观测

        walkers = self.env.walkers
        package_index = self.n_walkers * 5 + 1
        terrain_index = self.n_walkers * 5

        def _is_colliding(a: RigidBody, b: RigidBody):
            # Find axes of least penetration
            a_sep, _, _ = find_axis_of_least_penetration(a, b)
            b_sep, _, _ = find_axis_of_least_penetration(b, a)
            most_sep = jnp.maximum(a_sep, b_sep)
            is_colliding = (most_sep < 0) & a.active & b.active
            return is_colliding

        def _is_ground_contact(index: int):
            return _is_colliding(
                _extract_polygon(state, index), _extract_polygon(state, terrain_index)
            )

        def _get_obs(walker_idx):
            # hull
            hull_index = walker_idx * 5
            hull_state: RigidBody = _extract_polygon(state, hull_index)

            # joints
            joint_states = [
                _extract_joint(state, index)
                for index in range(walker_idx * 4, walker_idx * 4 + 4)
            ]
            # legs
            legs_on_floor = jnp.array(
                [
                    _is_ground_contact(index)
                    for index in range(walker_idx * 4, walker_idx * 4 + 4)
                ]
            )

            full_obs = jnp.array(
                [
                    hull_state.rotation,
                    hull_state.angular_velocity,
                    hull_state.velocity[0],
                    hull_state.velocity[1],
                    joint_states[0].rotation,
                    joint_states[0].motor_speed,
                    joint_states[1].rotation,
                    joint_states[1].motor_speed,
                    legs_on_floor[1],
                    joint_states[2].rotation,
                    joint_states[2].motor_speed,
                    joint_states[3].rotation,
                    joint_states[3].motor_speed,
                    legs_on_floor[3],
                    # above are the frist 14 observations
                    # however, LiDAR hasn't been added yet, so jumped 10 observations
                ]
            )
            return full_obs

        agent_obs = {
            agent: _get_obs(i) for agent, i in zip(self.agents, range(self.n_walkers))
        }

        # 获取邻居观测
        for i in range(self.n_walkers):
            neighbor_obs = []
            x = _extract_polygon(state, i * 5).position[0]
            y = _extract_polygon(state, i * 5).position[1]
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers:
                    neighbor_obs.append(0.0)
                    neighbor_obs.append(0.0)
                else:
                    xm = _extract_polygon(state, j * 5).position[0] - x
                    ym = _extract_polygon(state, j * 5).position[1] - y
                    xm = xm / self.package_length
                    ym = ym / self.package_length
                    neighbor_obs.append(
                        xm + jax.random.normal(self.key, ()) * self.position_noise
                    )
                    neighbor_obs.append(
                        ym + jax.random.normal(self.key, ()) * self.position_noise
                    )
            # 28 29, 与包裹的相对位置
            package_state = _extract_polygon(state, package_index)
            xd = package_state.position[0] - x
            yd = package_state.position[1] - y
            xd = xd / self.package_length
            yd = yd / self.package_length
            neighbor_obs.append(
                xd + jax.random.normal(self.key, ()) * self.position_noise
            )
            neighbor_obs.append(
                yd + jax.random.normal(self.key, ()) * self.position_noise
            )
            neighbor_obs.append(
                package_state.rotation
                + jax.random.normal(self.key, ()) * self.angle_noise
            )
            agent_obs[f"agent_{i}"] = jnp.concatenate(
                [agent_obs[f"agent_{i}"], jnp.array(neighbor_obs)]
            )

        return agent_obs

    def render(self, state: SimState):
        self.renderer = make_render_pixels(self.static_sim_params, self.screen_dim)


def main():
    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    # Initialise environment.
    env = MultiWalkerEnv(n_walkers=2)

    # Reset the environment.
    obs, state = env.reset(key_reset)

    # Sample random actions.
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_act[i])
        for i, agent in enumerate(env.agents)
    }

    # Perform the step transition.
    obs, state, reward, done, infos = env.step(key_step, state, actions)


if __name__ == "__main__":
    main()
