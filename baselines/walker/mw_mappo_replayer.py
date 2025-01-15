from typing import Dict

import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass
from omegaconf import OmegaConf
from tqdm import trange

from baselines.MAPPO.mappo_rnn_walker import ActorRNN, ScannedRNN
from jaxmarl.environments.multiwalker.mw_marl_env import MultiWalkerEnv
from jaxmarl.wrappers.baselines import load_params

seed = 42


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_walker"
)
def main(config):
    global seed
    config = OmegaConf.to_container(config)

    params = load_params(
        "checkpoints/rebuild-mw-mappo/confused-resonance-25/model.safetensors"
    )
    env: MultiWalkerEnv = MultiWalkerEnv(n_walkers=3)
    config["NUM_ENVS"] = 1
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    @jax.jit
    def _replay(c):
        key = jax.random.PRNGKey(seed)
        key, key_reset, key_act, key_step = jax.random.split(key, 4)

        actor_network = ActorRNN(
            env.action_space(env.agents[0]).shape[0], config=config
        )
        ac_params = params
        ac_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def batchify(x: dict, agent_list, num_actors):
            x = jnp.stack([x[a] for a in agent_list])
            return x.reshape((num_actors, -1))

        def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
            x = x.reshape((num_actors, num_envs, -1))
            return {a: x[i] for i, a in enumerate(agent_list)}

        # Reset the environment.
        last_obs, state = env.reset(key_reset)
        last_done = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
        rng = jax.random.PRNGKey(seed)

        @dataclass
        class RunnerState:
            step: int
            rng: jnp.ndarray
            state: jnp.ndarray
            last_obs: jnp.ndarray
            last_done: jnp.ndarray
            reward: Dict[str, jnp.ndarray]
            ac_hstate: jnp.ndarray
            frames: jnp.ndarray

        @jax.jit
        def _step(runner_state: RunnerState):
            step = runner_state.step
            rng = runner_state.rng
            last_state = runner_state.state
            last_obs = runner_state.last_obs
            last_done = runner_state.last_done
            reward = runner_state.reward
            ac_hstate = runner_state.ac_hstate
            frames = runner_state.frames

            rng, _rng = jax.random.split(rng)
            # jax.debug.print("last_obs={last_obs}", last_obs=last_obs["agent_1"][5])
            obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            ac_in = (
                obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            ac_hstate, pi = actor_network.apply(ac_params, ac_hstate, ac_in)
            action = pi.mode()
            # action = pi.sample(seed=_rng)
            # jax.debug.print("action={action}", action=action)
            env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

            # Perform the step transition.
            last_obs, state, reward, done, infos = env.step(
                key_step, last_state, env_act
            )
            last_done = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
            # frames = jnp.append(frames, env.render(state, step))
            frames = frames.at[step].set(env.render(state, step))

            return RunnerState(
                step + 1, rng, state, last_obs, last_done, reward, ac_hstate, frames
            )

        @jax.jit
        def _cond_step(runner_state: RunnerState):
            step = runner_state.step
            last_done = runner_state.last_done
            # jax.debug.print("step={step}", step=last_done)

            cond = jnp.where(last_done[0], False, step < 500)
            return cond

        init_runner_state = RunnerState(
            0,
            rng,
            state,
            last_obs,
            last_done,
            {agent: 0 for agent in env.agents} | {"__all__": 0},
            ac_hstate,
            jnp.zeros((500, 400, 1200, 3), dtype=jnp.float32),
        )
        runner_state = jax.lax.while_loop(_cond_step, _step, init_runner_state)
        return runner_state

    def _cond_func(carry):
        global seed
        print(f"[seed={seed}] reward={carry}")
        seed = seed + 1
        return carry < 0

    # result = jax.lax.while_loop(_cond_func, _replay, 0)

    for i in trange(10):
        runner_state = _replay(1)
        seed = seed + 1

        reward = runner_state.reward["__all__"]
        frames = runner_state.frames
        step = runner_state.step
        print(f"[seed={seed}] step={step} reward={reward}")
        jax.debug.print("reward={reward}", reward=reward)

        if reward > 0:
            imageio.mimsave(f"test_{i}.gif", frames, fps=20)


if __name__ == "__main__":
    main()
