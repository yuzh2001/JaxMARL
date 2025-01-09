import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from baselines.MAPPO.mappo_rnn_walker import ActorRNN, ScannedRNN
from jaxmarl.environments.multiwalker.multiwalker_env import MultiWalkerEnv
from jaxmarl.wrappers.baselines import load_params


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_walker"
)
def main(config):
    config = OmegaConf.to_container(config)
    config["NUM_ENVS"] = 1
    params = load_params(
        "checkpoints/rebuild-mw-mappo/gallant-glade-4/model.safetensors"
    )

    key = jax.random.PRNGKey(0)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    # Initialise environment.
    env: MultiWalkerEnv = MultiWalkerEnv(n_walkers=3)
    actor_network = ActorRNN(env.action_space(env.agents[0]).shape[0], config=config)
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]

    ac_params = params
    ac_init_hstate = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
    )

    # def batchify(x):
    #     return jnp.stack([x[agent] for agent in env.agents])

    # def unbatchify(x):
    #     return {agent: x[i] for i, agent in enumerate(env.agents)}
    def batchify(x: dict, agent_list, num_actors):
        x = jnp.stack([x[a] for a in agent_list])
        return x.reshape((num_actors, -1))

    def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
        x = x.reshape((num_actors, num_envs, -1))
        return {a: x[i] for i, a in enumerate(agent_list)}

    max_step = 500
    # Reset the environment.
    last_obs, state = env.reset(key_reset)
    last_done = jnp.zeros((config["NUM_ACTORS"]), dtype=bool)
    rng = jax.random.PRNGKey(0)
    frames = []
    for step in range(max_step):
        rng, _rng = jax.random.split(rng)
        obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
        print("obs_batch", obs_batch.shape)
        ac_in = (
            obs_batch[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        # print("ac_in", ac_in[0].shape, ac_in[1].shape)
        ac_hstate, pi = actor_network.apply(ac_params, ac_init_hstate, ac_in)
        action = pi.sample(seed=_rng)
        env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

        # Perform the step transition.
        last_obs, state, reward, last_done, infos = env.step(key_step, state, env_act)
        frames.append(env.render(state, step))

    imageio.mimsave("test.gif", frames, fps=20)


if __name__ == "__main__":
    main()
