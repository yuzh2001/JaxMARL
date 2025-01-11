import hydra
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from baselines.MAPPO.mappo_rnn_walker import ActorRNN, ScannedRNN
from jaxmarl.environments.multiwalker.mw_marl_env import MultiWalkerEnv
from jaxmarl.wrappers.baselines import load_params


@hydra.main(
    version_base=None, config_path="config", config_name="mappo_homogenous_rnn_walker"
)
def main(config):
    config = OmegaConf.to_container(config)
    params = load_params(
        "checkpoints/rebuild-mw-mappo/confused-resonance-25/model.safetensors"
    )

    key = jax.random.PRNGKey(42)
    key, key_reset, key_act, key_step = jax.random.split(key, 4)

    # Initialise environment.
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
    actor_network = ActorRNN(env.action_space(env.agents[0]).shape[0], config=config)

    ac_params = params
    ac_hstate = ScannedRNN.initialize_carry(
        config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
    )

    # actor_train_state = TrainState.create(
    #     apply_fn=actor_network.apply,
    #     params=actor_network_params,
    #     tx=actor_tx,
    # )

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
    rng = jax.random.PRNGKey(43)
    frames = []
    for step in range(max_step):
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
        jax.debug.print("action={action}", action=action)
        env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

        # Perform the step transition.
        last_obs, state, reward, done, infos = env.step(key_step, state, env_act)
        last_done = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

        frames.append(env.render(state, step))

        if done["__all__"]:
            print(f"Episode finished after {step} steps")
            jax.debug.print(
                "position-x={state}", state=state.state.polygon.position[3 * 5 + 1]
            )
            jax.debug.print("reward={reward}", reward=reward)
            break

    imageio.mimsave("test.gif", frames, fps=20)


if __name__ == "__main__":
    main()
