import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# from jaxmarl.gridworld.maze import Maze #, Actions
# from jaxmarl.gridworld.ma_maze import MAMaze
from jaxmarl.environments.overcooked_v2.overcooked import Overcooked, Actions
from jaxmarl.environments.overcooked_v2.layouts import overcooked_layouts as layouts
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer


def redraw(state, obs, extras):
    extras["viz"].render(extras["agent_view_size"], state, highlight=False)

    # if extras['obs_viz'] is not None:
    #     if extras['env'] == "MAMaze" or "Overcooked":
    #         obs_viz.render_grid(np.asarray(obs['image'][0]), k_rot90=3, agent_dir_idx=[3])
    #         obs_viz2.render_grid(np.asarray(obs['image'][1]), k_rot90=3, agent_dir_idx=[3])
    #     else:
    #         obs_viz.render_grid(np.asarray(obs['image']), k_rot90=3, agent_dir_idx=3)


def reset(key, env, extras):
    key, subkey = jax.random.split(extras["rng"])
    obs, state = extras["jit_reset"](subkey)

    extras["rng"] = key
    extras["obs"] = obs
    extras["state"] = state

    redraw(state, obs, extras)


def step(env, action, extras):
    key, subkey = jax.random.split(extras["rng"])

    actions = {"agent_0": jnp.array(action), "agent_1": jnp.array(action)}
    print("Actions : ", actions)
    obs, state, reward, done, info = jax.jit(env.step_env)(
        subkey, extras["state"], actions
    )
    extras["obs"] = obs
    extras["state"] = state
    print(f"t={state.time}: reward={reward['agent_0']}, done = {done['__all__']}")

    # print("Reward: ", reward)
    # print("obs shape: ", obs["agent_0"].shape)
    # print("obs: ", obs["agent_0"])
    # print(state.grid.shape)

    # if extras["debug"]:
    #     layers = [f"player_{i}_loc" for i in range(2)]
    #     layers.extend([f"player_{i // 4}_orientation_{i % 4}" for i in range(8)])
    #     layers.extend(
    #         [
    #             "pot_loc",
    #             "counter_loc",
    #             "onion_disp_loc",
    #             "tomato_disp_loc",
    #             "plate_disp_loc",
    #             "serve_loc",
    #             "onions_in_pot",
    #             "tomatoes_in_pot",
    #             "onions_in_soup",
    #             "tomatoes_in_soup",
    #             "soup_cook_time_remaining",
    #             "soup_done",
    #             "plates",
    #             "onions",
    #             "tomatoes",
    #             "urgency",
    #         ]
    #     )
    #     print("obs_shape: ", obs["agent_0"].shape)
    #     print("OBS: \n", obs["agent_0"])
    #     debug_obs = jnp.transpose(obs["agent_0"], (2, 0, 1))
    #     for i, layer in enumerate(layers):
    #         print(layer)
    #         print(debug_obs[i])
    # print(f"agent obs =\n {obs}")

    if done["__all__"]:
        key, subkey = jax.random.split(subkey)
        reset(subkey, env, extras)
    else:
        redraw(state, obs, extras)

    extras["rng"] = key


def key_handler(env, extras, event):
    print("pressed", event.key)

    match event.key:
        case "escape":
            window.close()
            return
        case "backspace":
            extras["jit_reset"]((env, extras))
            return
        case "left":
            action = Actions.left
        case "right":
            action = Actions.right
        case "up":
            action = Actions.up
        case "down":
            action = Actions.down
        case " ":
            action = Actions.interact
        case "tab":
            action = Actions.stay
        case "enter":
            action = Actions.done
        case _:
            return

    step(env, action, extras)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", type=str, help="Overcooked layout", default="cramped_room"
    )
    # parser.add_argument(
    #     '--random_reset',
    #     default=False,
    #     help="Reset to random state",
    #     action='store_true'
    # )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=0,
    )
    parser.add_argument(
        "--render_agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )
    # parser.add_argument(
    #     '--agent_view_size',
    #     default=5,
    #     type=int,
    #     help="Number of walls",
    # )
    parser.add_argument(
        "--debug", default=False, help="Debug mode", action="store_true"
    )
    args = parser.parse_args()

    if len(args.layout) == 0:
        raise ValueError("You must provide a layout.")
    layout = layouts[args.layout]
    env = Overcooked(layout=layout)

    viz = OvercookedV2Visualizer()
    obs_viz = None
    obs_viz2 = None

    with jax.disable_jit(False):
        jit_reset = jax.jit(env.reset)
        # jit_reset = env.reset_env
        key = jax.random.PRNGKey(args.seed)
        key, subkey = jax.random.split(key)
        o0, s0 = jit_reset(subkey)
        viz.render(0, s0, highlight=False)

        print("obs shape: ", o0["agent_0"].shape)
        print("obs: ", o0["agent_0"])

        key, subkey = jax.random.split(key)
        extras = {
            "rng": subkey,
            "state": s0,
            "obs": o0,
            "viz": viz,
            "obs_viz": obs_viz,
            "obs_viz2": obs_viz2,
            "jit_reset": jit_reset,
            # "agent_view_size": env.agent_view_size,
            "agent_view_size": 0,
            "debug": args.debug,
        }

        viz.window.reg_key_handler(partial(key_handler, env, extras))
        viz.show(block=True)
