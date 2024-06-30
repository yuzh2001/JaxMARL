"""
Short introduction to running the Overcooked environment and visualising it using random actions.
"""

import jax
from jaxmarl import make
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
import time

# Parameters + random keys
max_steps = 2
key = jax.random.PRNGKey(0)

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
layout = "cramped_room"

# Or make your own!
# custom_layout_grid = """
# WWOWW
# WA  W
# B P X
# W  AW
# WWOWW
# """
# layout = layout_grid_to_dict(custom_layout_grid)

# Instantiate environment
env = make("overcooked_v2", layout=layout, max_steps=max_steps)


def part_1(key):
    key, key_r, key_a = jax.random.split(key, 3)

    obs, state = env.reset(key_r)
    print("list of agents in environment", env.agents)

    # Sample random actions
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }
    print("example action dict", actions)

    # state_seq = []
    # for _ in range(max_steps):
    #     state_seq.append(state)
    #     # Iterate random keys and sample actions
    #     key, key_s, key_a = jax.random.split(key, 3)
    #     key_a = jax.random.split(key_a, env.num_agents)

    #     actions = {
    #         agent: env.action_space(agent).sample(key_a[i])
    #         for i, agent in enumerate(env.agents)
    #     }

    #     # Step environment
    #     obs, state, rewards, dones, infos = env.step(key_s, state, actions)

    def _step(state, key):
        key_action, key_step = jax.random.split(key)
        actions = {
            agent: env.action_space(agent).sample(key_action)
            for i, agent in enumerate(env.agents)
        }

        # Step environment
        obs, state, rewards, dones, infos = env.step(key_step, state, actions)

        return state, state

    keys = jax.random.split(key, max_steps)
    _, state_seq = jax.lax.scan(_step, state, keys)

    return state_seq


def part_2(state_seq):
    # Visualize
    viz = OvercookedV2Visualizer()

    # print(state_seq)

    # Or save an animation
    viz.animate(state_seq, filename="animation.gif", agent_view_size=1)


# viz = OvercookedV2Visualizer()

# # # Render to screen
# # for s in state_seq:
# #     viz.render(env.agent_view_size, s, highlight=False)
# #     time.sleep(0.25)

# # # Or save an animation
# viz.animate(state_seq, agent_view_size=5, filename="animation.gif")


if __name__ == "__main__":

    start_time = time.time()  # Renamed variable to avoid conflict

    with jax.disable_jit(True):
        state_seq = part_1(key)
        print("done part 1")

        part_2(state_seq)

        print("done")

    print(
        "time taken", time.time() - start_time
    )  # Updated to use the new variable name
