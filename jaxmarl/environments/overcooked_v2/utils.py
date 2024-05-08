import jax


def tree_select(predicate, a, b):
    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(predicate, x, y), a, b)


def compute_view_box(x, y, agent_view_size, height, width):
    """Compute the view box for an agent centered at (x, y)"""
    x_low = x - agent_view_size
    x_high = x + agent_view_size + 1
    y_low = y - agent_view_size
    y_high = y + agent_view_size + 1

    x_low = jax.lax.clamp(0, x_low, width)
    x_high = jax.lax.clamp(0, x_high, width)
    y_low = jax.lax.clamp(0, y_low, height)
    y_high = jax.lax.clamp(0, y_high, height)

    return x_low, x_high, y_low, y_high
