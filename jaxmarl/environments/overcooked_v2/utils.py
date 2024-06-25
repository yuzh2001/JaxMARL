import jax
import jax.numpy as jnp
from typing import List
import itertools


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

def get_possible_recipes(num_ingredients: int) -> List[List[int]]:
    """
    Get all possible recipes given the number of ingredients.
    """
    available_ingredients = list(range(num_ingredients)) * 3
    raw_combinations = itertools.combinations(available_ingredients, 3)
    unique_recipes = set(tuple(sorted(combination)) for combination in raw_combinations)
    possible_recipes = jnp.array(list(unique_recipes), dtype=jnp.int32)

    return possible_recipes
