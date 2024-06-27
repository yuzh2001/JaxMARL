import jax.numpy as jnp
from .common import StaticObject
from .settings import POT_COOK_TIME

INFO_NUM = (
    jnp.ones((len(StaticObject),))
    .at[StaticObject.POT]
    .set(POT_COOK_TIME + 1)
    .at[StaticObject.AGENT]
    .set(4)
    .at[StaticObject.SELF_AGENT]
    .set(4)
)


def encode_object(static_obj, extra_info):
    mask = jnp.arange(len(INFO_NUM)) < static_obj
    return jnp.sum(INFO_NUM * mask) + extra_info


def encode_ingreidients(ingredients):
    return ingredients


def encoded_object_num():
    return int(jnp.sum(INFO_NUM))


def encoded_ingreidients_num(num_ingredients):
    shift = 2 + 2 * num_ingredients
    return 1 << shift
