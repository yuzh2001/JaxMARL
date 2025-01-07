import jax
import jax.numpy as jnp
from jax2d.maths import rmat
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)


def make_render_pixels(static_sim_params, screen_dim):
    ppud = 12
    patch_size_x = 2000
    patch_size_y = 400
    screen_padding = 400
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    cleared_screen = clear_screen(full_screen_size, jnp.ones(3) * 200.0)

    polygon_shader = add_mask_to_shader(
        make_fragment_shader_convex_dynamic_ngon_with_edges(4)
    )
    quad_renderer = make_renderer(
        full_screen_size, polygon_shader, (patch_size_x, patch_size_y), batched=True
    )

    @jax.jit
    def render_pixels(state, color_table, step):
        pixels = cleared_screen

        def _world_space_to_pixel_spacestep(x):
            return jnp.array(
                [
                    [t[0] * ppud + screen_padding, t[1] * ppud + screen_padding]
                    for t in x
                ]
            )

        def _world_space_to_pixel_space(x):
            return x * ppud + screen_padding

        # Rectangles

        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :],
            repeats=static_sim_params.max_polygon_vertices,
            axis=1,
        )

        # vertices to pixel space
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :]
            + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )

        # calculate patch positions
        jax.debug.print("state.polygon.position: {x}", x=state.polygon.position)
        rect_positions_pixel_space = _world_space_to_pixel_spacestep(
            state.polygon.position
        )
        jax.debug.print("rect_positions_pixel_space: {x}", x=rect_positions_pixel_space)
        rect_patch_positions = (rect_positions_pixel_space - 100 / 2).astype(jnp.int32)
        jax.debug.print("rect_patch_positions: {x}", x=rect_patch_positions)
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)
        jax.debug.print("rect_patch_positions max: {x}", x=rect_patch_positions)

        rect_colours = jnp.array(
            [color_table[idx] for idx in range(static_sim_params.num_polygons)]
        )
        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            state.polygon.n_vertices,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)

        # Crop out the sides
        return jnp.rot90(
            pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]
        )

    return render_pixels


def render_bridge(world, renderer, step):
    return renderer(world.scene, world.color_table, step)
