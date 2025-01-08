import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.maths import rmat
from jax2d.scene import (
    add_circle_to_scene,
    add_polygon_to_scene,
    add_rectangle_to_scene,
    add_revolute_joint_to_scene,
)
from jax2d.sim_state import SimParams, StaticSimParams
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    add_mask_to_shader,
    fragment_shader_circle,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"


def make_render_pixels(static_sim_params, screen_dim):
    ppud = 100
    patch_size = 512
    screen_padding = patch_size
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * ppud + screen_padding

    cleared_screen = clear_screen(full_screen_size, jnp.zeros(3))

    circle_shader = add_mask_to_shader(fragment_shader_circle)
    circle_renderer = make_renderer(
        full_screen_size, circle_shader, (patch_size, patch_size), batched=True
    )

    polygon_shader = add_mask_to_shader(
        make_fragment_shader_convex_dynamic_ngon_with_edges(4)
    )
    quad_renderer = make_renderer(
        full_screen_size, polygon_shader, (patch_size, patch_size), batched=True
    )

    @jax.jit
    def render_pixels(state):
        pixels = cleared_screen

        # Rectangles
        rect_positions_pixel_space = _world_space_to_pixel_space(state.polygon.position)
        rectangle_rmats = jax.vmap(rmat)(state.polygon.rotation)
        rectangle_rmats = jnp.repeat(
            rectangle_rmats[:, None, :, :],
            repeats=static_sim_params.max_polygon_vertices,
            axis=1,
        )
        rectangle_vertices_pixel_space = _world_space_to_pixel_space(
            state.polygon.position[:, None, :]
            + jax.vmap(jax.vmap(jnp.matmul))(rectangle_rmats, state.polygon.vertices)
        )
        rect_patch_positions = (rect_positions_pixel_space - (patch_size / 2)).astype(
            jnp.int32
        )
        rect_patch_positions = jnp.maximum(rect_patch_positions, 0)

        rect_colours = jnp.ones((static_sim_params.num_polygons, 3)) * 128.0
        rect_uniforms = (
            rectangle_vertices_pixel_space,
            rect_colours,
            rect_colours,
            state.polygon.n_vertices,
            state.polygon.active,
        )

        pixels = quad_renderer(pixels, rect_patch_positions, rect_uniforms)

        # Circles
        circle_positions_pixel_space = _world_space_to_pixel_space(
            state.circle.position
        )
        circle_radii_pixel_space = state.circle.radius * ppud
        circle_patch_positions = (
            circle_positions_pixel_space - (patch_size / 2)
        ).astype(jnp.int32)
        circle_patch_positions = jnp.maximum(circle_patch_positions, 0)

        circle_colours = jnp.ones((static_sim_params.num_circles, 3)) * 255.0

        circle_uniforms = (
            circle_positions_pixel_space,
            circle_radii_pixel_space,
            circle_colours,
            state.circle.active,
        )

        pixels = circle_renderer(pixels, circle_patch_positions, circle_uniforms)

        # Crop out the sides
        return jnp.rot90(
            pixels[screen_padding:-screen_padding, screen_padding:-screen_padding]
        )

    return render_pixels


def main():
    screen_dim = (500, 500)

    # Create engine with default parameters
    static_sim_params = StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state = create_empty_sim(static_sim_params, floor_offset=0.0)

    # Create a rectangle for the car body
    sim_state, (_, r_index) = add_rectangle_to_scene(
        sim_state,
        static_sim_params,
        position=jnp.array([2.0, 1.0]),
        dimensions=jnp.array([1.0, 0.4]),
    )

    # Create circles for the wheels of the car
    sim_state, (_, c1_index) = add_circle_to_scene(
        sim_state, static_sim_params, position=jnp.array([1.5, 1.0]), radius=0.35
    )
    sim_state, (_, c2_index) = add_circle_to_scene(
        sim_state, static_sim_params, position=jnp.array([2.5, 1.0]), radius=0.35
    )

    # Join the wheels to the car body with revolute joints
    # Relative positions are from the centre of masses of each object
    sim_state, _ = add_revolute_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=c1_index,
        a_relative_pos=jnp.array([-0.5, 0.0]),
        b_relative_pos=jnp.zeros(2),
        motor_on=True,
    )
    sim_state, _ = add_revolute_joint_to_scene(
        sim_state,
        static_sim_params,
        a_index=r_index,
        b_index=c2_index,
        a_relative_pos=jnp.array([0.5, 0.0]),
        b_relative_pos=jnp.zeros(2),
        motor_on=True,
    )

    # Add a triangle for a ramp - we fixate the ramp so it can't move
    triangle_vertices = jnp.array(
        [
            [0.5, 0.1],
            [0.5, -0.1],
            [-0.5, -0.1],
        ]
    )
    sim_state, (_, t1) = add_polygon_to_scene(
        sim_state,
        static_sim_params,
        position=jnp.array([2.7, 0.1]),
        vertices=triangle_vertices,
        n_vertices=3,
        fixated=True,
    )

    # Renderer
    renderer = make_render_pixels(static_sim_params, screen_dim)

    # Step scene
    step_fn = jax.jit(engine.step)

    pygame.init()
    screen_surface = pygame.display.set_mode(screen_dim)

    # gif！
    frames = []
    max_step = 200
    step = 0
    while True:
        actions = -jnp.ones(
            static_sim_params.num_joints + static_sim_params.num_thrusters
        )
        sim_state, _ = step_fn(sim_state, sim_params, actions)

        # Render
        pixels = renderer(sim_state)
        frames.append(pixels.astype(np.uint8))

        # Update screen
        surface = pygame.surfarray.make_surface(np.array(pixels)[:, ::-1])
        screen_surface.blit(surface, (0, 0))
        pygame.display.flip()
        step += 1
        if step > max_step:
            imageio.mimsave("test.gif", frames, fps=30)
            return True


if __name__ == "__main__":
    debug = False

    if debug:
        print("JIT disabled")
        with jax.disable_jit():
            main()
    else:
        main()
