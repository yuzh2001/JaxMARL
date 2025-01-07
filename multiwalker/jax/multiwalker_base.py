import os

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import pygame
from constants import (
    HULL_POLY,
    LEG_DOWN,
    LEG_H,
    LEG_W,
    MOTORS_TORQUE,
    SCALE,
    TERRAIN_HEIGHT,
    TERRAIN_STARTPAD,
    TERRAIN_STEP,
)
from flax import struct
from jax2d.engine import PhysicsEngine, create_empty_sim
from jax2d.maths import rmat
from jax2d.scene import (
    add_polygon_to_scene,
    add_rectangle_to_scene,
    add_revolute_joint_to_scene,
)
from jax2d.sim_state import SimParams
from jaxgl.renderer import clear_screen, make_renderer
from jaxgl.shaders import (
    add_mask_to_shader,
    make_fragment_shader_convex_dynamic_ngon_with_edges,
)

# set SDL to use the dummy NULL video driver,
#   so it doesn't need a windowing system.
os.environ["SDL_VIDEODRIVER"] = "dummy"
MW_COLORS = {
    "hull": [jnp.array([127, 51, 229]), jnp.array([76, 76, 127])],
    "leg:L": [jnp.array([178, 101, 152]), jnp.array([127, 76, 101])],
    "leg:R": [jnp.array([153, 76, 127]), jnp.array([102, 51, 76])],
}

color_table = jnp.ones((12, 3))


@struct.dataclass
class MW_StaticSimParams:
    # State size
    num_polygons: int = 12
    num_circles: int = 12
    num_joints: int = 12
    num_thrusters: int = 12
    max_polygon_vertices: int = 4

    # Compute amount
    num_solver_iterations: int = 10
    solver_batch_size: int = 16
    do_warm_starting: bool = True
    num_static_fixated_polys: int = 4


class BipedalWalker:
    def __init__(
        self,
        scene,
        static_sim_params,
        init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
        init_y=TERRAIN_HEIGHT + 2 * LEG_H,
        n_walkers=3,
        seed=None,
    ):
        self.scene = scene
        self.static_sim_params = static_sim_params
        self._n_walkers = n_walkers
        self.init_x = init_x
        self.init_y = init_y
        self.walker_id = -int(self.init_x)
        self._seed(seed)

        # all parts
        self.hull_index = None
        self.leg_indexes = []
        self.joint_indexes = []

    def _seed(self, seed=None):
        self.key = jax.random.PRNGKey(seed if seed is not None else 0)
        self.uniform_fn = lambda min_v, max_v, **kwargs: jax.random.uniform(
            self.key, shape=(), minval=min_v, maxval=max_v, **kwargs
        )
        return [seed]

    def _reset(self):
        global color_table
        color_table = jnp.ones((self.static_sim_params.num_polygons, 3))
        # self._destroy()
        init_x = self.init_x
        init_y = self.init_y

        # add a hull
        hull_vertices = jnp.array([[x / SCALE, y / SCALE] for x, y in HULL_POLY])
        self.scene, (_, hull_index) = add_polygon_to_scene(
            self.scene,
            self.static_sim_params,
            position=jnp.array([init_x, init_y]),
            vertices=hull_vertices,
            n_vertices=len(HULL_POLY),
            density=5.0,
            friction=0.1,
            restitution=0.0,
        )
        self.hull_index = hull_index
        color_table = color_table.at[hull_index].set(MW_COLORS["hull"][0])

        # make a force to the hull
        # _uniform_res = self.uniform_fn(-INITIAL_RANDOM, INITIAL_RANDOM)
        # self.scene, hull_thruster_index = add_thruster_to_scene(
        #     self.scene,
        #     self.hull_index,
        #     jnp.array([0.0, 0.0]),
        #     0,
        #     power=_uniform_res,
        # )
        # self.hull_thruster_index = hull_thruster_index

        # add the legs
        self.leg_indexes = []
        self.joint_indexes = []
        for i in [-1, +1]:
            self.scene, (_, leg_index) = add_rectangle_to_scene(
                self.scene,
                self.static_sim_params,
                position=jnp.array([init_x, init_y - LEG_H / 2 - LEG_DOWN]),
                dimensions=jnp.array([LEG_W, LEG_H]),
                density=1.0,
                restitution=0.0,
                rotation=i * 0.05,
            )
            self.scene, leg_hull_joint_index = add_revolute_joint_to_scene(
                self.scene,
                self.static_sim_params,
                a_index=self.hull_index,
                b_index=leg_index,
                a_relative_pos=jnp.array([0.0, LEG_DOWN]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                motor_speed=i,
                motor_power=MOTORS_TORQUE,
                has_joint_limits=True,
                min_rotation=-0.8,
                max_rotation=1.1,
            )

            self.leg_indexes.append(leg_index)
            self.joint_indexes.append(leg_hull_joint_index)

            self.scene, (_, leg_lower_index) = add_rectangle_to_scene(
                self.scene,
                self.static_sim_params,
                position=jnp.array([init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN]),
                dimensions=jnp.array([0.8 * LEG_W, LEG_H]),
                density=1.0,
                restitution=0.0,
                rotation=i * 0.05,
            )

            self.scene, leg_lower_joint_index = add_revolute_joint_to_scene(
                self.scene,
                self.static_sim_params,
                a_index=leg_index,
                b_index=leg_lower_index,
                a_relative_pos=jnp.array([0.0, -LEG_H / 2]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                motor_speed=1,
                motor_power=MOTORS_TORQUE,
                has_joint_limits=True,
                min_rotation=-1.6,
                max_rotation=-0.1,
            )
            self.leg_indexes.append(leg_lower_index)
            self.joint_indexes.append(leg_lower_joint_index)

            if i == -1:
                color_table = color_table.at[leg_index].set(MW_COLORS["leg:L"][0])
                color_table = color_table.at[leg_lower_index].set(MW_COLORS["leg:L"][0])
            else:
                color_table = color_table.at[leg_index].set(MW_COLORS["leg:R"][0])
                color_table = color_table.at[leg_lower_index].set(MW_COLORS["leg:R"][0])

        # class LidarCallback(Box2D.b2.rayCastCallback):
        #     def ReportFixture(self, fixture, point, normal, fraction):
        #         if (fixture.filterData.categoryBits & 1) == 0:
        #             return -1
        #         self.p2 = point
        #         self.fraction = fraction
        #         return fraction

        # self.lidar = [LidarCallback() for _ in range(10)]

        return self.scene


def make_render_pixels(static_sim_params, screen_dim):
    ppud = 10
    patch_size = 512
    screen_padding = patch_size
    full_screen_size = (
        screen_dim[0] + 2 * screen_padding,
        screen_dim[1] + 2 * screen_padding,
    )

    def _world_space_to_pixel_space(x):
        return x * ppud + screen_padding

    cleared_screen = clear_screen(full_screen_size, jnp.ones(3) * 200.0)

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


def main():
    screen_dim = (600, 400)

    # Create engine with default parameters
    static_sim_params = MW_StaticSimParams()
    sim_params = SimParams()
    engine = PhysicsEngine(static_sim_params)

    # Create scene
    sim_state_scene = create_empty_sim(
        static_sim_params, add_floor=True, add_walls_and_ceiling=False, scene_size=10
    )

    walker = BipedalWalker(sim_state_scene, static_sim_params)
    sim_state_scene = walker._reset()

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
        sim_state_scene, _ = step_fn(sim_state_scene, sim_params, actions)

        # Render
        pixels = renderer(sim_state_scene)
        frames.append(pixels.astype(np.uint8))
        # Update screen
        surface = pygame.surfarray.make_surface(np.array(pixels)[:, ::-1])
        screen_surface.blit(surface, (0, 0))
        pygame.display.flip()
        step += 1
        if step >= max_step:
            imageio.mimsave("test.gif", frames, fps=10)
            return True


if __name__ == "__main__":
    debug = False

    if debug:
        print("JIT disabled")
        with jax.disable_jit():
            main()
    else:
        main()
