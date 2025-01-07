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
from jax2d.sim_state import SimParams, SimState, StaticSimParams
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


@struct.dataclass
class MW_StaticSimParams(StaticSimParams):
    # State size
    num_polygons: int = 12
    num_circles: int = 12
    num_joints: int = 12
    num_thrusters: int = 12
    max_polygon_vertices: int = 4

    # Compute amount
    num_solver_iterations: int = 10
    solver_batch_size: int = 16
    do_warm_starting: bool = False
    num_static_fixated_polys: int = 4


@struct.dataclass
class MW_SimParams(SimParams):
    # Timestep size
    dt: float = 1 / 60

    # Collision and joint coefficients
    slop: float = 0.01
    baumgarte_coefficient_joints_v: float = 2.0
    baumgarte_coefficient_joints_p: float = 0.7
    baumgarte_coefficient_fjoint_av: float = 2.0
    baumgarte_coefficient_rjoint_limit_av: float = 5.0
    baumgarte_coefficient_collision: float = 0.2
    joint_stiffness: float = 0.6

    # State clipping
    clip_position: float = 15
    clip_velocity: float = 100
    clip_angular_velocity: float = 50

    # Motors and thrusters
    base_motor_speed: float = 6.0  # rad/s
    base_motor_power: float = 900.0
    base_thruster_power: float = 10.0
    motor_decay_coefficient: float = 0.1
    motor_joint_limit: float = 0.1  # rad

    # Other defaults
    base_friction: float = 0.4


class BipedalWalker:
    def __init__(
        self,
        world,
        scene: SimState,
        static_sim_params: StaticSimParams,
        init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
        init_y=TERRAIN_HEIGHT / 2 + 2 * LEG_H,
        n_walkers=3,
        seed=None,
    ):
        self.world = world
        self.scene = scene
        self.static_sim_params = static_sim_params
        self._n_walkers = n_walkers
        self.init_x = init_x
        self.init_y = init_y
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
            density=1.0,
            friction=1,
            restitution=0.0,
        )
        self.hull_index = hull_index
        self.world.change_color(hull_index, MW_COLORS["hull"][0])

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
                rotation=i * 0.5,
            )
            self.scene, leg_hull_joint_index = add_revolute_joint_to_scene(
                self.scene,
                self.static_sim_params,
                a_index=self.hull_index,
                b_index=leg_index,
                a_relative_pos=jnp.array([0.0, LEG_DOWN]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                # motor_speed=1,
                # motor_power=MOTORS_TORQUE,
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
                rotation=i * 0.5,
            )

            self.scene, leg_lower_joint_index = add_revolute_joint_to_scene(
                self.scene,
                self.static_sim_params,
                a_index=leg_index,
                b_index=leg_lower_index,
                a_relative_pos=jnp.array([0.0, -LEG_H / 2]),
                b_relative_pos=jnp.array([0.0, LEG_H / 2]),
                motor_on=True,
                # motor_speed=1,
                # motor_power=MOTORS_TORQUE,
                has_joint_limits=True,
                min_rotation=-1.6,
                max_rotation=-0.1,
            )
            self.leg_indexes.append(leg_lower_index)
            self.joint_indexes.append(leg_lower_joint_index)

            if i == -1:
                self.world.change_color(leg_index, MW_COLORS["leg:L"][0])
                self.world.change_color(leg_lower_index, MW_COLORS["leg:L"][0])
            else:
                self.world.change_color(leg_index, MW_COLORS["leg:R"][0])
                self.world.change_color(leg_lower_index, MW_COLORS["leg:R"][0])

        # class LidarCallback(Box2D.b2.rayCastCallback):
        #     def ReportFixture(self, fixture, point, normal, fraction):
        #         if (fixture.filterData.categoryBits & 1) == 0:
        #             return -1
        #         self.p2 = point
        #         self.fraction = fraction
        #         return fraction

        # self.lidar = [LidarCallback() for _ in range(10)]

        return self.scene


class MultiWalkerWorld:
    def __init__(
        self,
        sim_params: MW_SimParams,
        static_sim_params: MW_StaticSimParams,
        scene: SimState = None,
    ):
        self.static_sim_params = static_sim_params
        self.sim_params = sim_params
        self.scene = scene
        self.color_table = jnp.ones((self.static_sim_params.num_polygons, 3))

    def reset(self):
        self.color_table = jnp.ones((self.static_sim_params.num_polygons, 3))
        self.scene = create_empty_sim(
            self.static_sim_params,
            add_floor=False,
            add_walls_and_ceiling=False,
            scene_size=10,
        )

        self.walker = BipedalWalker(
            self,
            self.scene,
            self.static_sim_params,
        )
        self.scene = self.walker._reset()
        self.walker1 = BipedalWalker(self, self.scene, self.static_sim_params, init_x=5)
        self.scene = self.walker1._reset()
        self.scene, _ = add_rectangle_to_scene(
            self.scene,
            self.static_sim_params,
            position=jnp.array([0, 0]),
            dimensions=jnp.array([100, 2]),
            density=1.0,
            restitution=0.0,
            friction=1.0,
            fixated=True,
        )
        return self.scene

    def change_color(self, index, color):
        self.color_table = self.color_table.at[index].set(color)


def make_render_pixels(static_sim_params, screen_dim):
    ppud = 40
    patch_size = 800
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
    def render_pixels(state, color_table):
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


def render_bridge(world: MultiWalkerWorld, renderer):
    return renderer(world.scene, world.color_table)


def main():
    screen_dim = (600, 400)

    # Create engine with default parameters
    static_sim_params = MW_StaticSimParams()
    sim_params = MW_SimParams()
    engine = PhysicsEngine(static_sim_params)

    world = MultiWalkerWorld(sim_params, static_sim_params)
    world.reset()

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
        actions = -jnp.zeros(
            static_sim_params.num_joints + static_sim_params.num_thrusters
        )
        world.scene, _ = step_fn(world.scene, sim_params, actions)

        # Render
        pixels = render_bridge(world, renderer)
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
