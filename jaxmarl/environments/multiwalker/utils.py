from jax2d.engine import Joint, RigidBody
from jax2d.sim_state import SimState


def _extract_polygon(state: SimState, index: int) -> RigidBody:
    position = state.polygon.position[index]
    rotation = state.polygon.rotation[index]
    velocity = state.polygon.velocity[index]
    angular_velocity = state.polygon.angular_velocity[index]
    inverse_mass = state.polygon.inverse_mass[index]
    inverse_inertia = state.polygon.inverse_inertia[index]
    friction = state.polygon.friction[index]
    restitution = state.polygon.restitution[index]
    collision_mode = state.polygon.collision_mode[index]
    active = state.polygon.active[index]
    n_vertices = state.polygon.n_vertices[index]
    vertices = state.polygon.vertices[index]
    radius = state.polygon.radius[index]
    return RigidBody(
        position,
        rotation,
        velocity,
        angular_velocity,
        inverse_mass,
        inverse_inertia,
        friction,
        restitution,
        collision_mode,
        active,
        n_vertices,
        vertices,
        radius,
    )


def _extract_joint(state: SimState, index: int) -> Joint:
    a_index = state.joint.a_index[index]
    b_index = state.joint.b_index[index]
    a_relative_pos = state.joint.a_relative_pos[index]
    b_relative_pos = state.joint.b_relative_pos[index]
    global_position = state.joint.global_position[index]
    active = state.joint.active[index]

    acc_impulse = state.joint.acc_impulse[index]
    acc_r_impulse = state.joint.acc_r_impulse[index]
    motor_speed = state.joint.motor_speed[index]
    motor_power = state.joint.motor_power[index]
    motor_on = state.joint.motor_on[index]
    motor_has_joint_limits = state.joint.motor_has_joint_limits[index]
    min_rotation = state.joint.min_rotation[index]
    max_rotation = state.joint.max_rotation[index]
    is_fixed_joint = state.joint.is_fixed_joint[index]
    rotation = state.joint.rotation[index]

    return Joint(
        a_index,
        b_index,
        a_relative_pos,
        b_relative_pos,
        global_position,
        active,
        acc_impulse,
        acc_r_impulse,
        motor_speed,
        motor_power,
        motor_on,
        motor_has_joint_limits,
        min_rotation,
        max_rotation,
        is_fixed_joint,
        rotation,
    )
