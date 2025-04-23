import numpy as np
import warp as wp
import pyvista as pv
import trimesh as tm
from scipy.spatial.transform import Rotation as R, Slerp

from examples.example_utils import init_robot_demo
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.sim_model import ASRModel
from warp_ipc.utils import log

from tqdm.rich import tqdm_rich as tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--num_envs", default=4, type=int)
parser.add_argument("--export_mesh", action="store_true")
args = parser.parse_args()

DATETIME_TAG, OUT_DIR = init_robot_demo(args, "peg_in_hole-detro-multi")

dt = 1 / 50
env_pos = (
    ASRModel.get_env_pos(
        args.num_envs,
        int(np.sqrt(args.num_envs)),
        int(args.num_envs / np.sqrt(args.num_envs)),
        0.15,
    )
    .cpu()
    .numpy()
)

sensor_1_keyframes = [
    (0.0, [-0.0171, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (0.1, [-0.0165, 0.005, 0.071], [0.70710677, 0.0, 0.70710677, 0.0]),
    (2.0, [-0.0165, -0.006, 0.066], [0.70710677, 0.0, 0.70710677, 0.0]),
    (4.0, [-0.0165, -0.006, 0.05], [0.70710677, 0.0, 0.70710677, 0.0]),
]

sensor_2_keyframes = [
    (0.0, [0.0171, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (0.1, [0.0165, 0.005, 0.071], [0.0, -0.70710677, 0.0, 0.70710677]),
    (2.0, [0.0165, -0.006, 0.066], [0.0, -0.70710677, 0.0, 0.70710677]),
    (4.0, [0.0165, -0.006, 0.05], [0.0, -0.70710677, 0.0, 0.70710677]),
]


def interpolate_keyframes(keyframes, time_step):
    p_traj = []
    q_traj = []
    for i in range(len(keyframes) - 1):
        t0, p0, q0 = keyframes[i]
        t1, p1, q1 = keyframes[i + 1]

        assert abs(t0 - round(t0 / time_step) * time_step) < 1e-6
        assert abs(t1 - round(t1 / time_step) * time_step) < 1e-6

        step0 = int(round(t0 / time_step))
        step1 = int(round(t1 / time_step))

        if i > 0:
            p_traj = p_traj[:-1]
            q_traj = q_traj[:-1]

        p_traj += np.linspace(p0, p1, int(step1 - step0 + 1)).tolist()
        slerp = Slerp([t0, t1], R.from_quat([q0, q1]))
        q_traj += slerp(np.linspace(t0, t1, int(step1 - step0 + 1))).as_quat().tolist()

    if len(p_traj) == 0:
        p_traj = [keyframes[0][1]]
        q_traj = [keyframes[0][2]]

    return np.array(p_traj), np.array(q_traj)


sensor_1_p_traj, sensor_1_q_traj = interpolate_keyframes(sensor_1_keyframes, dt)
sensor_2_p_traj, sensor_2_q_traj = interpolate_keyframes(sensor_2_keyframes, dt)


q_states = []
if __name__ == "__main__":
    wp.init()
    model = ASRModel(num_envs=args.num_envs)
    model.set_kinematic_stiffness(1e5)
    model.dhat = 1e-4
    model.tol = 1e-3
    model.epsv = 1e-2
    model.gravity = wp.vec3d([0, 0, -9.81])

    peg = tm.load("assets/objects/peg/peg.stl")
    hole = tm.load("assets/objects/peg/hole.stl")

    peg_handles, hole_handles = [], []
    for env_id in range(args.num_envs):
        peg_handle = model.add_affine_body(
            peg.vertices,
            peg.faces.astype(np.int32),
            1e3,
            1e9,
            0.2,
            mass_xi=0.0205 / len(peg.vertices),
            env_id=env_id,
        )
        hole_handle = model.add_affine_body(
            hole.vertices,
            hole.faces.astype(np.int32),
            1e3,
            1e9,
            0.2,
            0.0239 / len(peg.vertices),
            env_id=env_id,
        )
        peg_handles.append(peg_handle)
        hole_handles.append(hole_handle)

    sensor = pv.read("assets/robots/sensor.vtk")
    stick_idx = [1, 3, 5, 7, 12, 18, 24, 30, 36, 39, 42, 46, 49, 53, 56, 59, 63, 66, 69, 73, 76, 77, 82, 85, 86, 88, 90, 92, 94, 95, 96, 97, 99, 102, 103, 104, 105, 106, 107, 108, 115, 117, 118, 119, 120, 121, 122, 125, 126, 127, 128, 129, 131, 134, 135, 138]  # fmt: skip

    stick_mask = np.zeros(sensor.n_points, dtype=np.int32)
    stick_mask[stick_idx] = 1

    sensor_handles = []
    for env_id in range(args.num_envs):
        sensor_handles.append([model.add_soft_vol_body(sensor, density=1e3, E=1e5, nu=0.4, mu=1.0, env_id=env_id) for _ in range(2)])

    # Setup simulation
    model.init()

    # Set initial states
    [model.enable_affine_kinematic_constraint(hole_handle) for env_id, hole_handle in enumerate(hole_handles)]
    [model.set_affine_state(hole_handle, np.eye(3), np.zeros(3) + env_pos[env_id]) for env_id, hole_handle in enumerate(hole_handles)]
    [model.set_affine_kinematic_target(hole_handle, np.eye(3), np.zeros(3) + env_pos[env_id]) for env_id, hole_handle in enumerate(hole_handles)]
    [model.set_affine_state(peg_handle, np.eye(3), np.array([-0.0, 0.005, 0.021]) + env_pos[env_id]) for env_id, peg_handle in enumerate(peg_handles)]

    for env_id in range(args.num_envs):
        model.set_soft_kinematic_constraint(sensor_handles[env_id][0], stick_mask)
        model.set_soft_kinematic_constraint(sensor_handles[env_id][1], stick_mask)

    model.finalize()

    sensor_points = np.array(sensor.points)

    # Integrator
    integrator = IPCIntegrator()
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.max_cg_iter = 40
    integrator.cg_rel_tol = 1e-5
    integrator.max_newton_iter = 20

    # Start simulation
    sim_time = 0
    for t in tqdm(range(200)):
        T_1, T_2 = np.eye(4), np.eye(4)
        T_1[:3, 3] = sensor_1_p_traj[min(t, len(sensor_1_p_traj) - 1)]
        T_1[:3, :3] = R.from_quat(sensor_1_q_traj[min(t, len(sensor_1_p_traj) - 1)]).as_matrix()

        T_2[:3, 3] = sensor_2_p_traj[min(t, len(sensor_2_p_traj) - 1)]
        T_2[:3, :3] = R.from_quat(sensor_2_q_traj[min(t, len(sensor_2_p_traj) - 1)]).as_matrix()

        s1_target = sensor_points @ T_1[:3, :3].T + T_1[:3, 3]
        s2_target = sensor_points @ T_2[:3, :3].T + T_2[:3, 3]

        for env_id in range(args.num_envs):
            if t == 0:
                model.set_soft_state(sensor_handles[env_id][0], s1_target + env_pos[env_id])
                model.set_soft_state(sensor_handles[env_id][1], s2_target + env_pos[env_id])

            model.set_soft_kinematic_target(sensor_handles[env_id][0], s1_target + env_pos[env_id])
            model.set_soft_kinematic_target(sensor_handles[env_id][1], s2_target + env_pos[env_id])

            model.set_affine_kinematic_target(hole_handles[env_id], np.eye(3), np.zeros(3) + env_pos[env_id])
        integrator.simulate(model, dt=dt)

        sim_time += integrator.profile_helper.current_timestep_data["total_timestep"]
        log.separate()

        if args.export_mesh:
            model.write_scene(f"{OUT_DIR}/frames/peg_in_hole-detro-multi-{t}.ply")

        tqdm.write(f"Curr sim speed {(t + 1) * args.num_envs / sim_time} FPS")
