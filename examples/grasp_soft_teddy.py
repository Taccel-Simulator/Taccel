import os

import json
import os.path as osp
from argparse import ArgumentParser

import numpy as np
import pyvista as pv
import warp as wp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from warp.sim.render import SimRendererOpenGL

from taccel import TaccelModel, TactileRobot

import warp_ipc.utils.profile as abd_profile
from examples.example_utils import init_robot_demo
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.utils.constants import VolMaterialType
from warp_ipc.utils.log import LoggingLevel, set_logging_level


if __name__ == "__main__":
    wp.init()
    set_logging_level(LoggingLevel.INFO)

    abd_profile.WATCH_GPU_MEM_USAGE_ENABLED = True

    parser = ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    _, OUT_DIR = init_robot_demo(args, "soft_teddy")

    dt = 1 / 50

    model = TaccelModel(num_envs=args.num_envs, viz_envs=list(range(args.num_envs)) if args.viz else [])
    model.dhat = 5e-3
    model.kappa = 3e6
    model.k_elasticity_damping = 0

    stage_path = os.path.join(OUT_DIR, "teddy.usd")

    urdf_path, _, tac_path = TactileRobot.get_fabr_path("tactile-robotiq3f")
    for i_env in range(args.num_envs):
        model.add_robot(
            urdf_path,
            tac_fab_path=tac_path,
            env_id=i_env,
            start_coll_layer=2,
            coll_layers=[0],
            disable_coll_layers=[1],
        )

    for i_env, robot in enumerate(model.robots):
        model.add_vbts_to_sim(robot, coll_layers=1)

    env_translation = np.asarray([1.0, 0.0, 0.0])

    teddy_mesh = pv.read("assets/objects/teddy/teddy.vtk")
    teddy_handles = []
    for env_id in tqdm(range(args.num_envs)):
        mesh = teddy_mesh.copy()
        mesh.points[:, 1] += 0.005
        mesh.points[:] += env_translation * (env_id + 1)
        handle = model.add_soft_vol_body(mesh, density=1e3, E=3e4, nu=0.4, mu=0.3, env_id=env_id, coll_layer=0)
        teddy_handles.append(handle)

    model.add_plane(np.array([0, 1, 0], dtype=np.float64), np.zeros((3,), dtype=np.float64), 0.3)

    # Added all the bodies
    model.init()

    if args.viz:
        renderer = SimRendererOpenGL(
            model,
            stage_path,
            scaling=1,
            near_plane=0.001,
            far_plane=20.0,
            camera_fov=75.0,
            camera_pos=(2.0, 1.0, 2.0),
            camera_front=(0.0, -1.0, -2.0),
            camera_up=(0.0, 1.0, 0.0),
        )

    integrator = IPCIntegrator()
    integrator.use_cpu = False
    integrator.max_newton_iter = 30
    integrator.max_cg_iter = 300
    integrator.cg_rel_tol = 1e-3
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = integrator.inversion_free_im_tol = 1e-6
    integrator.inversion_free_cubic_coef_tol = integrator.inversion_free_cubic_coef_tol = 1e-10
    integrator.use_hard_kinematic_constraint = False
    integrator.soft_vol_material_type = VolMaterialType.NEO_HOOKEAN
    model.kinematic_helper.set_initial_stiffness(1e8)

    joint_params = {
        "finger_1_joint_1": 0.0,
        "finger_1_joint_2": 0.0,
        "finger_1_joint_3": 0.0,
        "finger_2_joint_1": 0.0,
        "finger_2_joint_2": 0.0,
        "finger_2_joint_3": 0.0,
        "finger_middle_joint_1": 0.0,
        "finger_middle_joint_2": 0.0,
        "finger_middle_joint_3": 0.0,
        "palm_finger_1_joint": 0.2,
        "palm_finger_2_joint": 0.2,
    }

    env_origin = np.asarray([0.0, 0.0, 0.0])

    hand_tf = np.eye(4)[None].repeat(args.num_envs, axis=0)
    hand_tf[:, :3, :3] = R.from_euler("z", np.pi).as_matrix()
    hand_tf[:, :3, 3] = np.asarray([0.0, 0.22, 0.0])
    hand_tf[:, :3, 3] += env_translation[:, None].T * (np.arange(args.num_envs)[:, None] + 1)

    model.set_robot_states([joint_params] * args.num_envs, hand_tf)
    # model.set_robot_targets(joint_params, hand_tf)
    model.finalize()

    for frame in range(60):
        joint_params = {
            "finger_1_joint_1": np.pi / 6 * 0.01 * frame,
            "finger_1_joint_2": np.pi / 6 * 0.01 * frame,
            "finger_1_joint_3": np.pi / 6 * 0.01 * frame,
            "finger_2_joint_1": np.pi / 6 * 0.01 * frame,
            "finger_2_joint_2": np.pi / 6 * 0.01 * frame,
            "finger_2_joint_3": np.pi / 6 * 0.01 * frame,
            "finger_middle_joint_1": np.pi / 6 * 0.01 * frame,
            "finger_middle_joint_2": np.pi / 6 * 0.01 * frame,
            "finger_middle_joint_3": np.pi / 6 * 0.01 * frame,
            "palm_finger_1_joint": 0.2,
            "palm_finger_2_joint": 0.2,
        }

        model.set_robot_targets([joint_params] * args.num_envs, hand_tf)
        integrator.simulate(model, dt=dt, control=None)

        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()

        model.write_scene(osp.join(OUT_DIR, f"frame_{frame}.ply"))

    for frame in range(60, 80):
        hand_tf[:, 1, 3] += 0.0025

        model.set_robot_targets([joint_params] * args.num_envs, hand_tf)
        integrator.simulate(model, dt=dt, control=None)

        # Render
        if args.viz:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()

        model.write_scene(osp.join(OUT_DIR, f"frame_{frame}.ply"))

    profile_data = integrator.profile_helper.full_json
    with open(osp.join(OUT_DIR, f"profile_{args.num_envs}.json"), "w") as f:
        json.dump(profile_data, f, indent=4)

    print("Task finished. The visualizer will remain open until you close it.")
    if args.viz:
        while True:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()
