import os
import os.path as osp
import sys
from enum import Enum

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh as tm
import warp as wp
from scipy.spatial.transform import Rotation as R
from warp.sim.render import SimRendererOpenGL

sys.path.append(".")

import argparse

from taccel.taccel import TaccelModel
from warp_ipc.body_handle import TetMeshBodyHandle
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.joint_handle import WorldJointHandle, WorldJointType
from warp_ipc.robots import Robot
from warp_ipc.utils import log

from .example_utils import get_interp_step, init_robot_demo

# Tac-man Parameters
DELTA0 = 4e-4
ALPHA = 0.6


class TacManState(Enum):
    EXECUTION = 0
    RECOVERY = 1
    TERMINATE = 2


def add_joint(
    model: TaccelModel,
    part_handle: TetMeshBodyHandle,
    joint: WorldJointType,
    edge_verts=None,
) -> WorldJointHandle | None:
    match joint:
        case WorldJointType.PRISMATIC:
            axis_x = 1.0
            axis_y, axis_z = 0.0, 0.0
            joint_axis = np.array([axis_x, axis_y, axis_z])
            joint_axis = joint_axis / np.linalg.norm(joint_axis)

            print(f"Creating prismatic joint towards {joint_axis}")
            joint_handle = model.add_world_prismatic_joint(part_handle, joint_axis)

        case WorldJointType.REVOLUTE:
            axis_i = np.random.randint(0, 3)
            axis_j = (axis_i + int(np.sign(np.random.uniform(-1, 1, 1)).item())) % 3
            axis_position = edge_verts[axis_i]
            joint_axis = edge_verts[axis_j] - edge_verts[axis_i]

            print(f"Creating revolute joint at {axis_position} towards {joint_axis}")
            joint_handle = model.add_world_revolute_joint(part_handle, axis_position, joint_axis)

        case WorldJointType.HELICAL:
            print("Loaded nut and bolt to simulate a helical joint")
            model.enable_affine_kinematic_constraint(part_handle)
            joint_handle = None

        case _:
            raise NotImplementedError()

    return joint_handle


def rigid_transf(P: np.array, Q: np.array):
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    t = centroid_Q - centroid_P
    p = P - centroid_P
    q = Q - centroid_Q
    H = np.dot(p.T, q)
    U, S, Vt = np.linalg.svd(H)
    if np.linalg.det(np.dot(Vt.T, U.T)) < 0.0:
        Vt[-1, :] *= -1.0
    R = np.dot(Vt.T, U.T)
    # rmsd = np.sqrt(np.sum(np.square(np.dot(p, R.T) - q)) / P.shape[0])
    return R, t  # , rmsd.


@torch.no_grad()
def run():
    wp.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--joint_type", default="revolute", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # DEBUG
    _, OUT_DIR = init_robot_demo(args, f"tac_man-{args.joint_type}-{args.seed}")

    joint_type = {
        "prismatic": WorldJointType.PRISMATIC,
        "revolute": WorldJointType.REVOLUTE,
        "helical": WorldJointType.HELICAL,
    }[args.joint_type]

    # ============================== Setup Simulation Model ==============================
    model = TaccelModel(num_envs=1, viz_envs=[])
    model.set_kinematic_stiffness(1e8)
    model.gravity = wp.vec3d([0, 0, 0])
    model.dhat = 5e-4

    if not args.headless:
        stage_path = os.path.join(OUT_DIR, "parallel_test.usd")
        renderer = SimRendererOpenGL(
            model,
            stage_path,
            scaling=1,
            near_plane=1.0,
            far_plane=100.0,
            camera_fov=45.0,
            camera_pos=(0.0, 2.0, 10.0),
            camera_front=(0.0, 0.0, -1.0),
            camera_up=(0.0, 1.0, 0.0),
        )

    if joint_type == WorldJointType.PRISMATIC or joint_type == WorldJointType.REVOLUTE:
        handle_extents = np.array([0.0075, 0.1, 0.01]) * 2
        obj_handle_mesh = tm.primitives.Box(extents=handle_extents).to_mesh()
        obj_handle_mesh.vertices = np.array(obj_handle_mesh.vertices) + np.array([[0.02, 0.0, 0.1]])

        board_extent = np.random.uniform(0.3, 0.3, [2])
        obj_board_mesh = tm.primitives.Box(extents=np.array([0.01, board_extent[0], board_extent[1]])).to_mesh()
        obj_board_mesh.vertices = obj_board_mesh.vertices + np.array([[-0.05, 0.0, 0.1]])

        edge_verts = np.array([
            [-0.055, board_extent[0] / 2, board_extent[1] / 2 + 0.1],
            [-0.055, -board_extent[0] / 2, board_extent[1] / 2 + 0.1],
            [-0.055, -board_extent[0] / 2, -board_extent[1] / 2 + 0.1],
            [-0.055, board_extent[0] / 2, -board_extent[1] / 2 + 0.1],
        ])

        obj_mesh = tm.boolean.union([obj_handle_mesh, obj_board_mesh])
        kine_handle = model.add_affine_body(obj_mesh.vertices, obj_mesh.faces.astype(np.int32), 1e3, 1e9, 0.3)

        gripper_init_pos = np.array([0.12, 0.0, 0.1])
        gripper_init_rot = R.from_euler("xyz", np.array([90, 0, -90]), degrees=True).as_matrix()
        gripper_release_q = 0.04
        gripper_grasp_q = handle_extents[2] / 2 - 8e-4 + model.dhat

        delta_tf = np.eye(4)
        delta_tf[:3, 3] = np.array([0.0, 0.0, -5e-4])

    elif joint_type == WorldJointType.HELICAL:
        nut_mesh = tm.load("assets/objects/bolt_and_nut/m20_nut_wt.stl")

        bolt_mesh = tm.load("assets/objects/bolt_and_nut/m20_bolt_wt.stl")
        bolt_handle = model.add_affine_body(
            bolt_mesh.vertices,
            bolt_mesh.faces.astype(np.int32),
            density=1e3,
            E=1e9,
            mu=1.0,
        )

        nut_handle = model.add_affine_body(nut_mesh.vertices, nut_mesh.faces.astype(np.int32), 1e3, 1e9, 1.0)
        kine_handle = nut_handle

        edge_verts = None

        delta_tf = np.eye(4)
        delta_tf[:3, :3] = R.from_euler("xyz", np.array([0, 0, 5.0]), degrees=True).as_matrix()

        gripper_init_pos = np.array([0.0, 0.0, 0.185])
        gripper_init_rot = R.from_euler("xyz", np.array([0.0, 180.0, 90.0]), degrees=True).as_matrix()

        gripper_grasp_q = 0.012 - model.dhat
        gripper_release_q = 0.04

    urdf_path, _, tac_path = Robot.get_fabr_path("tactile-pandahand", 1e-07)
    model.add_robot(
        urdf_path,
        tac_fab_path=tac_path,
        env_id=0,
        start_coll_layer=2,
        coll_layers=[],
        disable_coll_layers=[1],
    )
    model.add_vbts_to_sim(model.robots[0], coll_layers=1)
    model.init()
    joint_handle = add_joint(model, kine_handle, joint_type, edge_verts)

    hand_tf = np.eye(4)
    hand_tf[:3, 3] = gripper_init_pos
    hand_tf[:3, :3] = gripper_init_rot

    model.finalize()

    # ============================== Setup Physics Integrator ==============================
    integrator = IPCIntegrator()
    integrator.use_hard_kinematic_constraint = False
    integrator.use_cpu = False
    integrator.dt = 1 / 50
    integrator.max_newton_iter = 30
    integrator.use_inversion_free_step_size_filter = True
    integrator.inversion_free_im_tol = 1e-6
    integrator.inversion_free_cubic_coef_tol = 1e-10

    # Initialize states
    init_joint_params = {
        "panda_finger_joint1": gripper_release_q,
        "panda_finger_joint2": gripper_release_q,
    }
    if joint_type == WorldJointType.HELICAL:
        model.set_affine_state(nut_handle, np.eye(3), np.zeros(3))
        model.set_affine_kinematic_target(nut_handle, np.eye(3), np.zeros(3))
        model.set_affine_state(bolt_handle, np.eye(3), np.zeros(3))

    model.set_robot_states([init_joint_params], hand_tf[None])
    model.set_robot_targets([init_joint_params], hand_tf[None])

    # ============================== Grasping the Handle ==============================
    viz_env = 0
    hand_q = {}
    N_STEPS_GRASPS = 15
    for t in range(N_STEPS_GRASPS + 5):
        if t < N_STEPS_GRASPS:
            hand_q = {
                "panda_finger_joint1": get_interp_step(gripper_release_q, gripper_grasp_q, N_STEPS_GRASPS, t),
                "panda_finger_joint2": get_interp_step(gripper_release_q, gripper_grasp_q, N_STEPS_GRASPS, t),
            }
        else:
            hand_q = {
                "panda_finger_joint1": gripper_grasp_q,
                "panda_finger_joint2": gripper_grasp_q,
            }
        model.set_robot_targets([hand_q], hand_tf[None])
        integrator.simulate(model, dt=0.02)

        if not args.headless:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()

        all_rgbs, all_depths, all_normals = model.render_tactile(True, True)
        viz_rgb = all_rgbs[viz_env].reshape([800, 400, 3])
        text = f"Env {viz_env} | {model.frame}"
        viz_rgb = cv2.putText(viz_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        if not args.headless:
            cv2.imshow("TacMan - RGB", viz_rgb[..., [2, 1, 0]])
            cv2.waitKey(1)
        model.write_scene(osp.join(OUT_DIR, "frames", f"frame_{model.frame}.ply"))
        o3d.io.write_point_cloud(
            osp.join(OUT_DIR, "frames", f"frame-tac_{model.frame}_markers_GRASP.ply"),
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(model.tac_markers[0].cpu().numpy())),
        )

        if t == 0:
            canon_markers_local = model.tac_markers_local[0].cpu().numpy()

    # Track activated markers
    grasp_markers_local = model.tac_markers_local[0].cpu().numpy()
    grasp_markers_world = model.tac_markers[0].cpu().numpy()

    robot_tf = model.robots[0].root_transform
    grasp_marker_hand = np.matmul(
        robot_tf[:3, :3].T,
        (grasp_markers_world.reshape(-1, 3) - robot_tf[:3, 3][None]).T,
    ).T

    marker_depth = np.abs(grasp_markers_local[..., -1] - canon_markers_local[..., -1]).reshape(-1)
    tracking_marker_idx = np.where(marker_depth > 8e-4)[0].tolist()
    tracked_marker_diff = 0.0
    assert len(tracking_marker_idx) > 6

    # ============================== Manipulation ==============================
    N_STEPS_MANIP_MAX = int(50 / integrator.dt)  # 50 sec
    curr_state = TacManState.EXECUTION

    hand_tf_exe_dir = np.eye(4)
    hand_tf_exe_dir[1, 3] = -1e-3

    n_rec = 0

    grasp_markers_world = grasp_markers_world.reshape(-1, 3)
    curr_markers_world = model.tac_markers
    for t in range(N_STEPS_MANIP_MAX):
        match curr_state:
            case TacManState.EXECUTION:
                hand_tf = np.matmul(hand_tf, delta_tf)[:, None].squeeze()

                if tracked_marker_diff > DELTA0:
                    curr_state = TacManState.RECOVERY

                n_rec = 0

            case TacManState.RECOVERY:
                marker_dspl_R, marker_dspl_t = rigid_transf(
                    grasp_markers_world.reshape(-1, 3)[tracking_marker_idx],
                    curr_markers_world[tracking_marker_idx],
                )
                assert np.linalg.det(marker_dspl_R) > 0.9999
                marker_dspl = np.eye(4)
                marker_dspl[:3, :3] = marker_dspl_R
                marker_dspl[:3, 3] = marker_dspl_t
                hand_tf = np.matmul(marker_dspl, hand_tf)
                if tracked_marker_diff < DELTA0 * ALPHA:
                    curr_state = TacManState.EXECUTION

                n_rec += 1

            case _:
                break

        if n_rec == 200 or tracked_marker_diff > 4e-3:
            print("Recovery phase execution failed. Stopping simulation.")
            break

        if not joint_type == WorldJointType.HELICAL:
            joint_state = model.get_joint_value(joint_handle)

        model.set_robot_targets([hand_q], hand_tf[None])
        integrator.simulate(model, dt=0.02)

        if not args.headless:
            renderer.begin_frame(model.elapsed_time)
            renderer.render(model.state())
            renderer.end_frame()

        # Update markers
        robot_tf = model.robots[0].root_transform
        grasp_markers_world = np.matmul(robot_tf[:3, :3], grasp_marker_hand.T).T + robot_tf[:3, 3][None]
        curr_markers_world = model.tac_markers[0].cpu().numpy().reshape(-1, 3)
        marker_diff = np.linalg.norm(grasp_markers_world - curr_markers_world, axis=-1)
        tracked_marker_diff = np.mean(marker_diff[tracking_marker_idx])

        model.write_scene(osp.join(OUT_DIR, "frames", f"frame_{model.frame}.ply"))
        o3d.io.write_point_cloud(
            osp.join(
                OUT_DIR,
                "frames",
                f"frame-tac_{model.frame}_tracked_markers_{curr_state.name}_err={tracked_marker_diff:.4f}.ply",
            ),
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(curr_markers_world[tracking_marker_idx])),
        )
        o3d.io.write_point_cloud(
            osp.join(
                OUT_DIR,
                "frames",
                f"frame-tac_{model.frame}_reference_markers_{curr_state.name}_err={tracked_marker_diff:.4f}.ply",
            ),
            o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(grasp_markers_world[tracking_marker_idx])),
        )

        all_rgbs, all_depths, all_normals = model.render_tactile(True, True)
        viz_rgb = all_rgbs[viz_env].reshape([800, 400, 3])
        viz_d = all_depths[viz_env].reshape([800, 400])
        if joint_type == WorldJointType.HELICAL:
            text = f"Env {viz_env} | {model.frame} ({curr_state.name}) | Diff: {tracked_marker_diff * 1000:.2f} mm"
        else:
            text = f"Env {viz_env} | {model.frame} ({curr_state.name}) | Diff: {tracked_marker_diff * 1000:.2f} mm | Joint: {joint_state:.4f}"
        viz_rgb = cv2.putText(viz_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        if not args.headless:
            cv2.imshow("TacMan - RGB", viz_rgb[..., [2, 1, 0]])
            cv2.imshow("TacMan - Depth", viz_d)
            cv2.waitKey(1)

        log.info(f"Current state: {curr_state.name} | Tracking {len(tracking_marker_idx)} markers")
        log.info(f"Max marker diff: {tracked_marker_diff * 1000} mm")
        log.separate()


if __name__ == "__main__":
    run()
    cv2.destroyAllWindows()
