import os
import os.path as osp
import sys
from functools import cached_property
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh as tm
import warp as wp
from numpy.typing import NDArray

from warp_ipc.body_handle import TetMeshBodyHandle
from warp_ipc.sim_model import ASRModel
import warp_ipc.utils.log as log
from .utils.mesh_utils import (
    pv_to_tm,
    trimesh_select_by_index,
    trimesh_select_by_index_old,
)
from .vbts import VBTSMLP, depth_to_normal, raycast_depth_kernel
from .tactile_robot import TactileRobot


class TaccelModel(ASRModel):
    robots: list[TactileRobot] = []
    vbts_depths: wp.array3d(dtype=wp.float32)
    vbts_normals: wp.array3d(dtype=wp.vec3)

    @property
    def dummy_robot(self) -> TactileRobot | None:
        return self.robots[0] if len(self.robots) > 0 else None

    def _register_rest_markers(self) -> None:
        assert self.frame == 0, "Only call before the first advance for acquring local data"
        self.rest_markers, self.rest_coats = [], []
        for i_env in range(self.num_envs):
            self.rest_markers.append([])
            self.rest_coats.append([])
            for i_vbts, vbts_cfg in enumerate(self.robots[i_env].vbts_cfgs):
                v = self.get_element_by_handle(vbts_cfg.gel_body_handle, False)[0].float()
                rest_markers = (v[vbts_cfg.body_marker_bc_verts] * vbts_cfg.marker_bc_coords.unsqueeze(-1)).sum(dim=-2)
                self.rest_markers[i_env].append(rest_markers)

                elastomer = self.get_soft_body_from_handle(vbts_cfg.gel_body_handle)
                coat = trimesh_select_by_index_old(
                    pv_to_tm(elastomer.extract_surface()),
                    np.where(vbts_cfg.surf_coat_mask)[0],
                )
                self.rest_coats[i_env].append(coat)

    # TODO: Support multiple types of integration renderer in one sim
    def _load_vbts_render_model(self) -> None:
        self.vbts_mlps, self.ref_imgs = [], []
        self.coords = []
        for vbts_cfg in self.dummy_robot.vbts_cfgs:
            vbts_mlp = VBTSMLP(in_len=9, out_len=3, n_pe=2)
            vbts_mlp.load_state_dict(torch.load(vbts_cfg.mlp_ckpt_path))
            vbts_mlp = vbts_mlp.cuda()
            vbts_mlp.eval()

            ref_img = cv2.cvtColor(cv2.imread(vbts_cfg.ref_img_path), cv2.COLOR_BGR2RGB).astype(float) / 255
            img_h, img_w, _ = ref_img.shape
            ref_img = torch.from_numpy(ref_img).cuda().unsqueeze(0).tile([self.num_envs, 1, 1, 1])
            coords = np.stack(np.meshgrid(np.arange(img_h), np.arange(img_w)), axis=-1).transpose([1, 0, 2])  # N_env x N_tac x H x W x 2
            coords = torch.from_numpy(coords).float().reshape([1, -1, 2]).cuda().tile([self.num_envs, 1, 1])

            self.coords.append(coords)
            self.ref_imgs.append(ref_img)
            self.vbts_mlps.append(vbts_mlp)

    def _prepare_vbts_buffers(self):
        if len(self.dummy_robot.vbts_cfgs) == 0:
            return
        self.vbts_depths = wp.array3d(
            shape=(
                self.num_envs,
                self.dummy_robot.num_tac,
                self.dummy_robot.vbts_cfgs[0].cam_h * self.dummy_robot.vbts_cfgs[0].cam_w,
            ),
            dtype=wp.float32,
        )
        self.vbts_normals = wp.array3d(
            shape=(
                self.num_envs,
                self.dummy_robot.num_tac,
                self.dummy_robot.vbts_cfgs[0].cam_h * self.dummy_robot.vbts_cfgs[0].cam_w,
            ),
            dtype=wp.vec3,
        )
        self.gel_coat_meshes_wp = [
            wp.Mesh(
                wp.array(
                    [[0.0, 0.0, 0.0]] * vbts_cfg.body_coat_mask.sum().item(),
                    dtype=wp.vec3,
                ),
                wp.array([0, 1, 2], dtype=wp.int32),
            )
            for i_env in range(self.num_envs)
            for vbts_cfg in self.dummy_robot.vbts_cfgs
        ]

    def finalize(self) -> None:
        super().finalize()
        self._register_rest_markers()
        self._load_vbts_render_model()
        self._prepare_vbts_buffers()

    def set_robot_states(
        self,
        joint_params: list[dict[str, float]] | NDArray | torch.Tensor,
        root_tf: NDArray | torch.Tensor | None = None,
        env_ids: list[int] = [],
    ):
        if isinstance(joint_params, torch.Tensor):
            joint_params = joint_params.cpu().numpy()
        if isinstance(joint_params, np.ndarray):
            joint_params = [dict(zip(self.dummy_robot.joint_names, jp.tolist())) for jp in joint_params]
        if isinstance(root_tf, torch.Tensor):
            root_tf = root_tf.cpu().numpy()

        env_ids = env_ids if len(env_ids) > 0 else range(len(self.robots))
        for i, env_id in enumerate(env_ids):
            self.robots[env_id]._urdf.update_cfg(joint_params[i])
            if root_tf is not None:
                self.robots[env_id]._root_tf = root_tf[i]
            for link in self.robots[env_id].links:
                if link.handle is None:
                    continue
                tf = self.robots[env_id]._root_tf @ link.urdf_local_transform
                self.set_affine_state(link.handle, tf[:3, :3], tf[:3, 3])

            for i_sensor, sensor_cfg in enumerate(self.robots[env_id].vbts_cfgs):
                finger_link = self.robots[env_id].links_map[sensor_cfg.link_name]
                rel_pose = sensor_cfg.attch_rel_tf
                pose = finger_link.urdf_global_transform @ rel_pose
                points = sensor_cfg.gel_mesh.points @ pose[:3, :3].T + pose[:3, 3]
                self.set_soft_state(sensor_cfg.gel_body_handle, points)

    def set_robot_targets(
        self,
        joint_params: list[dict[str, float]] | NDArray | torch.Tensor,
        root_tf: NDArray | torch.Tensor | None = None,
        env_ids: list[int] = [],
    ):
        if isinstance(joint_params, torch.Tensor):
            joint_params = joint_params.cpu().numpy()
        if isinstance(joint_params, np.ndarray):
            joint_params = [dict(zip(self.dummy_robot.joint_names, jp)) for jp in joint_params]
        if isinstance(root_tf, torch.Tensor):
            root_tf = root_tf.cpu().numpy()

        env_ids = env_ids if len(env_ids) > 0 else range(len(self.robots))
        for i, env_id in enumerate(env_ids):
            self.robots[env_id]._urdf.update_cfg(joint_params[i])
            if root_tf is not None:
                self.robots[env_id]._root_tf = root_tf[i]
            for link in self.robots[env_id].links:
                if link.handle is None:
                    continue
                tf = self.robots[env_id]._root_tf @ link.urdf_local_transform
                self.set_affine_kinematic_target(link.handle, tf[:3, :3], tf[:3, 3])

            for i_sensor, sensor_cfg in enumerate(self.robots[env_id].vbts_cfgs):
                finger_link = self.robots[env_id].links_map[sensor_cfg.link_name]
                rel_pose = sensor_cfg.attch_rel_tf
                pose = finger_link.urdf_global_transform @ rel_pose
                points = sensor_cfg.gel_mesh.points @ pose[:3, :3].T + pose[:3, 3]
                self.set_soft_kinematic_target(sensor_cfg.gel_body_handle, points)

    def add_robot(
        self,
        urdf_path: str,
        tac_fab_path: str,
        env_id: int = 0,
        start_coll_layer: int = 2,
        coll_layers: list[int] = [],
        disable_coll_layers: list[int] = [],
        mu: float = 0.3,
        no_bodies: bool = False,
        self_collision: bool = False,
        use_collision_mesh: bool = False,
    ):
        robot = TactileRobot(urdf_path, env_id=env_id, tac_fab_path=tac_fab_path, use_collision=use_collision_mesh)

        for i_link, link in enumerate(robot.links):
            if link.rest_mesh == [] or len(link.rest_mesh.vertices) == 0:
                continue
            if not no_bodies:
                link.handle = self.add_affine_body(
                    np.asarray(link.rest_mesh.vertices, dtype=np.float64),
                    np.asarray(link.rest_mesh.faces, dtype=np.int32),
                    density=1e3,
                    E=1e6,
                    mu=mu,
                    mass_xi=0.01,
                    env_id=env_id,
                )

        self.robots.append(robot)
        self.robot_link_added = True

        rob_coll_layer = coll_layers
        rob_disable_coll_layer = disable_coll_layers

        def apply_robot_constraints(
            target_robot=robot,
            coll_layers=rob_coll_layer,
            disable_coll_layers=rob_disable_coll_layer,
        ):
            log.debug(f"Set robot at env {target_robot.env_id}: coll layer {rob_coll_layer}, ignore coll {rob_disable_coll_layer}")
            curr_layer = start_coll_layer
            for i_link, link in enumerate(target_robot.links):
                if link.handle is None:
                    continue
                link.collision_layer = curr_layer
                self.enable_affine_kinematic_constraint(link.handle)
                self.set_body_collision_layer(link.handle, link.collision_layer)

                for coll_layer in coll_layers:
                    self.set_collision_layer_filter(curr_layer, coll_layer, True)

                for coll_layer in disable_coll_layers:
                    self.set_collision_layer_filter(curr_layer, coll_layer, False)

                curr_layer += 1

            for i_layer in range(start_coll_layer, curr_layer):
                for j_layer in range(i_layer, curr_layer):
                    if i_layer == j_layer:
                        continue
                    self.set_collision_layer_filter(i_layer, j_layer, self_collision)

        self.constraint_fns.append((f"robot_constraints_{env_id}", apply_robot_constraints))

        return robot

    def add_vbts_to_sim(self, robot: TactileRobot, coll_layers: int | List[int] = 1) -> list[TetMeshBodyHandle]:
        """Add the VBTS of a given robot to the simulation, and set the colliding layers.

        Args:
            robot (TactileRobot): The robot to add VBTS to.
            coll_layers (int | List[int], optional): The colliding layer of each gel pad, int or list[int]. If int given, all the gel pads will be in the same colliding layer. Defaults to 1.

        Returns:
            list[TetMeshBodyHandle]: _description_
        """
        handles = []
        coll_layers = [coll_layers] * robot.num_tac if isinstance(coll_layers, int) else list(coll_layers)

        for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs):
            robot.tac_coll_layer = coll_layers
            handle = self.add_soft_vol_body(
                vbts_cfg.gel_mesh,
                vbts_cfg.density,
                vbts_cfg.E,
                vbts_cfg.nu,
                vbts_cfg.mu,
                robot.env_id,
                coll_layers[i_vbts],
                vbts_cfg.self_coll,
            )
            vbts_cfg.coll_layer = coll_layers[i_vbts]
            vbts_cfg.gel_body_handle = handle
            handles.append(handle)

        def apply_tac_constraints(target_robot=robot):
            log.debug(f"Applying VBTS constraints for robot at env {target_robot.env_id}")
            for vbts_cfg in robot.vbts_cfgs:
                self.set_soft_kinematic_constraint(vbts_cfg.gel_body_handle, vbts_cfg.body_attach_mask)
                for link in target_robot.links:
                    self.set_collision_layer_filter(
                        link.collision_layer,
                        vbts_cfg.coll_layer,
                        vbts_cfg.collide_with_robot,
                    )
                self.set_collision_layer_filter(0, vbts_cfg.coll_layer, True)

        self.constraint_fns.append((f"tac_constraints_{robot.env_id}", apply_tac_constraints))

        return handles

    @property
    def tac_tf(self) -> torch.Tensor:
        """Accquire the current state of the gel pads"

        Returns:
            torch.Tensor: gel pad frame transformation, in shape [num_envs, num_tac, 3, 3]
        """
        return torch.from_numpy(
            np.stack(
                [
                    np.stack(
                        [
                            (robot.links_map[vbts_cfg.link_name].urdf_global_transform @ vbts_cfg.attch_rel_tf)
                            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs)
                        ],
                        axis=0,
                    )
                    for i_env, robot in enumerate(self.robots)
                ],
                axis=0,
            ),
        ).to(self.device)

    @property
    def tac_tf_robot_local(self) -> torch.Tensor:
        """Accquire the current state of the gel pads"

        Returns:
            torch.Tensor: gel pad frame transformation, in shape [num_envs, num_tac, 3, 3]
        """
        return torch.from_numpy(
            np.stack(
                [
                    np.stack(
                        [
                            (robot.links_map[vbts_cfg.link_name].urdf_local_transform @ vbts_cfg.attch_rel_tf)
                            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs)
                        ],
                        axis=0,
                    )
                    for i_env, robot in enumerate(self.robots)
                ],
                axis=0,
            ),
        ).to(self.device)

    @property
    def tac_coat_rest_and_current(self) -> tuple[list[tm.Trimesh], list[tm.Trimesh]]:
        """Accquire the current and rest states of the gel pads.

        Returns:
            tuple[list[tm.Trimesh], list[tm.Trimesh]]: coat_current and coat_rest, each is a list of trimesh.Trimesh with length of num_envs
        """
        coat_rest, coat_current = [], []
        for i_env, robot in enumerate(self.robots):
            # root_state = torch.from_numpy(self.robots[i_env].root_state_matrix).to(self._device).float()
            root_state = torch.from_numpy(self.robots[i_env].root_transform).to(self.device).float()
            coat_rest.append([])
            coat_current.append([])
            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs):
                elastomer = self.sim.get_soft_body_from_handle(vbts_cfg.gel_body_handle)
                coat = trimesh_select_by_index_old(
                    pv_to_tm(elastomer.extract_surface()),
                    np.where(vbts_cfg.body_attach_mask)[0],
                )
                coat.vertices = (
                    torch.matmul(
                        torch.from_numpy(coat.vertices).to(self._device).float() - root_state[:3, 3],
                        root_state[:3, :3],
                    )
                    .cpu()
                    .numpy()
                )
                coat_current[-1].append(coat)

                v = np.asarray(vbts_cfg.gel_mesh.vertices, dtype=np.float32)
                pose = robot._urdf._link_map[vbts_cfg.link_name].urdf_local_transform @ vbts_cfg.attch_rel_tf
                v = v @ pose[:3, :3].T + pose[:3, 3]
                coat_rest[-1].append(tm.Trimesh(v, vbts_cfg.gel_mesh.faces))

        return coat_current, coat_rest

    @property
    def tac_markers(self) -> torch.Tensor:
        """Accquire the current markers of the gel pads.

        Returns:
            torch.Tensor markers, in shape [num_envs, num_markers_all, 3]
        """
        markers = []
        for i_env, robot in enumerate(self.robots):
            markers.append([])
            for i, vbts_cfg in enumerate(robot.vbts_cfgs):
                v = self.get_element_by_handle(vbts_cfg.gel_body_handle, False)[0].float()
                tac_markers = (v[vbts_cfg.body_marker_bc_verts] * vbts_cfg.marker_bc_coords.unsqueeze(-1)).sum(dim=-2)
                markers[-1].append(tac_markers)

            markers[-1] = torch.concat(markers[-1], dim=0)
        markers = torch.stack(markers, dim=0)

        return markers

    @property
    def tac_markers_rest_and_current_hand_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Accquire the current markers and their rest states (for reference) of the gel pads.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: markers and rest markers, each in shape [num_envs, num_markers_all, 3]
        """
        markers, rest_markers = [], []

        tac_tf = self.tac_tf_robot_local.float()

        for i_env, robot in enumerate(self.robots):
            markers.append([])
            rest_markers.append([])

            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs):
                v = self.get_element_by_handle(vbts_cfg.gel_body_handle, False)[0].float()
                tac_markers = (v[vbts_cfg.body_marker_bc_verts] * vbts_cfg.marker_bc_coords.unsqueeze(-1)).sum(dim=-2)
                # tac_markers = torch.matmul(self.robots[i_env].root_transform[:3, :3].T, (tac_markers - self.robots[i_env].root_transform[:3, 3]).T).T
                markers[-1].append(tac_markers.float())

                tac_markers_rest = self.rest_markers[i_env][i_vbts].float()
                tac_markers_rest = torch.matmul(tac_tf[i_env, i_vbts, :3, :3], tac_markers_rest.T).T + tac_tf[i_env, i_vbts, :3, 3]
                rest_markers[-1].append(tac_markers_rest)

            markers[-1] = torch.concat(markers[-1], dim=0)
            rest_markers[-1] = torch.concat(rest_markers[-1], dim=0)

        markers, rest_markers = torch.stack(markers, dim=0), torch.stack(rest_markers, dim=0)  # Each in B x N x 3
        # World to hand local
        robot_poses = torch.stack([torch.from_numpy(robot.root_transform).to(self.device) for robot in self.robots])
        markers = torch.matmul(robot_poses[:, :3, :3].transpose(-1, -2), (markers - robot_poses[:, :3, 3].unsqueeze(-2)).transpose(-1, -2)).transpose(
            -1, -2
        )
        return markers, rest_markers

    def get_tac_markers_and_rest(self, local: bool = False, return_numpy: bool = True):
        markers, rest_markers = [], []

        tac_tf = self.tac_tf.float()
        for i_env, robot in enumerate(self.robots):
            markers.append([])
            rest_markers.append([])

            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs):
                v = self.get_element_by_handle(vbts_cfg.gel_body_handle, False)[0].float()
                tac_markers = (v[vbts_cfg.body_marker_bc_verts] * vbts_cfg.marker_bc_coords.unsqueeze(-1)).sum(dim=-2).float()
                tac_rest_markers = self.rest_markers[i_env][i_vbts].float()

                tac_state = tac_tf[i_env, i_vbts]
                if local:
                    tac_markers = torch.matmul(tac_markers - tac_state[:3, 3].unsqueeze(-2), tac_state[:3, :3])
                else:
                    tac_rest_markers = torch.matmul(tac_rest_markers, tac_state[:3, :3].transpose(-1, -2)) + tac_state[:3, 3].unsqueeze(-2)

                markers[-1].append(tac_markers.cpu().numpy() if return_numpy else tac_markers)
                rest_markers[-1].append(tac_rest_markers.cpu().numpy() if return_numpy else tac_markers)

        return markers, rest_markers

    @property
    def tac_markers_local(self) -> torch.Tensor:
        markers = []
        tac_tfs = self.tac_tf
        for i_env, robot in enumerate(self.robots):
            markers.append([])
            for i_vbts, vbts_cfg in enumerate(robot.vbts_cfgs):
                v = self.get_element_by_handle(vbts_cfg.gel_body_handle, False)[0].float()
                tac_markers = (v[vbts_cfg.body_marker_bc_verts] * vbts_cfg.marker_bc_coords.unsqueeze(-1)).sum(dim=-2)
                tac_markers = torch.matmul(
                    tac_tfs[i_env, i_vbts, :3, :3].T,
                    (tac_markers - tac_tfs[i_env, i_vbts, :3, 3]).T,
                ).T
                markers[-1].append(tac_markers)
            markers[-1] = torch.concat(markers[-1], dim=0)

        markers = torch.stack(markers, dim=0)
        return markers

    @cached_property
    def _all_gel_body_handles(self) -> list[int]:
        return [vbts_cfg.gel_body_handle for robot in self.robots for vbts_cfg in robot.vbts_cfgs]

    @cached_property
    def gel_coat_mesh_indexers_wp(self):
        point_idxs, indices = [], []
        n_pts = 0
        for robot in self.robots:
            for vbts_cfg in robot.vbts_cfgs:
                coat_mesh = self.get_soft_body_from_handle(vbts_cfg.gel_body_handle).extract_surface()
                surface_mesh = pv_to_tm(coat_mesh)
                orig_idxs = np.array(coat_mesh.point_data["vtkOriginalPointIds"])

                tactile_mesh_coat, coat_orig_idxs = trimesh_select_by_index(surface_mesh, np.where(vbts_cfg.surf_coat_mask)[0])

                point_idxs.append(orig_idxs[coat_orig_idxs])
                indices.append(wp.array(tactile_mesh_coat.faces.flatten(), dtype=int, device=self.device))
                n_pts += len(tactile_mesh_coat.vertices)

        return point_idxs, indices

    def update_gel_coat_meshes_wp(self):
        point_idxs, indices = self.gel_coat_mesh_indexers_wp
        all_points = [self.get_soft_verts_from_handle(handle)[point_idx] for handle, point_idx in zip(self._all_gel_body_handles, point_idxs)]
        for i_tac, (points, indices) in enumerate(zip(all_points, indices)):
            # self.gel_coat_meshes_wp[i_tac].points = wp.array(points, dtype=wp.vec3)
            # self.gel_coat_meshes_wp[i_tac].indices = wp.array(indices, dtype=wp.int32)
            self.gel_coat_meshes_wp[i_tac] = wp.Mesh(wp.array(points, dtype=wp.vec3), wp.array(indices, dtype=wp.int32))

    @cached_property
    def sensor_xy_queries_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        xy_origins, xy_dirs = [], []
        for i_vbts, vbts_cfg in enumerate(self.dummy_robot.vbts_cfgs):
            assert vbts_cfg.ray_axis.lower() == "z", "Only support ray casting along the z-axis for now"
            xy_origin = torch.stack(
                torch.meshgrid(
                    torch.arange(
                        vbts_cfg.cam_reso[0] * vbts_cfg.reso_ds_ratio,
                        device=self.device,
                    ),
                    torch.arange(
                        vbts_cfg.cam_reso[1] * vbts_cfg.reso_ds_ratio,
                        device=self.device,
                    ),
                    indexing="ij",
                ),
                dim=-1,
            )
            xy_origin = xy_origin.reshape([-1, 2]) * 1e-3 * vbts_cfg.cam_pixel_size / vbts_cfg.reso_ds_ratio
            xy_origin = xy_origin - xy_origin.mean(dim=0)
            xy_origin = F.pad(xy_origin, [0, 1], value=vbts_cfg.ray_start_level)

            xy_origins.append(xy_origin.unsqueeze(0).tile([self.num_envs, 1, 1]))
            xy_dirs.append(
                torch.from_numpy(vbts_cfg.cam_fwd_dir)
                .to(torch.float32)
                .to(self.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .tile([self.num_envs, xy_origin.shape[0], 1])
            )

        origins = torch.stack(xy_origins, dim=1)
        dirs = torch.stack(xy_dirs, dim=1)
        # [num_envs*num_tac, H*W, 2] and [num_envs, num_tac, H*W, 3]
        return origins, dirs

    def _draw_marker_on_rgb(self, rgb: NDArray, markers: NDArray, flows: NDArray = None):
        for i_marker, marker in enumerate(markers):
            rgb = cv2.circle(rgb.copy(), marker.astype(int), 5, (0.0, 0.0, 0.0), -1)
            if flows is not None:
                rgb = cv2.arrowedLine(
                    rgb,
                    (marker - flows[i_marker]).astype(int),  # Rest position
                    (marker + 4 * flows[i_marker]).astype(int),  # 4x flow
                    color=(255, 0, 255),
                    thickness=4,
                    line_type=cv2.LINE_AA,
                    tipLength=0.3,
                )
        return rgb

    # TODO: Use the ray_start_level for each VBTS from the Config
    @torch.no_grad()
    def render_tactile(self, render_marker: bool = False, render_flow: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.update_gel_coat_meshes_wp()
        ray_origins, ray_dirs = self.sensor_xy_queries_local
        tac_tfs = self.tac_tf.to(torch.float32)
        ray_origins = torch.matmul(ray_origins, tac_tfs[..., :3, :3].transpose(-1, -2)) + tac_tfs[..., :3, 3].unsqueeze(-2)
        ray_dirs = torch.matmul(ray_dirs, tac_tfs[..., :3, :3].transpose(-1, -2))
        markers, markers_rest, markers_flow = [], [], []

        for env_id, robot in enumerate(self.robots):
            for i_tac, vbts_cfg in enumerate(robot.vbts_cfgs):
                ray_origins_wp = wp.array(ray_origins[env_id, i_tac], dtype=wp.vec3)
                ray_dirs_wp = wp.array(ray_dirs[env_id, i_tac], dtype=wp.vec3)
                mesh_id = self.gel_coat_meshes_wp[env_id * robot.num_tac + i_tac].id
                wp.launch(
                    kernel=raycast_depth_kernel,
                    dim=vbts_cfg.cam_h * vbts_cfg.cam_w,
                    inputs=[
                        mesh_id,
                        ray_origins_wp,
                        ray_dirs_wp,
                        self.vbts_depths[env_id][i_tac],
                        self.vbts_normals[env_id][i_tac],
                    ],
                )

        all_rgbs = []
        all_depths = wp.to_torch(self.vbts_depths)
        for i_tac, vbts_cfg in enumerate(self.dummy_robot.vbts_cfgs):
            all_depths[:, i_tac] = vbts_cfg.ray_dist_to_depth_offset - all_depths[:, i_tac]

        all_depths = all_depths.reshape(self.num_envs, self.dummy_robot.num_tac, *self.ref_imgs[i_tac].shape[1:3]).transpose(-1, -2).contiguous()
        all_depths = torch.clamp(all_depths, 0.0, 3e-3) * 1e3
        all_normals = depth_to_normal(all_depths)
        # all_normals = wp.to_torch(self.vbts_normals)
        for i_tac, (vbts_mlp, vbts_cfg) in enumerate(zip(self.vbts_mlps, self.dummy_robot.vbts_cfgs)):
            d_rgb = vbts_mlp(
                all_depths[:, i_tac].view([self.num_envs, -1]),
                all_normals[:, i_tac].view([self.num_envs, -1, 3]),
                self.coords[i_tac],
            ).reshape(self.ref_imgs[i_tac].shape)
            rgb = (self.ref_imgs[i_tac] + d_rgb).clone().cpu().numpy()  # [..., :: vbts_cfg.reso_ds_ratio, :: vbts_cfg.reso_ds_ratio, :]

            if render_marker:
                markers, markers_rest, markers_flow = self.get_marker_flow_2d()  # list: n_envs, n_tac, (n_markers x 2)
                for i_env in range(self.num_envs):
                    rgb[i_env] = self._draw_marker_on_rgb(
                        rgb[i_env],
                        markers[i_env][i_tac],
                        markers_flow[i_env][i_tac] if render_flow else None,
                    )

            all_rgbs.append(rgb)
        all_rgbs = np.stack(all_rgbs, axis=1)
        all_depths, all_normals = all_depths.cpu().numpy(), all_normals.cpu().numpy()

        return (
            all_rgbs,
            all_depths,
            all_normals,
        )  # , markers, markers_rest, markers_flow

    @torch.no_grad()
    def get_marker_flow_2d(self):
        """Transform the markers, rest markers, and their flows back to 2D space on the pixel coordinate."""
        markers, markers_rest, markers_flow = [], [], []

        markers_local, markers_local_rest = self.get_tac_markers_and_rest(local=True, return_numpy=True)
        for i_env in range(self.num_envs):
            markers.append([])
            markers_rest.append([])
            markers_flow.append([])
            for i_vbts, vbts_cfg in enumerate(self.dummy_robot.vbts_cfgs):
                marker_shift = np.array([[-5, 0]] + np.array(self.ref_imgs[i_vbts][i_env].shape)[:2] / 2)
                marker = markers_local[i_env][i_vbts][..., :2] * 1e3 / vbts_cfg.cam_pixel_size + marker_shift
                marker_rest = markers_local_rest[i_env][i_vbts][..., :2] * 1e3 / vbts_cfg.cam_pixel_size + marker_shift

                markers[-1].append(marker)
                markers_rest[-1].append(marker_rest)
                markers_flow[-1].append(marker - marker_rest)

        return markers, markers_rest, markers_flow
