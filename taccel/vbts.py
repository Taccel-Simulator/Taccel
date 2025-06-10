import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
import trimesh as tm
import warp as wp
from scipy.spatial.transform import Rotation as R

from warp_ipc.body_handle import TetMeshBodyHandle, TriMeshBodyHandle


@wp.kernel
def raycast_depth_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    result_depths: wp.array(dtype=wp.float32),
    result_normals: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    max_dist = 0.03  # max raycast distance

    query = wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f)
    if query:
        result_depths[tid] = t
        result_normals[tid] = n


@dataclass
class VBTSConfig:
    """Vision-Based Tactile Sensor configuration.

    Attributes:
        sensor_name: Name of the sensor.
        link_name: Name of the link.
        attch_rel_pose: Relative pose of the sensor w.r.t. the link.
        gel_mesh_path: Path to the gel mesh.
        gel_resolution: Resolution of the gel.
        shell_mesh_path: Path to the shell mesh.
        body_coat_mask: Body coat mask.
        surf_coat_mask: Surface coat mask.
        body_attach_mask: Body attach mask.
        with_marker: Whether to use markers.
        body_marker_bc_verts: Marker barycentric vertices, indices on the gel body mesh.
        body_marker_bc_coords: Marker barycentric coordinates.
        density: Density of the material.
        E: Young's modulus.
        nu: Poisson's ratio.
        mu: Friction coefficient.
        collide_with_robot: Whether to collide with the robot.
        shell_body_handle: Shell body handle.
        gel_body_handle: Gel body
        coll_layer: Specify the collision layer
        self_coll: Specify whether to enable self-collision for the soft body

    """

    sensor_name: str = "sensor"
    link_name: str = "link"
    attch_rel_pose: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0, 0]))

    # Gel
    gel_mesh_path: str = ""
    gel_resolution: float = 1e-07

    # Shell
    shell_mesh_path: str = ""

    # Camera
    cam_h: int = 400
    cam_w: int = 400
    cam_fov: float = 60
    cam_pixel_size: float = 0.079375
    cam_intrinsics: np.ndarray = None
    cam_rel_pose: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0, 0]))
    ref_img_path: str = "taccel/data/6k_ref.png"
    mlp_ckpt_path: str = "taccel/data/n2rgb.pth"
    cam_fwd_dir: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))  # tmp
    ray_axis: str = "z"
    ray_start_level: float = 0.01
    ray_rest_level: float = 0.0
    reso_ds_ratio: int = 1

    @property
    def ray_dist_to_depth_offset(self):
        return np.abs(self.ray_start_level - self.ray_rest_level)

    # Coat
    body_coat_mask: np.ndarray = None
    surf_coat_mask: np.ndarray = None
    body_attach_mask: np.ndarray = None
    surf_attach_mask: np.ndarray = None

    # Markers, using PyTorch tensors for efficiency
    with_marker: bool = True
    body_marker_bc_verts: torch.Tensor = None
    marker_bc_coords: torch.Tensor = None

    # Simulation
    density: float = 1e3
    E: float = 5e5
    nu: float = 0.4
    mu: float = 1.0
    collide_with_robot: bool = False
    shell_body_handle: TriMeshBodyHandle = None
    gel_body_handle: TetMeshBodyHandle = None
    coll_layer: int | None = None
    self_coll: bool = False

    @cached_property
    def gel_mesh(self) -> pv.UnstructuredGrid:
        """The tet mesh (in pv.UnstructuredGrid format) of the gel pad.
        This member is a cached property for efficiency.
        """
        return pv.read(self.gel_mesh_path)

    @cached_property
    def shell_mesh(self) -> tm.Trimesh:
        """The trimesh of the shell of the gel pad.
        It is currently omitted and will be added to the sim for realistic visualization.
        This member is a cached property for efficiency.
        """
        return tm.load(self.shell_mesh_path)

    @cached_property
    def attch_rel_tf(self):
        """Relative transformation from the gel pad to the robot link that the sensor is attached to.

        Returns:
            np.ndarray: Relative transformation.
        """
        attch_rel_tf = np.eye(4)
        attch_rel_tf[:3, 3] = self.attch_rel_pose[:3]
        attch_rel_tf[:3, :3] = R.from_euler("xyz", self.attch_rel_pose[3:]).as_matrix()
        return attch_rel_tf

    @property
    def cam_reso(self) -> tuple[int, int]:
        """VBTS Camera resolution, H x W"""
        return (self.cam_h, self.cam_w)

    @property
    def n_pixels(self) -> int:
        """The amount of pixels in total for a VBTS camera"""
        return self.cam_h * self.cam_w

    @cached_property
    def cam_rel_tf(self):
        """Relative transformation from the gel pad to the camera.

        Returns:
            np.ndarray: Relative transformation.
        """
        cam_rel_tf = np.eye(4)
        cam_rel_tf[:3, 3] = self.cam_rel_pose[:3]
        cam_rel_tf[:3, :3] = R.from_euler("xyz", self.cam_rel_pose[3:]).as_matrix()
        return cam_rel_tf

    # TODO: Fix this according to the latest members
    def cfgs_from_json(json_path: str):
        """Load configuration from JSON file.

        Args:
            json_path: Path to the JSON file.

        Returns:
            list[VBTSConfig]: Vision-Based Tactile Sensor configuration.
        """
        with open(json_path, "r") as f:
            config = json.load(f)
        return [VBTSConfig(**cfg) for cfg in config]


# Calculate surface normals based on depth map
def depth_to_normal(depth_map):
    dz_dx = torch.gradient(depth_map, dim=-2)[0]
    dz_dy = torch.gradient(depth_map, dim=-1)[0]
    normals = torch.stack((-dz_dx, -dz_dy, torch.ones_like(depth_map)), dim=-1)
    norms = torch.norm(normals, dim=-1, keepdim=True)
    normals /= norms  # Normalize the vectors
    return normals  # Shape: (..., H, W, 3)


def depth_to_normal_np(depth_map):
    dz_dx = np.gradient(depth_map, axis=1)
    dz_dy = np.gradient(depth_map, axis=0)
    normals = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth_map)))
    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    normals /= norms  # Normalize the vectors
    return normals  # Shape: (H, W, 3)


class VBTSMLP(nn.Module):
    def __init__(self, in_len=14, out_len=3, img_h=512, img_w=512, n_pe=4, input_depth=False):
        super(VBTSMLP, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_len + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, out_len),
        )

        self.input_depth = input_depth

        self.img_h, self.img_w = img_h, img_w
        self.n_pe = n_pe
        self.x_pe_fn = [(lambda x: torch.sin(x * 2**i * torch.pi / img_h)) for i in range(n_pe)]
        self.y_pe_fn = [(lambda x: torch.cos(x * 2**i * torch.pi / img_w)) for i in range(n_pe)]

    def pos_enc(self, coord):
        x, y = coord[..., 0], coord[..., 1]
        return torch.concat(
            [
                torch.stack([x] + [fn(x) for fn in self.x_pe_fn], dim=-1),
                torch.stack([y] + [fn(y) for fn in self.y_pe_fn], dim=-1),
            ],
            dim=-1,
        )

    def forward(self, depth, normal, coord):
        if self.input_depth:
            x = torch.concat([depth, normal, self.pos_enc(coord)], dim=-1)
        else:
            x = torch.concat([normal, self.pos_enc(coord)], dim=-1)
        y = self.fc1(x)
        y = self.fc2(torch.concat([x, y], dim=-1))
        y = torch.clip(y, -1.0, 1.0)
        return y  # .permute((0, 2, 3, 1))
