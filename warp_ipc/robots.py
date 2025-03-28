import os.path as osp
import sys
from functools import cached_property
from typing import Dict, List
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R
sys.path.append('.')
from warp_ipc.body_handle import BodyHandle

def geom2mesh(geometry: yourdfpy.Geometry, urdf_dir: str):
    ret = None
    if geometry.box is not None:
        ret = trimesh.primitives.Box(extents=geometry.box.size).to_mesh()
    elif geometry.sphere is not None:
        ret = trimesh.primitives.Sphere(radius=geometry.sphere.radius).to_mesh()
    elif geometry.cylinder is not None:
        ret = trimesh.primitives.Cylinder(radius=geometry.cylinder.radius, height=geometry.cylinder.length).to_mesh()
    elif geometry.mesh is not None:
        filename = geometry.mesh.filename
        PREFIX = 'package://'
        if filename.startswith(PREFIX):
            filename = filename[len(PREFIX):]
        ret = trimesh.load_mesh(osp.join(urdf_dir, filename))
    else:
        raise NotImplementedError()
    return ret

class Link:

    def __init__(self, robot: 'Robot', urdf_link: yourdfpy.Link, link_meshes_dir: str, use_collision: bool=False) -> None:
        self.use_collision = use_collision
        self.handle: BodyHandle | None = None
        self.robot = robot
        self.urdf_link = urdf_link
        self.rest_mesh = self._compute_rest_mesh(link_meshes_dir)
        self.collision_layer = 0

    def _compute_rest_mesh(self, link_meshes_dir) -> trimesh.Trimesh:
        reset_meshes = []
        for v in self.urdf_link.collisions if self.use_collision else self.urdf_link.visuals:
            rest_visual_mesh = geom2mesh(v.geometry, link_meshes_dir)
            if v.origin is not None:
                rest_visual_mesh.apply_transform(v.origin)
            reset_meshes.append(rest_visual_mesh)
        rest_mesh = trimesh.util.concatenate(reset_meshes)
        return rest_mesh

    @property
    def name(self):
        return self.urdf_link.name

    @property
    def urdf_local_transform(self):
        return self.robot._urdf.get_transform(self.name)

    @property
    def urdf_global_transform(self):
        return self.robot._root_tf @ self.urdf_local_transform

    @property
    def urdf_mesh(self):
        mesh = self.rest_mesh.copy()
        mesh.apply_transform(self.urdf_local_transform)
        return mesh

class Kinematic:

    def __init__(self, urdf_path: str, mesh_dir: str | None=None, actuated_joints: list[str] | None=None, env_id: int=0, use_collision: bool=False) -> None:
        if mesh_dir is None:
            mesh_dir = osp.dirname(urdf_path)
        self._mesh_dir = mesh_dir
        self._env_id = env_id
        self._urdf = yourdfpy.URDF.load(urdf_path, force_mesh=True)
        self._root_tf = np.eye(4)
        self._links: List[Link] = []
        self._links_map: Dict[str, Link] = {}
        for (_, link) in self._urdf._link_map.items():
            robot_link = Link(self, link, self._mesh_dir, use_collision)
            self._links.append(robot_link)
            self._links_map[link.name] = robot_link
        self.vbts_mlp = None
        self._actuated_joint_names = actuated_joints or self._urdf.actuated_joint_names
        self._actuated_joint_idx = [self._urdf.actuated_joint_names.index(j) for j in self.actuated_joint_names]

    @property
    def links(self):
        return self._links

    @property
    def env_id(self):
        return self._env_id

    @property
    def num_joints(self) -> int:
        return len(self._urdf.joint_names)

    @property
    def joint_names(self) -> list[str]:
        return self._urdf.joint_names

    @property
    def num_actuated_joints(self) -> int:
        return len(self._actuated_joint_names)

    @property
    def actuated_joint_names(self) -> list[str]:
        return self._actuated_joint_names

    @property
    def actuated_joint_idx(self) -> list[int]:
        return self._actuated_joint_idx

    @property
    def links_map(self) -> Dict[str, Link]:
        return self._links_map

    @property
    def base_link_name(self) -> str:
        return self._urdf.base_link

    @property
    def base_link(self) -> Link:
        return self._links_map[self.base_link_name]

    @cached_property
    def q_lower(self):
        return [j.limit.lower for j in self._urdf.actuated_joints]

    @cached_property
    def q_upper(self):
        return [j.limit.upper for j in self._urdf.actuated_joints]

    @property
    def triangular_meshes(self) -> list[trimesh.Trimesh] | trimesh.Trimesh:
        meshes = []
        for link in self._links:
            mesh = link.rest_mesh.copy()
            if len(mesh.vertices) == 0:
                continue
            v_homo = np.pad(mesh.vertices, ((0, 0), (0, 1)), constant_values=1.0)
            mesh.vertices = (self._root_tf @ link.urdf_local_transform @ v_homo.T).T[:, :3]
            meshes.append(mesh)
        return meshes

    @property
    def root_transform(self):
        return self._root_tf

    def get_link(self, link_name: str) -> Link:
        return self._links_map[link_name]

    @cached_property
    def link_handles(self):
        return [link.handle for link in self._links]

class Robot(Kinematic):

    def __init__(self, urdf_path: str, mesh_dir: str | None=None, actuated_joints: list[str] | None=None, env_id: int=0, finger_link_names: List[str]=[], fingertip_link_names: List[str]=[], use_collision: bool=False):
        super().__init__(urdf_path, mesh_dir, actuated_joints, env_id, use_collision)
        self._finger_links = [self.get_link(name) for name in finger_link_names]
        self._fingertip_links = [self.get_link(name) for name in fingertip_link_names]

    @staticmethod
    def get_fabr_path(model_name: str='tactile-pandahand', tac_reso: float=None) -> str:
        if model_name == 'single-sensor':
            tac_reso = tac_reso or 1e-07
            urdf_filepath = 'assets/robots/single_sensor/single_sensor.urdf'
            tac_fabr_filepath = f'assets/robots/single_sensor/tac_fabr_{tac_reso}.json'
        if model_name == 'tactile-pandahand':
            tac_reso = tac_reso or 1e-05
            urdf_filepath = 'assets/robots/franka_description/panda_tac.urdf'
            tac_fabr_filepath = f'assets/robots/franka_description/tac_fabr_{tac_reso}.json'
        elif model_name == 'tactile-robotiq3f':
            tac_reso = tac_reso or 1e-05
            urdf_filepath = 'assets/robots/robotiq_3f_gripper/robotiq-3f-gripper_articulated.urdf'
            tac_fabr_filepath = f'assets/robots/robotiq_3f_gripper/meshes/robotiq-3f-gripper_articulated/tac_fabr_square_{tac_reso}.json'
        elif model_name == 'tactile-gelhand':
            tac_reso = tac_reso or 1e-09
            urdf_filepath = 'assets/robots/f_tac_hand_description/urdf/f_tac_hand_rigid.urdf'
            tac_fabr_filepath = f'assets/robots/f_tac_hand_description/tac_fab_{tac_reso}.json'
        elif model_name == 'tactile-shadowhand':
            urdf_filepath = 'assets/robots/deft_shadow_hand_description/shadowhand_tactile_ipc.urdf'
            tac_fabr_filepath = 'assets/robots/deft_shadow_hand_description/tac_fab_default.json'
        elif model_name == 'tactile-allegro':
            tac_reso = tac_reso or 1e-10
            urdf_filepath = 'assets/robots/allegro_hand_description/allegro_hand_description_right.urdf'
            tac_fabr_filepath = f'assets/robots/allegro_hand_description/tac_fabr_{tac_reso}.json'
        elif model_name == 'allegro-digit360':
            tac_reso = tac_reso or 1e-07
            urdf_filepath = 'assets/robots/allegro_digit360/digit360_allegro_right.urdf'
            tac_fabr_filepath = f'assets/robots/allegro_digit360/tac_fabr_{tac_reso}.json'
        elif model_name == 'deft-shadow':
            tac_reso = tac_reso or 5e-08
            urdf_filepath = 'assets/robots/deft_description/deft_ur10e_rigid.urdf'
            tac_fabr_filepath = f'assets/robots/deft_description/tac_fab_{tac_reso}.json'
        mesh_filepath = osp.dirname(urdf_filepath)
        return (urdf_filepath, mesh_filepath, tac_fabr_filepath)

    @property
    def finger_links(self) -> List[Link]:
        return self._finger_links

    @property
    def fingertip_links(self) -> List[Link]:
        return self._fingertip_links