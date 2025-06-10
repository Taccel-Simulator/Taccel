import os.path as osp
from functools import cached_property
from typing import Dict, List
import numpy as np
import torch
import trimesh
import yourdfpy
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
        self.collision_layer = None

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

    @cached_property
    def joint_stiffness(self) -> list[float]:
        return [1.0] * len(self._urdf.actuated_joints)

    @cached_property
    def joint_damping(self) -> list[float]:
        return [float(joint.dynamics.damping) for joint in self._urdf.actuated_joints]

    @cached_property
    def joint_max_vel(self) -> list[float]:
        return [joint.limit.velocity for joint in self._urdf.actuated_joints]

    @cached_property
    def joint_max_force(self) -> list[float]:
        return [joint.limit.effort for joint in self._urdf.actuated_joints]

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
            tac_reso = tac_reso or 1e-05
            urdf_filepath = 'assets/robots/deft_shadow_hand_description/shadowhand_tactile_ipc.urdf'
            tac_fabr_filepath = f'assets/robots/deft_shadow_hand_description/tac_fab_{tac_reso}.json'
        elif model_name == 'tactile-allegro':
            tac_reso = tac_reso or 1e-07
            urdf_filepath = 'assets/robots/allegro_hand_description/allegro_hand_description_right.urdf'
            tac_fabr_filepath = f'assets/robots/allegro_hand_description/tac_fabr_{tac_reso}.json'
        elif model_name == 'allegro-digit360':
            tac_reso = tac_reso or 1e-07
            urdf_filepath = 'assets/robots/allegro_digit360/digit360_allegro_right.urdf'
            tac_fabr_filepath = f'assets/robots/allegro_digit360/tac_fabr_{tac_reso}.json'
        mesh_filepath = osp.dirname(urdf_filepath)
        return (urdf_filepath, mesh_filepath, tac_fabr_filepath)

    @property
    def finger_links(self) -> List[Link]:
        return self._finger_links

    @property
    def fingertip_links(self) -> List[Link]:
        return self._fingertip_links

    @cached_property
    def link_coll_layers(self) -> List[int]:
        return [link.collision_layer for link in self._links if link.collision_layer is not None]

class JointController:
    default_stiffness: float = 1.0
    default_damping: float = 0.1
    default_max_effort: float = 1.0
    default_max_velocity: float = 1.57

    def __init__(self, example_robot: Robot, num_envs: int, device: torch.device=torch.device('cuda:0')):
        self._num_envs = num_envs
        self._example_robot = example_robot
        self.all_joint_names = self._example_robot.actuated_joint_names
        self.lower = torch.tensor(self._example_robot.q_lower, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.upper = torch.tensor(self._example_robot.q_upper, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.q = torch.zeros([self.num_envs, self.num_dof], dtype=torch.float32, device=device)
        stiffness = [k or self.default_stiffness for k in self._example_robot.joint_stiffness]
        damping = [k or self.default_damping for k in self._example_robot.joint_damping]
        max_dq = [dq or self.default_max_velocity for dq in self._example_robot.joint_max_vel]
        max_effort = [tau or self.default_max_effort for tau in self._example_robot.joint_max_force]
        self.stiffness = torch.tensor(stiffness, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.damping = torch.tensor(damping, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.max_dq = torch.tensor(max_dq, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.max_effort = torch.tensor(max_effort, dtype=torch.float32, device=device).unsqueeze(0).tile([num_envs, 1])
        self.arange = list(range(self.num_envs))
        self._device = device

    @property
    def num_dof(self):
        return self._example_robot.num_actuated_joints

    @property
    def num_envs(self):
        return self._num_envs

    def set_state(self, q: torch.Tensor, dq: torch.Tensor=None, ddq: torch.Tensor=None, i_envs: List[int]=None):
        i_envs = self.arange if i_envs is None else i_envs
        q = torch.clamp(q, self.lower[i_envs], self.upper[i_envs])
        self.q[i_envs] = q

    def compute_target(self, q_d: torch.Tensor, i_envs: List[int]=None) -> torch.Tensor:
        i_envs = self.arange if i_envs is None else i_envs
        q_d = torch.clamp(q_d, self.lower[i_envs], self.upper[i_envs])
        self.q[i_envs] = q_d
        return self.q[i_envs]

class MimicTorqueController(JointController):

    def __init__(self, example_robot: Robot, num_envs: int, dt: float=0.02, ttddq: float=10.0, device: torch.device=torch.device('cuda:0')):
        super().__init__(example_robot, num_envs, device)
        self.dq = torch.zeros([self.num_envs, self.num_dof], dtype=torch.float32, device=device)
        self.ddq = torch.zeros([self.num_envs, self.num_dof], dtype=torch.float32, device=device)
        self.dt = dt
        self.tau_to_ddq = ttddq
        self.arange = list(range(self.num_envs))
        self._device = device

    def set_state(self, q: torch.Tensor, dq: torch.Tensor=None, ddq: torch.Tensor=None, i_envs: List[int]=None):
        i_envs = self.arange if i_envs is None else i_envs
        q = torch.clamp(q, self.lower[i_envs], self.upper[i_envs])
        self.q[i_envs] = q
        self.dq[i_envs] = 0.0 if dq is None else dq
        self.ddq[i_envs] = 0.0 if ddq is None else ddq

    def compute_target(self, q_d: torch.Tensor, i_envs: List[int]=None) -> torch.Tensor:
        i_envs = self.arange if i_envs is None else i_envs
        q_d = torch.clamp(q_d, self.lower[i_envs], self.upper[i_envs])
        self.ddq[i_envs] = self.tau_to_ddq * torch.clamp(self.stiffness[i_envs] * (q_d - self.q[i_envs]) - self.damping[i_envs] * self.dq[i_envs], -self.max_effort[i_envs], self.max_effort[i_envs])
        self.dq[i_envs] = torch.clamp(self.dq[i_envs] + self.ddq[i_envs] * self.dt, -self.max_dq[i_envs], self.max_dq[i_envs])
        self.q[i_envs] = torch.clamp(self.q[i_envs] + self.dq[i_envs] * self.dt, self.lower[i_envs], self.upper[i_envs])
        return self.q[i_envs]