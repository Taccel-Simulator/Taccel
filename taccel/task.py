from enum import Enum
from functools import cached_property
import os
from abc import abstractmethod

import numpy as np
import torch

from warp.sim import Control, Integrator, Model
from warp.sim.render import SimRendererOpenGL

from taccel.taccel import TaccelModel
from taccel.utils.mesh_utils import mesh_worker
from taccel.utils.torch_utils import tensor_clamp
from warp_ipc.energy import nhk_elastic_energy, rigidity_energy
from warp_ipc.ipc_integrator import IPCIntegrator
from warp_ipc.utils.log import LoggingLevel, set_logging_level
from warp_ipc.utils import log
from warp_ipc.utils.constants import (
    ENV_STATE_INVALID,
    ENV_STATE_VALID,
    BodyType,
    MembraneType,
)
from pytorch3d import transforms

set_logging_level(LoggingLevel.INFO)


class TimestepSchedule(Enum):
    CONSTANT = 0
    LINEAR = 1
    EXPONENTIAL = 2


class TactileTaskCfg:
    max_consecutive_successes: int = 0
    episode_length: int = 50
    """Max episode length for the policy to run."""
    dt: float = 1 / 50
    substeps = 1

    invalid_state_penalty: float = -50

    fix_robot: bool = False
    """If False, the robot pose action shape will be (6,) and applied as an EE delta pose on the robot."""
    gripper_mode: bool = True
    """If True, the finger joint action shape will be (1,) and casted to (2,) for applying on the robot."""

    max_root_velocity_axis: float = 0.1
    max_root_velocity_angle: float = np.pi / 3

    use_robot_collision_mesh: bool = True

    headless: bool = True
    dump_scene_mesh: bool = False
    scene_mesh_keep_last: int = 500

    nhk_energy_penalty_thres: float = 0.002  # TODO: A better value
    nhk_energy_penalty_scale: float = -25.0

    rgd_energy_penalty_thres: float = 0.01  # TODO: A better value
    rgd_energy_penalty_scale: float = -25.0


class TactileTask:
    """
    A class to represent a tactile robotic task.

    Attributes:
    ----------
    output_dir : str
        The directory where output files will be saved.
    model : TaccelModel
        The model used for the task.
    integrator : IPCIntegrator
        The integrator used for simulation.
    renderer : SimRendererOpenGL
        The renderer used for visualization.
    """

    task_name: str = "tactile_task"
    env_cfg: TactileTaskCfg

    def __init__(self, output_dir: str, num_envs: int = 1, env_cfg: TactileTaskCfg = None):
        """
        Initializes the task with the specified output directory and number of environments.

        Args:
            output_dir (str): The directory where output files will be saved.
            num_envs (int, optional): The number of environments to initialize. Defaults to 1.
        """
        env_cfg = env_cfg or TactileTaskCfg()
        self.env_cfg = env_cfg
        self.output_dir = output_dir

        self.substeps = self.env_cfg.substeps
        self.dt = self.env_cfg.dt / self.substeps

        self.model = TaccelModel(num_envs=num_envs)
        self.model.dhat = 1e-4

        n_row = int(np.sqrt(num_envs))
        n_col = int(np.ceil(num_envs / n_row))
        self.env_origins = TaccelModel.get_env_pos(self.num_envs, n_row, n_col, env_spacing=0.5, device=self.device)

        self.integrator = IPCIntegrator()
        self.integrator.use_cpu = False
        self.integrator.max_newton_iter = 15
        self.integrator.tol = 1e-2
        self.integrator.max_cg_iter = 300
        self.integrator.cg_rel_tol = 1e-3
        self.integrator.use_inversion_free_step_size_filter = True
        self.integrator.inversion_free_cubic_coef_tol = 1e-6

        self.setup_scene()

        if not self.env_cfg.headless:
            self._setup_renderer()
        if self.env_cfg.dump_scene_mesh:
            self._setup_scene_dumper()

        self.prep_buffers()

    def _setup_scene_dumper(self):
        from multiprocessing import Process, Queue

        self.scene_mesh_queue = Queue()

        mesh_dump_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(mesh_dump_dir, exist_ok=True)
        self.scene_mesh_process = Process(
            target=mesh_worker,
            args=(
                self.scene_mesh_queue,
                mesh_dump_dir,
                self.env_cfg.scene_mesh_keep_last,
            ),
        )
        self.scene_mesh_process.start()

    def __del__(self):
        if self.env_cfg.dump_scene_mesh:
            self.scene_mesh_queue.put(None)
            self.scene_mesh_process.join()

    def _setup_renderer(self):
        stage_path = os.path.join(self.output_dir, "parallel_test.usd")
        self.renderer = SimRendererOpenGL(
            self.model,
            stage_path,
            scaling=1,
            near_plane=1.0,
            far_plane=100.0,
            camera_fov=45.0,
            camera_pos=(0.0, 2.0, 10.0),
            camera_front=(0.0, 0.0, -1.0),
            camera_up=(0.0, 1.0, 0.0),
        )

    @property
    @abstractmethod
    def num_obs(self) -> int:
        pass

    @property
    @abstractmethod
    def num_state_obs(self):
        return 0

    @cached_property
    def num_act(self) -> int:
        """Size of actions output by the policy."""
        _num_act = 0
        if not self.env_cfg.fix_robot:
            _num_act += 6
        if self.env_cfg.gripper_mode:
            _num_act += 1
        else:
            _num_act += self.model.dummy_robot.num_actuated_joints
        return _num_act

    @property
    def num_act_full(self) -> int:
        """Size of actions that is finally passed to `pre_physics_step`. Will be larger than `num_act` if the policy outputs in a subspace of the action space."""
        return self.num_act

    @property
    def num_envs(self) -> int:
        return self.model.num_envs

    @abstractmethod
    def setup_scene(self):
        pass

    @property
    def device(self):
        return self.model.device

    def prep_buffers(self):
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float32)
        self.state_obs_buf = torch.zeros(
            (self.num_envs, self.num_state_obs),
            device=self.device,
            dtype=torch.float32,
        )
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.reset_goal_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.consecutive_successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self.successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.actions = torch.zeros(
            (self.num_envs, self.num_act_full),
            device=self.device,
            dtype=torch.float32,
        )

        self.applied_targets = torch.zeros(
            (self.num_envs, self.model.dummy_robot.num_actuated_joints),
            device=self.device,
            dtype=torch.float32,
        )

        self.extras = {}

        self.env_markers, self.env_rest_markers = None, None

    def initialize(self):
        self.model.init()

    def finalize(self):
        self.model.finalize()

    def compute_tactile(self):
        # [num_envs, num_markers_per_env, 3]
        self.env_markers, self.env_rest_markers = self.model.tac_markers_rest_and_current_hand_local
        # markers = markers - self.env_origins.unsqueeze(1)
        # rest_markers = rest_markers - self.env_origins.unsqueeze(1)
        self.marker_buf = self.env_markers - self.env_rest_markers  # [num_envs, num_markers_per_env, 3]

    def reset_idx(self, env_ids: list[int]):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self.domain_randomization(env_ids)

    def domain_randomization(self, env_ids):
        pass

    @abstractmethod
    def reset_target(self, env_ids: list[int]):
        pass

    def pre_physics_step(self, actions):
        """Apply actions to robot"""

        # Handle resets first
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if self.model.frame == 0:
            action_env_ids = []
        else:
            action_env_ids = self.reset_buf.logical_not().nonzero(as_tuple=False).squeeze(-1).tolist()
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        if len(action_env_ids) == 0:
            return

        # Store actions for reward computation
        self.actions[action_env_ids] = actions[action_env_ids].clone()

        # Split actions into pose and gripper commands
        pose_actions = actions[action_env_ids, :6]  # [dx, dy, dz, drx, dry, drz]
        pose_actions = (
            pose_actions
            * torch.tensor(
                [self.env_cfg.max_root_velocity_axis] * 3 + [self.env_cfg.max_root_velocity_angle] * 3,
                device=self.device,
            )
            * self.dt
        )
        hand_joint_actions = actions[:, -1:] * 0.1 * self.dt  # max speed 10cm / s
        if self.env_cfg.gripper_mode:
            hand_joint_actions = hand_joint_actions.tile([1, self.model.dummy_robot.num_actuated_joints])

        # Get current robot poses
        self.robot_poses = torch.stack([torch.from_numpy(robot.root_transform).to(self.device) for robot in self.model.robots])
        robot_poses = self.robot_poses[action_env_ids]
        # Apply delta transforms
        # delta_pose = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        # delta_pose[:, :3, 3] = pose_actions[:, :3]  # translations
        # delta_pose[:, :3, :3] = transforms.euler_angles_to_matrix(pose_actions[:, 3:], convention="XYZ")
        # robot_poses = robot_poses @ delta_pose

        robot_poses[:, :3, :3] = torch.matmul(
            transforms.euler_angles_to_matrix(pose_actions[:, 3:], convention="XYZ"),
            robot_poses[:, :3, :3],
        )
        robot_poses[:, :3, 3] = robot_poses[:, :3, 3] + pose_actions[:, :3]

        # Set new robot states
        self.applied_targets[action_env_ids] = tensor_clamp(
            self.applied_targets[action_env_ids] + hand_joint_actions[action_env_ids],
            self.hand_dof_lower_limits.unsqueeze(0),
            self.hand_dof_upper_limits.unsqueeze(0),
        )
        self.model.set_robot_targets(self.applied_targets[action_env_ids], robot_poses, action_env_ids)

    @cached_property
    def hand_dof_lower_limits(self):
        return torch.tensor(self.model.dummy_robot.q_lower, device=self.device)

    @cached_property
    def hand_dof_upper_limits(self):
        return torch.tensor(self.model.dummy_robot.q_upper, device=self.device)

    @abstractmethod
    def get_observations(self):
        pass

    def compute_sim_energies(self):
        hat_h = self.model.dx_div_dv_scale_list[self.model.time_int_rule] * self.env_cfg.dt
        scale = hat_h * hat_h

        # TODO: Should these energies be averaged over all nodes?
        self.rigidity_energy = rigidity_energy.val(self.model.y, self.model, scale, True)
        self.rigidity_energy = torch.nan_to_num(self.rigidity_energy, nan=0.0)

        self.vol_elastic_energy = nhk_elastic_energy.val(self.model.particle_q, self.model, scale, True)
        self.vol_elastic_energy = torch.nan_to_num(self.vol_elastic_energy, nan=0.0)

    def calculate_metrics(self):
        """Base class for calculating metrics: rewards, resets, success, etc.

        This implementation (in the base class) is for handling invalid states.
        The task-specific implementation should override this method but call
        super().calculate_metrics to handle invalid states afterwards.
        """
        invalid_envs = torch.from_numpy(self.model.get_environment_states() == 1).to(self.device)
        if invalid_envs.sum() > 0:
            log.warn(f"Reset invalid envs: {torch.where(invalid_envs)[0]}")
            self.reset_buf += invalid_envs
            self.rew_buf -= self.env_cfg.invalid_state_penalty * invalid_envs.float()

        rgd_rew = torch.clamp(self.rigidity_energy - self.env_cfg.rgd_energy_penalty_thres, min=0) * self.env_cfg.rgd_energy_penalty_scale
        nhk_rew = torch.clamp(self.vol_elastic_energy - self.env_cfg.nhk_energy_penalty_thres, min=0) * self.env_cfg.nhk_energy_penalty_scale
        self.rew_buf += rgd_rew + nhk_rew
        self.extras["sim/rigidity_energy"] = self.rigidity_energy
        self.extras["sim/nhk_energy"] = self.vol_elastic_energy

    @abstractmethod
    def compute_states(self):
        pass

    def post_physics_step(self):
        self.progress_buf += 1

        # Update states
        self.compute_states()
        self.compute_tactile()
        self.compute_sim_energies()

        self.get_observations()
        self.calculate_metrics()

        return (
            self.obs_buf,
            self.state_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def step(self, control: Control = None):
        # TODO: Interpolation
        iter = 0
        for i_step in range(self.substeps):
            iter += self.integrator.simulate(self.model, dt=self.dt, control=control)

        if not self.env_cfg.headless:
            self.renderer.begin_frame(self.model.elapsed_time)
            self.renderer.render(self.model.state())
            self.renderer.end_frame()

        # [num_envs, num_markers_per_env, 3]
        env_markers, env_rest_markers = self.env_markers, self.env_rest_markers
        env_markers = (env_markers + self.env_origins.unsqueeze(1)).cpu().numpy().reshape(-1, 3) if env_markers is not None else env_markers
        env_rest_markers = (
            (env_rest_markers + self.env_origins.unsqueeze(1)).cpu().numpy().reshape(-1, 3) if env_rest_markers is not None else env_rest_markers
        )
        if self.env_cfg.dump_scene_mesh:
            self.scene_mesh_queue.put((
                self.model.get_scene_mesh(),
                (env_markers, env_rest_markers),
                self.model.frame,
            ))

        return iter
