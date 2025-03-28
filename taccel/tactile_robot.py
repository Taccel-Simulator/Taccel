import json
import os.path as osp
import pickle as pkl
import sys
from functools import cached_property
from typing import List

import numpy as np
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(".")


from taccel.vbts import VBTSConfig
from warp_ipc.robots import Robot


class TactileRobot(Robot):
    """
    A class representing a tactile robot with various tactile sensors.

    Attributes:
        vbts_cfgs (list[VBTSConfig]): List of VBTSConfig objects loaded from the tactile fabrication file.

    Methods:
        gel_handles: Returns a list of gel body handles from the VBTS configurations.
        tac_rest_meshes: Returns a list of unstructured grids representing the rest meshes of the tactile sensors.
        tac_gel_density: Returns a list of densities for the tactile gels.
        tac_gel_E: Returns a list of Young's modulus values for the tactile gels.
        tac_gel_nu: Returns a list of Poisson's ratio values for the tactile gels.
        tac_gel_friction: Returns a list of friction coefficients for the tactile gels.
        num_tactile_markers: Returns the total number of tactile markers.
        num_tac_verts: Returns a list of the number of vertices for each tactile rest mesh.
        num_tac: Returns the number of tactile sensors.
        _load_tac_fabrications: Loads the tactile fabrications from a given path and returns a list of VBTSConfig objects.
    """

    def __init__(
        self,
        urdf_path: str,
        tac_fab_path: str = "",
        mesh_dir: str | None = None,
        actuated_joints: list[str] | None = None,
        env_id: int = 0,
        finger_link_names: List[str] = [],
        fingertip_link_names: List[str] = [],
        device: torch.device = torch.device("cuda:0"),
        use_collision: bool = False,
    ):
        """Initializes the TactileRobot object.

        Args:
            urdf_path (str): Path to the URDF file.
            tac_fab_path (str, optional): Path to the tactile fabrication file. Defaults to "".
            mesh_dir (str | None, optional): Directory containing mesh files. Defaults to None.
            actuated_joints (list[str] | None, optional): List of actuated joints. Defaults to None.
            env_id (int, optional): Environment ID. Defaults to 0.
            finger_link_names (List[str], optional): List of finger link names. Defaults to [].
            fingertip_link_names (List[str], optional): List of fingertip link names. Defaults to [].
            device (torch.device, optional): Device to load the tactile data onto. Defaults to torch.device("cuda:0").
        """
        super().__init__(urdf_path, mesh_dir, actuated_joints, env_id, finger_link_names, fingertip_link_names, use_collision)
        self.vbts_cfgs: list[VBTSConfig] = self._load_tac_fabrications(tac_fab_path, device)

    @cached_property
    def gel_handles(self):
        return [cfg.gel_body_handle for cfg in self.vbts_cfgs]

    @cached_property
    def tac_rest_meshes(self) -> list[pv.UnstructuredGrid]:
        return [cfg.gel_mesh for cfg in self.vbts_cfgs]

    @cached_property
    def tac_gel_density(self) -> list[float]:
        return [cfg.density for cfg in self.vbts_cfgs]

    @cached_property
    def tac_gel_E(self) -> list[float]:
        return [cfg.E for cfg in self.vbts_cfgs]

    @cached_property
    def tac_gel_nu(self) -> list[float]:
        return [cfg.nu for cfg in self.vbts_cfgs]

    @cached_property
    def tac_gel_friction(self) -> list[float]:
        return [cfg.mu for cfg in self.vbts_cfgs]

    @cached_property
    def num_tactile_markers(self) -> int:
        return sum([len(indices) for indices in self.marker_vert_idx])

    @cached_property
    def num_tac_verts(self):
        return [m.points.shape[0] for m in self.tac_rest_meshes]

    @cached_property
    def num_tac(self):
        return len(self.vbts_cfgs)

    @staticmethod
    def _load_tac_fabrications(tac_fab_path: str, load_device: torch.device) -> list[VBTSConfig]:
        if tac_fab_path is None:
            return []

        tac_fab_data = json.load(open(tac_fab_path, "r"))
        tac_fab_dir = osp.dirname(tac_fab_path)

        finger_link_names = [fab["link_name"] for fab in tac_fab_data]

        tac_mesh_rel_poses = np.eye(4, dtype=np.float64)[None, :, :].repeat(len(finger_link_names), axis=0)

        # Load tactile sensor meshes from files
        tac_stick_masks, tac_coat_masks, tac_coat_masks_surf = [], [], []
        tac_rest_meshes = []
        marker_vert_idx, marker_vert_idx_surf, marker_coords = [], [], []

        vbts_cfgs: List[VBTSConfig] = []

        for i_tac, fab in enumerate(tac_fab_data):
            # Load tet mesh for gel
            mesh_vtk_file = fab.get(
                "mesh_vtk_file",
                osp.join(osp.dirname(fab["mesh_path"]), "pad_maxv={}.vtk".format(fab["reso"])),
            )
            mesh_vtk_file = osp.join(tac_fab_dir, mesh_vtk_file)
            tac_rest_meshes.append(pv.read(mesh_vtk_file))

            # Load metadata
            mesh_metadata_file = fab.get(
                "mesh_metadata_file",
                osp.join(osp.dirname(fab["mesh_path"]), "pad_maxv={}.pkl".format(fab["reso"])),
            )
            mesh_metadata_file = osp.join(tac_fab_dir, mesh_metadata_file)
            metadata = pkl.load(open(mesh_metadata_file, "rb"))

            # Load metadata - sticking mask and coat mask
            tac_stick_mask = metadata["stick_mask"]
            tac_coat_mask = metadata["coat_mask"]
            tac_coat_mask_surf = metadata["coat_mask_surf"]
            assert tac_stick_mask.shape == (tac_rest_meshes[i_tac].points.shape[0],)
            assert tac_coat_mask.shape == (tac_rest_meshes[i_tac].points.shape[0],)

            tac_stick_masks.append(tac_stick_mask)
            tac_coat_masks.append(tac_coat_mask)
            tac_coat_masks_surf.append(tac_coat_mask_surf)

            # Load metadata - markers
            marker_vert_idx.append(torch.tensor(metadata["marker_vert_idx"], dtype=torch.long, device=load_device))
            marker_vert_idx_surf.append(
                torch.tensor(
                    metadata["marker_vert_idx_surf"],
                    dtype=torch.long,
                    device=load_device,
                )
            )
            marker_coords.append(np.array(metadata["marker_bc_coords"]))

            tac_mesh_rel_poses[i_tac, :3, :3] = R.from_euler("xyz", tac_fab_data[i_tac]["rot"]).as_matrix()
            tac_mesh_rel_poses[i_tac, :3, 3] = np.asarray(tac_fab_data[i_tac]["pos"])

            cfg = VBTSConfig(
                sensor_name=fab.get("sensor_name", "sensor"),
                link_name=fab["link_name"],
                attch_rel_pose=np.array(fab["pos"] + fab["rot"]),
                gel_mesh_path=mesh_vtk_file,
                shell_mesh_path=fab.get("shell_mesh_path"),
                body_coat_mask=tac_coat_mask,
                surf_coat_mask=tac_coat_mask_surf,
                body_attach_mask=tac_stick_mask,
                with_marker=True,
                body_marker_bc_verts=marker_vert_idx[i_tac],
                marker_bc_coords=torch.from_numpy(marker_coords[i_tac]).to(load_device),
            )
            vbts_cfgs.append(cfg)

        return vbts_cfgs
