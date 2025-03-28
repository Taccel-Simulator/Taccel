import warp as wp
from numpy.typing import NDArray
from warp_ipc.utils.constants import BodyType

class TriMeshBodyHandle:

    def __init__(self, x_num: int, x_offset: int, surf_x_num: int, surf_x_offset_num: int, surf_tri_num: int, surf_tri_offset: int, surf_edge_num: int, surf_edge_offset: int, flipped: bool, is_closed: bool, is_affine: bool, body_type: BodyType, ABD_center: NDArray, body_index: int, E: float, mu: float, nu: float, xi: float, volume: float | None=None, mass: float | None=None, wp_mesh: wp.sim.Mesh=None) -> None:
        self.x_num = x_num
        self.x_offset = x_offset
        self.surf_x_num = surf_x_num
        self.surf_x_offset_num = surf_x_offset_num
        self.surf_tri_num = surf_tri_num
        self.surf_tri_offset = surf_tri_offset
        self.surf_edge_num = surf_edge_num
        self.surf_edge_offset = surf_edge_offset
        self.flipped = flipped
        self.is_closed = is_closed
        self.is_affine = is_affine
        self.body_type = body_type
        self.ABD_center = ABD_center
        self.body_index = body_index
        self.E = E
        self.mu = mu
        self.nu = nu
        self.xi = xi
        self.volume = volume
        self.mass = mass
        self.wp_mesh = wp_mesh

    def __str__(self) -> str:
        return f'TriMeshBodyHandle(x_num={self.x_num}, x_offset={self.x_offset}, surf_x_num={self.surf_x_num}, surf_x_offset_num={self.surf_x_offset_num}, ' + f'surf_tri_num={self.surf_tri_num}, surf_tri_offset={self.surf_tri_offset}, surf_edge_num={self.surf_edge_num}, surf_edge_offset={self.surf_edge_offset}, ' + f'flipped={self.flipped}, is_closed={self.is_closed}, is_affine={self.is_affine}, body_type={self.body_type}, ' + f'ABD_center={self.ABD_center}, body_index={self.body_index}), E={self.E}, mu={self.mu}, ' + f'nu={self.nu}, xi={self.nu})'

    def __repr__(self) -> str:
        return str(self)

class TetMeshBodyHandle:

    def __init__(self, surf_handle: TriMeshBodyHandle, tets_num: int, tets_offset: int) -> None:
        self.surf_handle = surf_handle
        self.tets_num = tets_num
        self.tets_offset = tets_offset

    @property
    def volume(self):
        return self.surf_handle.volume

    @property
    def mass(self):
        return self.surf_handle.mass

    @property
    def x_offset(self):
        return self.surf_handle.x_offset

    @property
    def x_num(self):
        return self.surf_handle.x_num

    @property
    def surf_tri_num(self):
        return self.surf_handle.surf_tri_num

    @property
    def surf_tri_offset(self):
        return self.surf_handle.surf_tri_offset

    @property
    def is_affine(self):
        return False

    @property
    def body_type(self):
        return self.surf_handle.body_type

    @property
    def body_index(self):
        return self.surf_handle.body_index

    @property
    def ABD_center(self):
        return self.surf_handle.ABD_center

    def __str__(self) -> str:
        return f'TetMeshBodyHandle(surf_handle={self.surf_handle}, tets_num={self.tets_num}, tets_offset={self.tets_offset})'

    def __repr__(self) -> str:
        return str(self)
BodyHandle = TetMeshBodyHandle | TriMeshBodyHandle