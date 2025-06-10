from enum import Enum
import warp.sim
from numpy.typing import NDArray
from .body_handle import BodyHandle

class JointType(Enum):
    REVOLUTE = 0
    PRISMATIC = 1
    HELICAL = 2
    BEZIER = 3

class WorldJointHandle:

    def __init__(self, body_handle: BodyHandle, joint_type: JointType) -> None:
        self.body_handle = body_handle
        self.joint_type = joint_type

class LocalJointHandle:

    def __init__(self, parent_handle: BodyHandle, child_handle: BodyHandle, joint_type: JointType, joint_origin_transform: NDArray, joint_axis: NDArray) -> None:
        self.parent_handle = parent_handle
        self.child_handle = child_handle
        self.joint_type = joint_type
        self.joint_origin_transform: NDArray = joint_origin_transform
        self.joint_axis: NDArray = joint_axis