from enum import Enum
from typing import TYPE_CHECKING
from .body_handle import BodyHandle

class WorldJointType(Enum):
    REVOLUTE = 0
    PRISMATIC = 1
    HELICAL = 2
    BEZIER = 3

class WorldJointHandle:

    def __init__(self, body_handle: BodyHandle, joint_type: WorldJointType) -> None:
        self.body_handle = body_handle
        self.joint_type = joint_type