from enum import Enum
import warp as wp
TRIANGLE_POINT_PAIR = wp.constant(wp.int32(0))
POINT_TRIANGLE_PAIR = wp.constant(wp.int32(1))
EDGE_EDGE_PAIR = wp.constant(wp.int32(2))
BARRIER = wp.constant(0)
FRICTION = wp.constant(1)
NUM_COLLISION_OFF_BLOCKS_PER_PAIR = 8 * 7
FLOAT64_EPSILON = 2.23e-16
ENV_STATE_VALID = wp.constant(wp.int32(0))
ENV_STATE_INVALID = wp.constant(wp.int32(1))
ENV_STATE_NEWTON_SOLVED = wp.constant(wp.int32(2))

class BodyType(Enum):
    SOFT_SHELL_LIKE = 0
    SOFT_VOLUMETRIC = 1
    AFFINE = 2
__1 = wp.constant(wp.float64(-1.0))
_0 = wp.constant(wp.float64(0.0))
_1 = wp.constant(wp.float64(1.0))
_2 = wp.constant(wp.float64(2.0))
id_3 = wp.constant(wp.mat33d(_1, _0, _0, _0, _1, _0, _0, _0, _1))

class MembraneType(Enum):
    BARAFF_WITKIN = 0
    ARAP = 1
    MASS_SPRING = 2
    DEBUG_ZERO = 3

class VolMaterialType(Enum):
    NEO_HOOKEAN = 0
    STVK = 1
    DEBUG_ZERO = 2
    STRAIN_CONSTRAINT = 3

class AffineMaterialType(Enum):
    RIGIDITY = 0
    NEO_HOOKEAN = 1