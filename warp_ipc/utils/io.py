import numpy as np
from plyfile import PlyData, PlyElement

def write_ply(filename, points, tris):
    vertex = np.array([tuple(v) for v in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    face_data = tris.astype('i4')
    face = np.empty(len(face_data), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = face_data
    el = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el, el2]).write(filename)

def load_obj(file_name):
    vertices = []
    faces = []
    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                parsed = line.strip().split()[1:]
                for p in parsed:
                    faces.append(int(p.split('/')[0]) - 1)
    return (np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32).reshape(-1, 3))