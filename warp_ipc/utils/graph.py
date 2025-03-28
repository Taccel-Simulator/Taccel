import torch
import warp as wp
import numpy as np
from collections import defaultdict

def collect_edge_and_point(boundary_faces):
    boundary_point_set = set()
    boundary_edge_map = defaultdict(int)
    for face in boundary_faces:
        v0 = int(face[0])
        v1 = int(face[1])
        v2 = int(face[2])
        boundary_point_set.add(v0)
        boundary_point_set.add(v1)
        boundary_point_set.add(v2)
        edge01 = (min(v0, v1), max(v0, v1))
        edge12 = (min(v1, v2), max(v1, v2))
        edge20 = (min(v0, v2), max(v0, v2))
        boundary_edge_map[edge01] += 1
        boundary_edge_map[edge12] += 1
        boundary_edge_map[edge20] += 1
    boundary_points = list(boundary_point_set)
    boundary_edges = []
    is_closed = True
    for key in boundary_edge_map:
        boundary_edges.append([key[0], key[1]])
        is_closed &= boundary_edge_map[key] > 1
    return (is_closed, boundary_edges, boundary_points)

def collect_codim2_boundary(faces):
    boundary_edge_map = defaultdict(int)
    for face in faces:
        v0 = int(face[0])
        v1 = int(face[1])
        v2 = int(face[2])
        edge01 = (min(v0, v1), max(v0, v1))
        edge12 = (min(v1, v2), max(v1, v2))
        edge20 = (min(v0, v2), max(v0, v2))
        boundary_edge_map[edge01] += 1
        boundary_edge_map[edge12] += 1
        boundary_edge_map[edge20] += 1
    codim2_boundary_points = set()
    for key in boundary_edge_map:
        if boundary_edge_map[key] == 1:
            codim2_boundary_points.add(key[0])
            codim2_boundary_points.add(key[1])
    return list(codim2_boundary_points)

def collect_bending(triangles):
    wedges = {}
    triangle_list = triangles
    for tri in triangle_list:
        for i in range(3):
            key = tuple(sorted([tri[(i + 1) % 3], tri[(i + 2) % 3]]))
            value = [tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]]
            if key not in wedges:
                wedges[key] = [value]
            else:
                wedges[key].append(value)
    stencils = []
    for tris in wedges.values():
        if len(tris) == 2:
            tri0 = tris[0]
            tri1 = tris[1]
            stencils.append([tri0[0], tri0[1], tri0[2], tri1[0]])
    return stencils