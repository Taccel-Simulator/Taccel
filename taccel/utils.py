import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import pyvista as pv
import trimesh as tm


def pv_to_tm(pv_mesh: pv.UnstructuredGrid):
    verts = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    return tm.Trimesh(vertices=verts, faces=faces)


def o3d_to_nvisii_mesh(
    name: str, mesh: o3d.geometry.TriangleMesh, coat_mask=None, color=False
):
    verts, faces, normals = (
        np.asarray(mesh.vertices).flatten().tolist(),
        np.asarray(mesh.faces).flatten().tolist(),
        np.asarray(mesh.vertex_normals).flatten().tolist(),
    )
    if color:
        colors = np.ones([len(mesh.vertices), 4])
        if coat_mask is not None:
            colors[np.where(~coat_mask)[0], 3] = 0.0

        colors = colors.flatten().tolist()
        return nvisii.mesh.create_from_data(
            name,
            positions=verts,
            normals=normals,
            colors=colors,
            color_dimensions=4,
            indices=faces,
        )
    else:
        return nvisii.mesh.create_from_data(
            name, positions=verts, normals=normals, indices=faces
        )


def tm_to_nvisii_mesh(name: str, mesh: tm.Trimesh, coat_mask=None, color=False):
    verts, faces, normals = (
        np.asarray(mesh.vertices).flatten().tolist(),
        np.asarray(mesh.faces).flatten().tolist(),
        np.asarray(mesh.vertex_normals).flatten().tolist(),
    )
    if color:
        colors = np.ones([len(mesh.vertices), 4], dtype=float)
        if coat_mask is not None:
            colors[np.where(~coat_mask)[0], 3] = 0.0
        colors = colors.flatten().tolist()
        return nvisii.mesh.create_from_data(
            name,
            positions=verts,
            normals=normals,
            colors=colors,
            color_dimensions=4,
            indices=faces,
        )
    else:
        return nvisii.mesh.create_from_data(
            name, positions=verts, normals=normals, indices=faces
        )


def trimesh_select_by_index(mesh: tm.Trimesh, vertex_indices: np.ndarray):
    """
    Extract a sub-mesh from a given mesh using the specified vertex indices.

    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.
    - vertex_indices (np.ndarray): Array of vertex indices to extract.

    Returns:
    - trimesh.Trimesh: The extracted sub-mesh.
    - dict: Mapping from new vertex indices to original indices.
    """
    # Convert indices to a set for fast lookup
    vertex_set = set(vertex_indices)

    # Find faces that are entirely contained within the selected vertices
    mask = np.all(np.isin(mesh.faces, vertex_indices), axis=1)
    sub_faces = mesh.faces[mask]

    # Get the unique vertex indices used in the selected faces
    unique_vertices, inverse_indices = np.unique(sub_faces, return_inverse=True)

    # Create the new vertex mapping (new index -> original index)
    index_mapping = unique_vertices

    # Create the new vertex positions
    sub_vertices = mesh.vertices[unique_vertices]

    # Remap the faces to the new vertex indexing
    remapped_faces = inverse_indices.reshape(-1, 3)

    # Create the new sub-mesh
    sub_mesh = tm.Trimesh(vertices=sub_vertices, faces=remapped_faces, process=False)

    return sub_mesh, index_mapping


def trimesh_select_by_index_old(mesh: tm.Trimesh, indices: np.ndarray):
    all_verts = set(range(len(mesh.vertices)))
    unwanted_verts = all_verts - set(indices)

    mask = np.ones(len(mesh.faces), dtype=bool)
    for unwanted_idx in unwanted_verts:
        mask &= ~(mesh.faces == unwanted_idx).any(axis=1)

    submesh_faces = mesh.faces[mask]
    new_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}
    new_faces = np.vectorize(new_indices_map.get)(submesh_faces)

    # Extract the vertex positions for the new submesh
    submesh_vertices = mesh.vertices[indices]
    submesh = tm.Trimesh(vertices=submesh_vertices, faces=new_faces)

    return submesh


def plot_point_cloud(point_cloud: np.ndarray, color="red"):
    return go.Scatter3d(
        x=point_cloud[:, 0],
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode="markers",
        marker=dict(size=2, color=color, colorscale="Viridis"),
    )
