import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import pyvista as pv
import trimesh as tm

from multiprocessing import Queue
import os
import re


def save_mesh(directory, mesh, timestep, limit=50):
    """
    Save a mesh file and maintain a folder with the latest `limit` files.

    Args:
        directory (str): Directory where the mesh files are saved.
        mesh (trimesh.Trimesh): The mesh object to save.
        timestep (int): The current time step.
        limit (int): Maximum number of files to keep in the folder.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the new mesh file
    file_name = f"frame-{timestep}.ply"
    file_path = os.path.join(directory, file_name)
    mesh.export(file_path)

    # Get the list of files and sort them by timestep
    if limit == -1:
        return
    files = [f for f in os.listdir(directory) if re.match(r"frame-\d+\.ply", f)]
    timesteps_files = [(int(re.search(r"\d+", f).group()), f) for f in files]
    timesteps_files.sort(key=lambda x: x[0])

    # Remove oldest files if the limit is exceeded
    while len(timesteps_files) > limit:
        oldest_timestep, oldest_file = timesteps_files.pop(0)
        os.remove(os.path.join(directory, oldest_file))


def save_pc(directory, pc, timestep, limit=50, name="pc"):
    """
    Save a point cloud file and maintain a folder with the latest `limit` files.

    Args:
        directory (str): Directory where the mesh files are saved.
        pc (np.ndarray): The point cloud to save, in shape [n_points, 3].
        timestep (int): The current time step.
        limit (int): Maximum number of files to keep in the folder.
    """
    if pc is None:
        return
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the new mesh file
    file_name = f"frame-{timestep}-{name}.ply"
    file_path = os.path.join(directory, file_name)
    o3d.io.write_point_cloud(file_path, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc)))

    # Get the list of files and sort them by timestep
    if limit == -1:
        return
    files = [f for f in os.listdir(directory) if re.match(r"frame-\d+-" + name + "\.ply", f)]
    timesteps_files = [(int(re.search(r"\d+", f).group()), f) for f in files]
    timesteps_files.sort(key=lambda x: x[0])

    # Remove oldest files if the limit is exceeded
    while len(timesteps_files) > limit:
        oldest_timestep, oldest_file = timesteps_files.pop(0)
        os.remove(os.path.join(directory, oldest_file))


def mesh_worker(queue: Queue, directory, limit=50):
    """
    Worker function to handle mesh saving in a multiprocessing context.

    Args:
        queue (Queue): The multiprocessing queue for incoming tasks.
        directory (str): Directory where the mesh files are saved.
        limit (int): Maximum number of files to keep in the folder.
    """
    while task := queue.get():  # Stop signal
        if task is None:
            break
        mesh, (markers, markers_rest), timestep = task
        save_mesh(directory, mesh, timestep, limit)
        save_pc(directory, markers, timestep, limit, "markers")
        save_pc(directory, markers_rest, timestep, limit, "markers_rest")
        del mesh


def pv_to_tm(pv_mesh: pv.UnstructuredGrid):
    verts = pv_mesh.points
    faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
    return tm.Trimesh(vertices=verts, faces=faces)


def o3d_to_nvisii_mesh(name: str, mesh: o3d.geometry.TriangleMesh, coat_mask=None, color=False):
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
        return nvisii.mesh.create_from_data(name, positions=verts, normals=normals, indices=faces)


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
        return nvisii.mesh.create_from_data(name, positions=verts, normals=normals, indices=faces)


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
