import open3d as o3d
import time
import os
import glob
import re  # For natural sorting
import sys
import cv2
from datetime import datetime
import numpy as np


# Use case: python examples/viz/viz_frames.py /path/to/run/dir
folder_path = sys.argv[1]
FPS = 25


def find_ply_files(folder):
    """Finds and sorts ply files numerically based on the number in the filename."""
    pattern = os.path.join(folder, "frame-*.ply")
    files = glob.glob(pattern)

    def sort_key(filepath):
        filename = os.path.basename(filepath)
        # Extract number using regex (handles varying digits)
        match = re.search(r"frame-(\d+)\.ply", filename)
        if match:
            return int(match.group(1))
        return float("inf")  # Put files that don't match at the end

    files.sort(key=sort_key)
    if not files:
        print(f"Error: No files matching frame-*.ply found in folder: {folder}")
    return files


def load_mesh(filepath):
    """Loads a mesh, computes normals if needed, and returns it."""
    try:
        mesh = o3d.io.read_triangle_mesh(filepath)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        # Optional: Color the mesh if it doesn't have colors
        if not mesh.has_vertex_colors():
            mesh.paint_uniform_color([0.7, 0.7, 0.7])  # Grey color
        return mesh
    except Exception as e:
        print(f"Error loading mesh {filepath}: {e}")
        return None


def main():
    # ply_files = find_ply_files(os.path.join(folder_path))
    ply_files = find_ply_files(os.path.join(folder_path))
    if not ply_files:
        return

    print(f"Found {len(ply_files)} PLY files.")
    print("Press [SPACE] to Play/Pause animation.")
    print("Press [LEFT ARROW] for previous frame.")
    print("Press [RIGHT ARROW] for next frame.")
    print("Press [Q] or [ESC] to quit.")

    # --- Global state for callbacks ---
    global current_frame_index, paused, last_update_time, current_mesh_geometry
    current_frame_index = 0
    paused = True  # Start paused
    last_update_time = time.time()
    current_mesh_geometry = load_mesh(ply_files[current_frame_index])

    if not current_mesh_geometry:
        print("Error: Could not load the initial mesh.")
        return

    # --- Setup Visualizer ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Animation: {folder_path}")

    # Add initial geometry and axes
    vis.add_geometry(current_mesh_geometry)
    world_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(world_axes)

    # Configure rendering options
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = True  # Show edges
    render_option.mesh_show_back_face = True  # Render back faces for wireframe visibility

    frame_delay = 1.0 / FPS

    # --- Global state for recording ---
    global recording, video_writer, video_filename
    recording = False
    video_writer = None
    video_filename = None

    def toggle_recording(vis_obj):
        """Start or stop recording the current perspective."""
        global recording, video_writer, video_filename

        if not recording:
            # Start recording
            video_filename = os.path.join(folder_path, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            print(f"Recording started: {video_filename}")

            # Get the current window size
            width, height = (
                vis_obj.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.width,
                vis_obj.get_view_control().convert_to_pinhole_camera_parameters().intrinsic.height,
            )

            # Initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
            video_writer = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
            recording = True
        else:
            # Stop recording
            print(f"Recording stopped: {video_filename}")
            recording = False
            if video_writer:
                video_writer.release()
                video_writer = None

    def capture_frame(vis_obj):
        """Capture the current frame and write it to the video."""
        global video_writer, recording

        if recording and video_writer:
            # Capture the current frame as an image
            image = np.asarray(vis_obj.capture_screen_float_buffer(do_render=True)) * 255
            image = image.astype(np.uint8)

            # Overlay the current PLY file name
            text = ply_files[current_frame_index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (255, 0, 0)  # White
            text_position = (10, 30)  # Top-left corner
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            cv2.putText(
                image_bgr,
                text,
                text_position,
                font,
                font_scale,
                text_color,
                font_thickness,
            )

            # Write the frame to the video
            video_writer.write(image_bgr)

    # Register the ENTER key callback for toggling recording
    vis.register_key_callback(257, toggle_recording)  # ENTER key

    # --- Key Callback Functions ---
    def update_mesh_display(vis_obj, direction):
        global current_frame_index, current_mesh_geometry, paused

        new_index = current_frame_index + direction
        # Loop around
        new_index %= len(ply_files)

        if new_index != current_frame_index:
            new_mesh = load_mesh(ply_files[new_index])
            if new_mesh:
                # Remove old mesh, add new mesh
                vis_obj.remove_geometry(current_mesh_geometry, reset_bounding_box=False)
                vis_obj.add_geometry(new_mesh, reset_bounding_box=False)  # Add without resetting view unless first frame

                current_mesh_geometry = new_mesh
                current_frame_index = new_index
                print(f"PLY Animation: {os.path.basename(ply_files[current_frame_index])} {'[PAUSED]' if paused else ''}")
                return True  # Indicate update occurred
        return False  # Indicate no update

    def toggle_pause(vis_obj):
        global paused, last_update_time
        paused = not paused
        last_update_time = time.time()  # Reset timer when pausing/unpausing
        title_suffix = " [PAUSED]" if paused else ""
        print("Animation Paused" if paused else "Animation Resumed")
        print(f"PLY Animation: {os.path.basename(ply_files[current_frame_index])}{title_suffix}")

    def next_frame(vis_obj):
        global paused, last_update_time
        if not paused:  # Pause if playing
            toggle_pause(vis_obj)
        if update_mesh_display(vis_obj, 1):
            vis_obj.update_geometry(current_mesh_geometry)  # Signal geometry update
            vis_obj.poll_events()
            vis_obj.update_renderer()
        last_update_time = time.time()  # Reset timer on manual step

    def prev_frame(vis_obj):
        global paused, last_update_time
        if not paused:  # Pause if playing
            toggle_pause(vis_obj)
        if update_mesh_display(vis_obj, -1):
            vis_obj.update_geometry(current_mesh_geometry)  # Signal geometry update
            vis_obj.poll_events()
            vis_obj.update_renderer()
        last_update_time = time.time()  # Reset timer on manual step

    def close_window(vis_obj):
        print("Closing window.")
        vis_obj.destroy_window()
        # A bit of a workaround to signal the main loop to exit
        # Alternatively, use a global 'running' flag modified here
        global paused
        paused = True  # Stop animation processing
        # Setting a specific index or flag could also work if needed
        global current_frame_index
        current_frame_index = -999  # Signal exit condition for the run loop

    # Register key callbacks (GLFW key codes)
    # https://www.glfw.org/docs/latest/group__keys.html
    vis.register_key_callback(32, toggle_pause)  # Space bar
    vis.register_key_callback(262, next_frame)  # Right arrow
    vis.register_key_callback(263, prev_frame)  # Left arrow
    vis.register_key_callback(81, close_window)  # Q key
    vis.register_key_callback(256, close_window)  # Escape key

    # Modify the animation_step to capture frames if recording
    def animation_step(vis_obj):
        global paused, last_update_time, current_frame_index

        # Exit check based on close_window callback modification
        if current_frame_index == -999:
            return False  # Stop the idle loop

        current_time = time.time()
        if not paused and (current_time - last_update_time) >= frame_delay:
            if update_mesh_display(vis_obj, 1):
                vis_obj.update_geometry(current_mesh_geometry)  # Ensure geometry update is signaled
            last_update_time = current_time

        # Capture the current frame if recording
        capture_frame(vis_obj)

        return True  # Keep the idle task running

    # Register the idle callback for animation
    vis.register_animation_callback(animation_step)

    # --- Run the visualizer ---
    print("Starting visualization...")
    # Set initial title correctly
    title_suffix = " [PAUSED]" if paused else ""
    print(f"PLY Animation: {os.path.basename(ply_files[current_frame_index])}{title_suffix}")

    vis.run()  # This enters the main event loop

    # Cleanup happens implicitly when vis.run() exits after destroy_window()
    print("Visualization finished.")


if __name__ == "__main__":
    main()
