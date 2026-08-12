import numpy as np
import pandas as pd
import vispy
from vispy import scene


def build_color_lut(node_registry_df, unknown_node_color=(255, 51, 153)):
    rgba = node_registry_df[["red", "green", "blue", "alpha"]].to_numpy(np.float32)
    rgba[:, :3] /= 255.0
    node_ids = node_registry_df["node_id"].to_numpy(np.int32)
    param2 = node_registry_df["param2"].to_numpy(np.int32)
    unknown = np.array([*np.asarray(unknown_node_color, dtype=np.float32) / 255.0, 1.0])
    lut = np.tile(unknown, (int(node_ids.max()) + 1, int(param2.max()) + 1, 1))
    lut[node_ids, param2] = rgba

    def get_rgba(node_ids_params):
        ids = node_ids_params[:, 0].clip(0, lut.shape[0] - 1)
        params = node_ids_params[:, 1].clip(0, lut.shape[1] - 1)
        return lut[ids, params]

    return get_rgba


def plot_node_ids_batched(
    node_ids_params_batch,
    coords,
    get_rgba,
    ignore_node_ids=(126, 127),
    spherical=False,
    figsize=(512, 512),
    alpha=0.8,
    backend="sdl2",
):
    if node_ids_params_batch.ndim == 2:
        node_ids_params_batch = node_ids_params_batch[None]

    vispy.use(backend)
    canvas = scene.SceneCanvas(keys=None, show=False, size=figsize, bgcolor="white")
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(
        fov=45, distance=2.5, elevation=30, azimuth=-60
    )
    scatter = scene.visuals.Markers(alpha=alpha, spherical=spherical, antialias=0)
    view.add(scatter)

    ignored = np.asarray(ignore_node_ids, dtype=np.int32)
    images = []
    for node_ids_params in node_ids_params_batch:
        mask = ~np.isin(node_ids_params[:, 0], ignored)
        if mask.any():
            scatter.set_data(
                coords[mask],
                face_color=get_rgba(node_ids_params[mask])[:, :3],
                size=8,
                edge_width=0,
            )
        else:
            scatter.set_data(
                np.zeros((1, 3), dtype=np.float32),
                face_color=np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32),
                size=1,
                edge_width=0,
            )
        images.append(canvas.render())

    canvas.close()
    return np.stack(images)
