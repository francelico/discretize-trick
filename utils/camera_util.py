import torch
import torch.nn.functional as F
import utils3d
from einops import rearrange


def project_voxel_local_to_worldcam(camera):
    batch, timesteps = camera.shape[:2]
    camera = rearrange(camera, "b t d -> (b t) d")
    rotation = rotation_6d_to_matrix(camera[..., :6])
    camera[..., -4:-1] = local_to_worldcam(camera[..., -4:-1], rotation)
    return rearrange(camera, "(b t) d -> b t d", b=batch, t=timesteps)


def local_to_worldcam(position, rotation):
    return -torch.matmul(rotation, position[..., None])[..., 0]


def camera_params_to_matrices(
    camera_params: torch.Tensor,
    image_width: int = 640,
    image_height: int = 360,
    near: float = 0.001,
    far: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = camera_params.device
    image_width = torch.tensor(image_width, device=device)
    image_height = torch.tensor(image_height, device=device)
    near = torch.tensor(near, device=device)
    far = torch.tensor(far, device=device)

    if camera_params.shape[-1] == 8:
        quaternion = camera_params[..., :4]
        translation = camera_params[..., 4:7]
        fov = camera_params[..., 7]
        quaternion = torch.stack(
            [quaternion[..., 3], quaternion[..., 0], quaternion[..., 1], quaternion[..., 2]],
            dim=-1,
        )
        rotation = utils3d.torch.quaternion_to_matrix(quaternion)
    elif camera_params.shape[-1] == 10:
        rotation = rotation_6d_to_matrix(camera_params[..., :6])
        translation = camera_params[..., 6:9]
        fov = camera_params[..., 9]
    else:
        raise ValueError(
            f"camera_params must have 8 or 10 dimensions, got {camera_params.shape[-1]}"
        )

    extrinsics = utils3d.torch.to4x4(rotation, translation)
    view = utils3d.torch.extrinsics_to_view(extrinsics)
    projection = utils3d.torch.perspective_from_fov(
        fov=fov,
        width=image_width,
        height=image_height,
        near=near,
        far=far,
    )
    return view, projection


def rotation_6d_to_matrix(rotation_6d: torch.Tensor) -> torch.Tensor:
    a1, a2 = rotation_6d[..., :3], rotation_6d[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def compute_dpitch_dyaw_from_cam_dir(cam_dir):
    direction = cam_dir / cam_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pitch = torch.rad2deg(
        torch.atan2(
            direction[..., 2],
            torch.sqrt(direction[..., 0] ** 2 + direction[..., 1] ** 2),
        )
    )
    yaw = torch.rad2deg(torch.atan2(direction[..., 1], direction[..., 0]))
    dpitch = pitch[..., 1:] - pitch[..., :-1]
    dyaw = (yaw[..., 1:] - yaw[..., :-1] + 180.0) % 360.0 - 180.0
    return dpitch.numpy(), dyaw.numpy()
