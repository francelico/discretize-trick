from importlib import import_module

_MODEL_MODULES = {
    "ResNet3dDecoder": ".vae_voxel",
    "ResNet3dEncoder": ".vae_voxel",
    "ResNetDecoderArgs": ".vae_voxel",
    "ResNetEncoderArgs": ".vae_voxel",
    "VoxelDiT": ".dit_voxel",
    "VoxelDitDenoiserArgs": ".dit_voxel",
}


def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_MODEL_MODULES[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "ResNet3dDecoder",
    "ResNet3dEncoder",
    "ResNetDecoderArgs",
    "ResNetEncoderArgs",
    "VoxelDiT",
    "VoxelDitDenoiserArgs",
]
