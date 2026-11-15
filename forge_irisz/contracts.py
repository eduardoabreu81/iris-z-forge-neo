# Iris-Z support for Forge Neo
# State shared by the script callbacks and the DiT, which Forge calls directly.
#
# The error mechanism: Forge swallows exceptions raised in script callbacks, so an error set in before_process is
# re-raised from inside the model and the generation dies with the right message instead of a wrong image.
#
# The plan: an Iris-Z sequence always holds two grids, and Forge's sampler only knows the one it samples. The plan
# carries the other: for an image, the depth grid as fixed pure noise (one per seed, as the SDK draws it next to the
# image noise and never denoises it); for Image to Depth, the clean image latent the depth is read from; for Depth
# to Image, the clean depth latent the image is painted over.

from dataclasses import dataclass

IMAGE = "image"
DEPTH = "depth"
PAINT = "paint"

_pending_error = None
_plan = None
_default_noise = {}


class IrisZError(RuntimeError):
    """A message the user can act on."""


def set_pending_error(error):
    global _pending_error
    _pending_error = error


def raise_pending_error():
    if _pending_error is not None:
        raise _pending_error


@dataclass
class Plan:
    mode: str = IMAGE
    # IMAGE: the depth grid's noise [B, C, h, w] (CPU); DEPTH / PAINT: the given image / depth latent [1 or B, C, h, w]
    companion: object = None


def set_plan(plan):
    global _plan
    _plan = plan


def current_plan():
    return _plan


def rows(tensor, x):
    """tensor on x's device and dtype with x's batch: Forge stacks cond and uncond rows of the same samples, so
    the batch is a whole number of copies of the samples."""
    tensor = tensor.to(device=x.device, dtype=x.dtype)
    count = x.shape[0]
    if tensor.shape[0] == count:
        return tensor
    if count % tensor.shape[0] == 0:
        return tensor.repeat(count // tensor.shape[0], 1, 1, 1)
    return tensor[:1].expand(count, -1, -1, -1)


def companion_noise(plan, x):
    """The depth grid next to the image being sampled: the plan's noise, or a fixed draw when there is none for
    this shape (an API call without the script, a hires pass at another size)."""
    noise = getattr(plan, "companion", None) if plan is not None and plan.mode == IMAGE else None
    if noise is None or tuple(noise.shape[1:]) != tuple(x.shape[1:]):
        noise = default_noise(tuple(x.shape[1:]))
    return rows(noise, x)


def default_noise(shape):
    import torch

    if shape not in _default_noise:
        _default_noise.clear()
        _default_noise[shape] = torch.randn((1, *shape), generator=torch.Generator().manual_seed(0))
    return _default_noise[shape]


def seeded_noise(seeds, shape):
    """One depth-grid noise per seed, [len(seeds), *shape], drawn on CPU so a seed gives the same image anywhere."""
    import torch

    return torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(int(seed) + DEPTH_SEED_OFFSET))
                        for seed in seeds])


# the depth grid is a second noise per seed; offset from the image's own seed so the two draws differ
DEPTH_SEED_OFFSET = 1_000_003


def given_rgb(plan, x):
    given = plan.companion
    if given is None or tuple(given.shape[1:]) != tuple(x.shape[1:]):
        raise IrisZError("[Iris-Z] Image to Depth lost its input image: run it from the img2img tab with an image.")
    return rows(given, x)


def given_depth(plan, x):
    given = plan.companion
    if given is None or tuple(given.shape[1:]) != tuple(x.shape[1:]):
        raise IrisZError("[Iris-Z] Depth to Image lost its depth map: run it from the img2img tab with a depth map.")
    return rows(given, x)
