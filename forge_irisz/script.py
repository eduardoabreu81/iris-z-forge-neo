# Iris-Z support for Forge Neo
# The script's generation callbacks: the plan each Iris-Z generation runs (contracts.py) and Image to Depth.
#
# before_process runs before Forge loads the checkpoint, so p.sd_model there is still the previous model: the guards
# read the selected checkpoint's header instead.
#
# Image to Depth runs through Forge's own img2img pipeline, so model loading, LoRAs typed in the prompt, saving and
# infotext work as usual. As the SDK's image_to_depth: the image at its working size (512 or 1024), the empty
# prompt, no guidance, 4 Euler steps on the FLUX.2 schedule from pure noise (denoising 1), the image latent given
# next to the depth grid being sampled.
#
# Depth to Image runs through img2img too, as the SDK's depth_to_image: the depth map becomes the clean depth grid
# (the VAE encodes it, depth.paint_input), and the image is sampled from pure noise next to it with the user's
# prompt, steps and guidance, at about one megapixel.

import re

from . import checkpoints, depth
from .contracts import (
    DEPTH,
    IMAGE,
    PAINT,
    IrisZError,
    Plan,
    current_plan,
    seeded_noise,
    set_pending_error,
    set_plan,
)

MODE_IMAGE = "Image"
MODE_DEPTH = "Image to Depth"
MODE_PAINT = "Depth to Image"
MODES = (MODE_IMAGE, MODE_DEPTH, MODE_PAINT)
DEPTH_STEPS = 4
_EXTRA_NETWORK = re.compile(r"<[^<>]+>")

DEPTH_CHECKPOINT = "[Iris-Z] {mode} needs the Iris-Z checkpoint (iris-z-small) selected."


def selected_is_iris_z(p) -> bool:
    """Whether the checkpoint Forge is about to load (override included) is an Iris-Z."""
    from modules import sd_models, shared

    name = getattr(p, "override_settings", {}).get("sd_model_checkpoint", shared.opts.sd_model_checkpoint)
    info = sd_models.get_closet_checkpoint_match(name) if name else None
    if info is None:
        # nothing to read: the model in memory is the one Forge will keep
        return getattr(p.sd_model, "is_iris_z", False)
    return checkpoints.file_is_iris_z(info.filename)


def is_img2img(p) -> bool:
    # Forge Neo's processing objects have no is_img2img (A1111's); only img2img carries init_images
    return getattr(p, "init_images", None) is not None


def _refuse(message: str) -> None:
    error = IrisZError(message)
    set_pending_error(error)
    print(error)


def before_process(p, mode: str = MODE_IMAGE) -> None:
    # a refusal or a plan from an earlier generation must not reach this one
    set_pending_error(None)
    set_plan(None)
    p.iris_z_task = None
    task = mode if is_img2img(p) and mode in (MODE_DEPTH, MODE_PAINT) else None
    if not selected_is_iris_z(p):
        if task:
            _refuse(DEPTH_CHECKPOINT.format(mode=task))
        return
    if task == MODE_DEPTH:
        return _prepare_depth(p)
    if task == MODE_PAINT:
        return _prepare_paint(p)
    set_plan(Plan(IMAGE))


def _usable_input(p, mode) -> bool:
    if getattr(p, "image_mask", None) is not None:
        _refuse(f"[Iris-Z] {mode} works on the whole image: clear the inpaint mask.")
        return False
    if not getattr(p, "init_images", None):
        _refuse(f"[Iris-Z] {mode} needs an input image.")
        return False
    return True


def _from_pure_noise(p) -> None:
    """img2img sampling from pure noise: the input only conditions, through the plan."""
    p.resize_mode = 0
    p.denoising_strength = 1.0
    p.initial_noise_multiplier = 1.0
    # Forge's extra img2img noise would add to the noise sampling starts from; overrides apply after before_process
    p.override_settings = {**(getattr(p, "override_settings", None) or {}), "img2img_extra_noise": 0}


def _prepare_paint(p) -> None:
    if not _usable_input(p, MODE_PAINT):
        return
    given = depth.paint_input(p.init_images[0])
    p.init_images = [given]
    p.width, p.height = given.size
    _from_pure_noise(p)
    p.extra_generation_params["Iris-Z task"] = MODE_PAINT
    set_plan(Plan(PAINT))


def _prepare_depth(p) -> None:
    from PIL import Image

    if not _usable_input(p, MODE_DEPTH):
        return
    image = p.init_images[0].convert("RGB")
    size = depth.work_size(*image.size)
    p.init_images = [image.resize(size, Image.Resampling.LANCZOS)]
    p.width, p.height = size
    _from_pure_noise(p)
    # the depth is read with the empty prompt; LoRA tags stay for Forge to apply. Without tags the prompt is a space,
    # not "": the engine encodes both as the empty prompt, and sd-dynamic-prompts raises on an empty template
    p.prompt = " ".join(_EXTRA_NETWORK.findall(p.prompt or "")) or " "
    p.negative_prompt = ""
    p.cfg_scale = 1.0
    p.batch_size = 1
    p.steps = DEPTH_STEPS
    p.sampler_name, p.scheduler = "Euler", "Flux2"
    p.extra_generation_params["Iris-Z task"] = MODE_DEPTH
    p.iris_z_task = {"size": image.size}
    set_plan(Plan(DEPTH))


def before_sampling(p, noise, x=None) -> None:
    """The plan's other grid, once Forge has its own: the given image latent, or the depth grid's noise per seed."""
    if not getattr(p.sd_model, "is_iris_z", False):
        set_plan(None)
        return
    plan = current_plan()
    if plan is None:
        plan = Plan(IMAGE)
        set_plan(plan)
    if plan.mode in (DEPTH, PAINT):
        # img2img hands the clean input latent as x: the image for Image to Depth, the depth map for Depth to Image
        plan.companion = None if x is None else x.detach().clone()
        return
    plan.companion = seeded_noise(p.seeds, tuple(noise.shape[1:]))


def postprocess_image(p, pp) -> None:
    task = getattr(p, "iris_z_task", None)
    if task:
        pp.image = depth.depth_map(pp.image, task["size"])


def postprocess(p) -> None:
    set_plan(None)
