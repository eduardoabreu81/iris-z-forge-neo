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

import re

from . import checkpoints, depth
from .contracts import (
    DEPTH,
    IMAGE,
    IrisZError,
    Plan,
    current_plan,
    seeded_noise,
    set_pending_error,
    set_plan,
)

MODE_IMAGE = "Image"
MODE_DEPTH = "Image to Depth"
MODES = (MODE_IMAGE, MODE_DEPTH)
DEPTH_STEPS = 4
_EXTRA_NETWORK = re.compile(r"<[^<>]+>")

DEPTH_CHECKPOINT = "[Iris-Z] Image to Depth needs the Iris-Z checkpoint (iris-z-small) selected."


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
    wants_depth = is_img2img(p) and mode == MODE_DEPTH
    if not selected_is_iris_z(p):
        if wants_depth:
            _refuse(DEPTH_CHECKPOINT)
        return
    if wants_depth:
        return _prepare_depth(p)
    set_plan(Plan(IMAGE))


def _prepare_depth(p) -> None:
    from PIL import Image

    if getattr(p, "image_mask", None) is not None:
        return _refuse(f"[Iris-Z] {MODE_DEPTH} works on the whole image: clear the inpaint mask.")
    if not getattr(p, "init_images", None):
        return _refuse(f"[Iris-Z] {MODE_DEPTH} needs an input image.")
    image = p.init_images[0].convert("RGB")
    size = depth.work_size(*image.size)
    p.init_images = [image.resize(size, Image.Resampling.LANCZOS)]
    p.width, p.height = size
    p.resize_mode = 0
    # the depth is read with the empty prompt; LoRA tags stay for Forge to apply. Without tags the prompt is a space,
    # not "": the engine encodes both as the empty prompt, and sd-dynamic-prompts raises on an empty template
    p.prompt = " ".join(_EXTRA_NETWORK.findall(p.prompt or "")) or " "
    p.negative_prompt = ""
    p.cfg_scale = 1.0
    p.batch_size = 1
    p.steps = DEPTH_STEPS
    p.denoising_strength = 1.0
    p.initial_noise_multiplier = 1.0
    p.sampler_name, p.scheduler = "Euler", "Flux2"
    # Forge's extra img2img noise would add to the noise the depth starts from; overrides apply after before_process
    p.override_settings = {**(getattr(p, "override_settings", None) or {}), "img2img_extra_noise": 0}
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
    if plan.mode == DEPTH:
        # img2img hands the clean image latent as x
        plan.companion = None if x is None else x.detach().clone()
        return
    plan.companion = seeded_noise(p.seeds, tuple(noise.shape[1:]))


def postprocess_image(p, pp) -> None:
    task = getattr(p, "iris_z_task", None)
    if task:
        pp.image = depth.depth_map(pp.image, task["size"])


def postprocess(p) -> None:
    set_plan(None)
