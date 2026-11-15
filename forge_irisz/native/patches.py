# Iris-Z support for Forge Neo
# Runtime hooks into Forge Neo's loader. Every hook hands anything that is not Iris-Z to the original function
# untouched; Forge Neo itself stays unchanged. Follows the Iris 3B extension's patches.py.

import functools
import logging

import torch
from backend import loader
from backend.nn import flux
from huggingface_guess import detection, model_list

from . import detect
from .engine import IrisZEngine
from .model import IrisZ
from .transformer import IrisZDiT

logger = logging.getLogger("forge_irisz")

_applied = False


def _hook_detection() -> None:
    original = detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        dit_config = detect.detect(state_dict, key_prefix)
        if dit_config is not None:
            return dit_config
        return original(state_dict, key_prefix, *args, **kwargs)

    detection.detect_unet_config = detect_unet_config


def _hook_components() -> None:
    original = loader.load_huggingface_component

    # wraps leaves __wrapped__ behind, which inspect.getsource follows: the compat checks of the other extensions
    # that read Forge's loader source still reach it through this hook
    @functools.wraps(original)
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        if isinstance(guess, IrisZ) and cls_name == "Flux2Transformer2DModel":
            # Forge's FLUX.2 branch does the dtype, quantization and device handling; it imports
            # backend.nn.flux.IntegratedFluxTransformer2DModel inside the function, so the swap routes the build to us
            original_flux = flux.IntegratedFluxTransformer2DModel
            flux.IntegratedFluxTransformer2DModel = IrisZDiT
            try:
                return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)
            finally:
                flux.IntegratedFluxTransformer2DModel = original_flux
        return original(guess, component_name, lib_name, cls_name, repo_path, state_dict)

    loader.load_huggingface_component = load_huggingface_component


def apply() -> None:
    global _applied
    if _applied:
        return

    _hook_detection()
    if IrisZ not in model_list.models:
        model_list.models.append(IrisZ)
    if IrisZEngine not in loader.possible_models:
        loader.possible_models = (*loader.possible_models, IrisZEngine)
    _hook_components()
    try:
        from . import presets

        presets.register()
    except Exception as e:
        # the model still works: Euler, Flux2, 50 steps, CFG 4 by hand
        logger.warning(f"[Iris-Z] could not add the iris_z UI preset: {e}")

    _applied = True
    print(f"[Iris-Z] native backend enabled (torch {torch.__version__})")
