# Iris-Z support for Forge Neo
# The Iris-Z diffusion engine: Forge's FLUX.2 [klein] engine (text encoder, VAE, predictor) without Klein's reference
# images. The DiT places its second grid itself, and an img2img input is the image Image to Depth reads, never a
# reference to edit from.

import torch
from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine
from backend.diffusion_engine.flux2 import Flux2

from ..contracts import raise_pending_error
from .model import IrisZ


class IrisZEngine(Flux2):
    matched_guesses = [IrisZ]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_iris_z = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt):
        raise_pending_error()
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        dynamic_args.ref_latents.clear()
        # the SDK encodes a blank prompt as the empty one; the script passes " " where Forge needs a non-empty string
        return self.text_processing_engine_qwen(["" if not text.strip() else text for text in prompt])

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        return ForgeDiffusionEngine.encode_first_stage(self, x)
