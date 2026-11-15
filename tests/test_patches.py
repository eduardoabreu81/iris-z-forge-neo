# Iris-Z support for Forge Neo
# The loader hooks with Forge Neo's modules stubbed: detection answers only Iris-Z, the FLUX.2 branch builds the
# Iris-Z DiT only for the Iris-Z guess and is restored afterwards, and apply registers everything once.

import gzip
import importlib
import json
import os
import sys
import types
import unittest
from unittest import mock

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "iris-z-small_bf16_header.json.gz")


def module(name, **attributes):
    result = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(result, key, value)
    return result


class Flux2K9B:
    huggingface_repo = "black-forest-labs/FLUX.2-klein-9B"
    unet_config = {"image_model": "flux2", "hidden_size": 4096}
    memory_usage_factor = 19.5

    def __init__(self, unet_config=None):
        self.unet_config = dict(unet_config or {})


class IntegratedFlux:
    pass


def forge_stubs(calls):
    def load_huggingface_component(guess, component_name, lib_name, cls_name, repo_path, state_dict):
        # what Forge's FLUX.2 branch does: import the class at call time and build it
        from backend.nn.flux import IntegratedFluxTransformer2DModel

        calls.append((cls_name, IntegratedFluxTransformer2DModel))
        return IntegratedFluxTransformer2DModel

    flux = module("backend.nn.flux", IntegratedFluxTransformer2DModel=IntegratedFlux)
    loader = module("backend.loader", load_huggingface_component=load_huggingface_component,
                    possible_models=("Flux2",))
    model_list = module("huggingface_guess.model_list", Flux2K9B=Flux2K9B, models=[Flux2K9B])
    detection = module("huggingface_guess.detection", detect_unet_config=lambda sd, prefix, *a, **k: {"forge": 1})
    flux2 = module("backend.diffusion_engine.flux2", Flux2=type("Flux2", (), {"__init__": lambda self, *a, **k: None}))
    base = module("backend.diffusion_engine.base", ForgeDiffusionEngine=object)
    backend = module("backend", loader=loader, memory_management=module("backend.memory_management"),
                     args=module("backend.args", dynamic_args=types.SimpleNamespace(ref_latents=[])))
    nn = module("backend.nn", flux=flux)
    backend.nn = nn
    return {
        "backend": backend, "backend.loader": loader, "backend.nn": nn, "backend.nn.flux": flux,
        "backend.memory_management": backend.memory_management, "backend.args": backend.args,
        "backend.diffusion_engine": module("backend.diffusion_engine"), "backend.diffusion_engine.flux2": flux2,
        "backend.diffusion_engine.base": base,
        "huggingface_guess": module("huggingface_guess", model_list=model_list, detection=detection),
        "huggingface_guess.model_list": model_list, "huggingface_guess.detection": detection,
    }


class TestPatches(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.stubs = forge_stubs(self.calls)
        self.modules = mock.patch.dict(sys.modules, self.stubs)
        self.modules.start()
        for name in [n for n in sys.modules if n.startswith("forge_irisz.native.") and n.split(".")[-1] in
                     ("patches", "engine", "model", "presets")]:
            del sys.modules[name]
        self.patches = importlib.import_module("forge_irisz.native.patches")
        with mock.patch.object(self.patches, "logger"), mock.patch("builtins.print"):
            self.patches.apply()

    def tearDown(self):
        self.modules.stop()
        for name in [n for n in sys.modules if n.startswith("forge_irisz.native.") and n.split(".")[-1] in
                     ("patches", "engine", "model", "presets")]:
            del sys.modules[name]

    def test_detection_answers_iris_z_only(self):
        with gzip.open(FIXTURE, "rt") as f:
            names = json.load(f)
        names.pop("__metadata__")
        detection = self.stubs["huggingface_guess.detection"]
        self.assertEqual(detection.detect_unet_config(names, "")["image_model"], "iris_z")
        del names["modality_in.weight"]
        self.assertEqual(detection.detect_unet_config(names, ""), {"forge": 1})

    def test_flux2_branch_builds_iris_z_for_iris_z(self):
        from forge_irisz.native.model import IrisZ
        from forge_irisz.native.transformer import IrisZDiT

        loader = self.stubs["backend.loader"]
        loader.load_huggingface_component(IrisZ({}), "transformer", "diffusers", "Flux2Transformer2DModel", "", {})
        loader.load_huggingface_component(Flux2K9B({}), "transformer", "diffusers", "Flux2Transformer2DModel", "", {})
        loader.load_huggingface_component(IrisZ({}), "vae", "diffusers", "AutoencoderKLFlux2", "", {})
        self.assertEqual([c[1] for c in self.calls], [IrisZDiT, IntegratedFlux, IntegratedFlux])
        self.assertIs(self.stubs["backend.nn.flux"].IntegratedFluxTransformer2DModel, IntegratedFlux)

    def test_registered_once(self):
        with mock.patch("builtins.print"):
            self.patches.apply()
        model_list = self.stubs["huggingface_guess.model_list"]
        self.assertEqual(sum(1 for m in model_list.models if m.__name__ == "IrisZ"), 1)
        self.assertEqual(len(self.stubs["backend.loader"].possible_models), 2)

    def test_guess(self):
        from forge_irisz.native.model import IrisZ

        self.assertEqual(IrisZ.huggingface_repo, "black-forest-labs/FLUX.2-klein-9B")
        self.assertEqual(IrisZ.memory_usage_factor, 39.0)
        self.assertEqual(IrisZ.unet_config, {"image_model": "iris_z"})


if __name__ == "__main__":
    unittest.main()
