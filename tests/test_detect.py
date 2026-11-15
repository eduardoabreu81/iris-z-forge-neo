# Iris-Z support for Forge Neo
# Detection on the real Iris-Z-small header (tests/fixtures, names and shapes only), and the round trip Forge runs:
# detect -> IrisZDiT(**config) -> load_state_dict, strict, at the toy sizes of the same layout.

import gzip
import json
import os
import tempfile
import unittest

import torch

from forge_irisz import checkpoints
from forge_irisz.native import detect
from forge_irisz.native.transformer import IrisZDiT
from tests.test_transformer import TOY

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "iris-z-small_bf16_header.json.gz")


def header():
    with gzip.open(FIXTURE, "rt") as f:
        result = json.load(f)
    result.pop("__metadata__")
    return result


class TestDetect(unittest.TestCase):
    def test_iris_z_small(self):
        config = detect.detect(header())
        self.assertEqual(config, {
            "image_model": "iris_z", "in_channels": 128, "hidden_size": 4096, "num_heads": 32,
            "axes_dim": [32, 32, 32, 32], "theta": 2000, "mlp_ratio": 3.0, "depth": 8, "depth_single_blocks": 24,
            "context_in_dim": 12288, "num_modalities": 4})

    def test_plain_klein_is_not_iris_z(self):
        names = header()
        del names["modality_in.weight"]
        self.assertIsNone(detect.detect(names))
        self.assertFalse(checkpoints.is_iris_z(names))

    def test_prefixed_checkpoint(self):
        names = {f"model.diffusion_model.{k}": v for k, v in header().items()}
        self.assertTrue(checkpoints.is_iris_z(names))
        self.assertEqual(detect.detect(names, "model.diffusion_model.")["depth"], 8)

    def test_big_is_refused(self):
        names = header()
        names["final_layer.adaLN_modulation.1.weight"] = {"shape": [12288, 6144]}
        with self.assertRaises(ValueError):
            detect.detect(names)
        names = header()
        names["guidance_in.in_layer.weight"] = {"shape": [4096, 256]}
        with self.assertRaises(ValueError):
            detect.detect(names)

    def test_build_round_trip(self):
        source = IrisZDiT(**TOY)
        config = detect.detect(source.state_dict())
        config.pop("image_model")
        built = IrisZDiT(**config)
        built.load_state_dict(source.state_dict(), strict=True)
        self.assertEqual(config["num_modalities"], 4)
        self.assertEqual(config["mlp_ratio"], 3.0)

    def test_file_header(self):
        from safetensors.torch import save_file

        source = IrisZDiT(**TOY)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "toy.safetensors")
            save_file({k: v.contiguous() for k, v in source.state_dict().items()}, path)
            self.assertTrue(checkpoints.file_is_iris_z(path))
            other = os.path.join(folder, "other.safetensors")
            save_file({"x.weight": torch.zeros(2)}, other)
            self.assertFalse(checkpoints.file_is_iris_z(other))


if __name__ == "__main__":
    unittest.main()
