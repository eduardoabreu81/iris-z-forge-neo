# Iris-Z support for Forge Neo
# The DiT against the SDK's own transformer (tests/reference, unmodified) at toy sizes: the SDK's weights load
# strictly (same parameter names) and both give the same velocities for every grid layout the extension runs.

import unittest

import torch

from forge_irisz import contracts
from forge_irisz.native import transformer
from forge_irisz.native.transformer import GIVEN_RGB, IrisZDiT
from tests.reference import iris_z_sdk_model as sdk

TOY = dict(in_channels=8, context_in_dim=12, hidden_size=64, num_heads=2, depth=1, depth_single_blocks=2,
           axes_dim=(8, 8, 8, 8), theta=2000, mlp_ratio=3.0)


def toy_models(seed=0):
    torch.manual_seed(seed)
    params = sdk.Klein9BParams(in_channels=8, context_in_dim=12, hidden_size=64, num_heads=2, depth=1,
                               depth_single_blocks=2, axes_dim=[8, 8, 8, 8])
    reference = sdk.Flux2(params, num_modalities=4).eval()
    with torch.no_grad():
        for p in reference.parameters():
            p.copy_(torch.randn_like(p) * 0.2)
    ours = IrisZDiT(**TOY).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    return reference, ours


def sdk_ids(height, width, t, batch):
    ids = torch.cartesian_prod(torch.tensor([t]), torch.arange(height), torch.arange(width), torch.tensor([0]))
    return ids.float()[None].expand(batch, -1, -1)


def sdk_velocity(reference, rgb, depth, rgb_t, depth_t, main_t, modalities, context):
    batch, channels, height, width = rgb.shape
    n = height * width
    tokens = torch.cat([rgb.flatten(2).transpose(1, 2), depth.flatten(2).transpose(1, 2)], dim=1)
    ids = torch.cat([sdk_ids(height, width, 0, batch), sdk_ids(height, width, 1, batch)], dim=1)
    modality = torch.tensor([modalities[0]] * n + [modalities[1]] * n)[None].expand(batch, -1)
    ctx_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1),
                                   torch.arange(context.shape[1])).float()[None].expand(batch, -1, -1)
    out = reference(x=tokens, x_ids=ids, timesteps=main_t, ctx=context, ctx_ids=ctx_ids, guidance=None,
                    modality_ids=modality, rgb_timesteps=rgb_t, depth_timesteps=depth_t, depth_start=n)
    grid = out.transpose(1, 2).reshape(batch, channels, 2 * height, width)
    return grid[:, :, :height], grid[:, :, height:]


class TestTransformer(unittest.TestCase):
    def setUp(self):
        contracts.set_plan(None)
        self.reference, self.ours = toy_models()
        torch.manual_seed(1)
        self.rgb = torch.randn(2, 8, 3, 4)
        self.depth = torch.randn(2, 8, 3, 4)
        self.context = torch.randn(2, 5, 12)

    def tearDown(self):
        contracts.set_plan(None)

    def assertClose(self, a, b):
        self.assertEqual(a.shape, b.shape)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)

    def check(self, rgb_t, depth_t, main_t, modalities):
        with torch.no_grad():
            ours = self.ours.forward_grids(self.rgb, self.depth, rgb_t, depth_t, main_t, modalities, self.context)
            reference = sdk_velocity(self.reference, self.rgb, self.depth, rgb_t, depth_t, main_t, modalities,
                                     self.context)
        self.assertClose(ours[0], reference[0])
        self.assertClose(ours[1], reference[1])

    def test_image_with_noise_depth(self):
        t = torch.tensor([0.7, 0.3])
        self.check(t, torch.ones(2), t, (transformer.RGB, transformer.DEPTH))

    def test_depth_from_given_image(self):
        t = torch.tensor([0.9, 0.5])
        self.check(torch.zeros(2), t, t, (GIVEN_RGB, transformer.DEPTH))

    def test_modality_and_grid_levels_matter(self):
        # the comparison above would hold trivially if the extra conditioning did nothing
        t = torch.tensor([0.7, 0.3])
        with torch.no_grad():
            base = self.ours.forward_grids(self.rgb, self.depth, t, torch.ones(2), t, (0, 1), self.context)[0]
            given = self.ours.forward_grids(self.rgb, self.depth, t, torch.ones(2), t, (2, 1), self.context)[0]
            level = self.ours.forward_grids(self.rgb, self.depth, t, torch.zeros(2), t, (0, 1), self.context)[0]
        self.assertGreater((base - given).abs().max().item(), 1e-3)
        self.assertGreater((base - level).abs().max().item(), 1e-3)

    def test_forge_forward_image_mode(self):
        sigma = torch.tensor([0.6, 0.6])
        contracts.set_plan(contracts.Plan(contracts.IMAGE, self.depth))
        with torch.no_grad():
            out = self.ours(self.rgb, sigma, self.context)
            expected, _ = self.ours.forward_grids(self.rgb, self.depth, sigma, torch.ones(2), sigma,
                                                  (transformer.RGB, transformer.DEPTH), self.context)
        self.assertClose(out, expected)

    def test_forge_forward_depth_mode(self):
        sigma = torch.tensor([0.4, 0.4])
        contracts.set_plan(contracts.Plan(contracts.DEPTH, self.rgb[:1]))
        with torch.no_grad():
            out = self.ours(self.depth, sigma, self.context)
            _, expected = self.ours.forward_grids(self.rgb[:1].expand(2, -1, -1, -1), self.depth, torch.zeros(2),
                                                  sigma, sigma, (GIVEN_RGB, transformer.DEPTH), self.context)
        self.assertClose(out, expected)

    def test_cfg_rows_repeat_the_companion(self):
        # Forge stacks [cond rows, uncond rows]: two copies of a one-sample companion
        noise = torch.randn(1, 8, 3, 4)
        x = torch.randn(2, 8, 3, 4)
        self.assertTrue(torch.equal(contracts.companion_noise(contracts.Plan(contracts.IMAGE, noise), x)[1], noise[0]))

    def test_missing_noise_is_fixed(self):
        x = torch.randn(1, 8, 3, 4)
        a = contracts.companion_noise(None, x)
        b = contracts.companion_noise(contracts.Plan(contracts.IMAGE, torch.randn(1, 8, 5, 5)), x)
        self.assertTrue(torch.equal(a, b))

    def test_depth_without_image_raises(self):
        contracts.set_plan(contracts.Plan(contracts.DEPTH, None))
        with self.assertRaises(contracts.IrisZError):
            self.ours(self.depth, torch.tensor([0.5, 0.5]), self.context)


if __name__ == "__main__":
    unittest.main()
