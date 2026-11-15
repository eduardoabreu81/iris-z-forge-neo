# Iris-Z support for Forge Neo
# The generation callbacks with Forge's processing object stubbed: the plan each generation runs, Image to Depth's
# settings and output, and the sizes the SDK works at.

import sys
import types
import unittest
from unittest import mock

import torch
from PIL import Image

from forge_irisz import contracts, depth, script


def processing(img2img=False, **fields):
    p = types.SimpleNamespace(prompt="a lighthouse <lora:detail:0.5>", negative_prompt="blurry", cfg_scale=4.0,
                              batch_size=2, steps=50, denoising_strength=0.75, sampler_name="DPM++ 2M",
                              scheduler="Beta", width=1024, height=1024, resize_mode=1, override_settings={},
                              extra_generation_params={}, seeds=[11, 12], image_mask=None,
                              sd_model=types.SimpleNamespace(is_iris_z=True))
    if img2img:
        p.init_images = [Image.new("RGB", (1200, 800), (90, 120, 200))]
    for key, value in fields.items():
        setattr(p, key, value)
    return p


class TestScript(unittest.TestCase):
    def setUp(self):
        contracts.set_plan(None)
        contracts.set_pending_error(None)
        self.selected = mock.patch.object(script, "selected_is_iris_z", return_value=True)
        self.selected.start()

    def tearDown(self):
        self.selected.stop()
        contracts.set_plan(None)
        contracts.set_pending_error(None)

    def test_txt2img_plans_the_image(self):
        p = processing()
        script.before_process(p)
        self.assertEqual(contracts.current_plan().mode, contracts.IMAGE)
        noise = torch.randn(2, 128, 64, 64)
        script.before_sampling(p, noise, noise)
        companion = contracts.current_plan().companion
        self.assertEqual(tuple(companion.shape), (2, 128, 64, 64))
        # one draw per seed, the same draw for the same seed
        again = contracts.seeded_noise([12], (128, 64, 64))
        self.assertTrue(torch.equal(companion[1], again[0]))
        self.assertFalse(torch.equal(companion[0], companion[1]))

    def test_other_models_get_no_plan(self):
        with mock.patch.object(script, "selected_is_iris_z", return_value=False):
            script.before_process(processing())
        self.assertIsNone(contracts.current_plan())
        p = processing(sd_model=types.SimpleNamespace())
        contracts.set_plan(contracts.Plan())
        script.before_sampling(p, torch.zeros(1, 4, 8, 8))
        self.assertIsNone(contracts.current_plan())

    def test_depth_settings(self):
        p = processing(img2img=True)
        script.before_process(p, script.MODE_DEPTH)
        self.assertEqual(contracts.current_plan().mode, contracts.DEPTH)
        self.assertEqual((p.width, p.height), p.init_images[0].size)
        self.assertEqual(p.init_images[0].size, depth.work_size(1200, 800))
        self.assertEqual((p.steps, p.cfg_scale, p.denoising_strength, p.batch_size), (4, 1.0, 1.0, 1))
        self.assertEqual((p.sampler_name, p.scheduler, p.resize_mode), ("Euler", "Flux2", 0))
        self.assertEqual(p.prompt, "<lora:detail:0.5>")
        self.assertEqual(p.override_settings["img2img_extra_noise"], 0)
        latent = torch.randn(1, 128, 30, 46)
        script.before_sampling(p, torch.randn_like(latent), latent)
        self.assertTrue(torch.equal(contracts.current_plan().companion, latent))

    def test_depth_prompt_without_tags_is_blank(self):
        p = processing(img2img=True, prompt="a lighthouse")
        script.before_process(p, script.MODE_DEPTH)
        self.assertEqual(p.prompt, " ")

    def test_depth_refusals(self):
        with mock.patch.object(script, "selected_is_iris_z", return_value=False):
            script.before_process(processing(img2img=True), script.MODE_DEPTH)
        with self.assertRaises(contracts.IrisZError):
            contracts.raise_pending_error()
        contracts.set_pending_error(None)
        script.before_process(processing(img2img=True, image_mask=Image.new("L", (8, 8))), script.MODE_DEPTH)
        with self.assertRaises(contracts.IrisZError):
            contracts.raise_pending_error()

    def test_depth_task_only_in_img2img(self):
        p = processing()
        script.before_process(p, script.MODE_DEPTH)
        self.assertEqual(contracts.current_plan().mode, contracts.IMAGE)
        self.assertEqual(p.steps, 50)

    def test_depth_output(self):
        p = processing(img2img=True)
        script.before_process(p, script.MODE_DEPTH)
        decoded = Image.new("RGB", p.init_images[0].size, (40, 40, 40))  # near: dark in the decoded grid
        pp = types.SimpleNamespace(image=decoded)
        script.postprocess_image(p, pp)
        self.assertEqual(pp.image.mode, "L")
        self.assertEqual(pp.image.size, (1200, 800))
        self.assertEqual(pp.image.getpixel((5, 5)), 215)

    def test_paint_settings(self):
        p = processing(img2img=True)
        script.before_process(p, script.MODE_PAINT)
        self.assertEqual(contracts.current_plan().mode, contracts.PAINT)
        # about one megapixel at the map's aspect ratio, the user's prompt, steps and guidance kept
        self.assertEqual(p.init_images[0].size, depth.enlarged_size(1200, 800))
        self.assertEqual((p.width, p.height), p.init_images[0].size)
        self.assertEqual((p.prompt, p.steps, p.cfg_scale, p.batch_size), ("a lighthouse <lora:detail:0.5>", 50, 4.0, 2))
        self.assertEqual((p.denoising_strength, p.resize_mode), (1.0, 0))
        self.assertEqual(p.override_settings["img2img_extra_noise"], 0)
        self.assertEqual(p.extra_generation_params["Iris-Z task"], script.MODE_PAINT)
        latent = torch.randn(1, 128, 30, 46)
        script.before_sampling(p, torch.randn_like(latent), latent)
        self.assertTrue(torch.equal(contracts.current_plan().companion, latent))
        # the output is the painted picture itself
        pp = types.SimpleNamespace(image=Image.new("RGB", (8, 8), (10, 200, 30)))
        script.postprocess_image(p, pp)
        self.assertEqual(pp.image.getpixel((1, 1)), (10, 200, 30))

    def test_paint_refusals(self):
        with mock.patch.object(script, "selected_is_iris_z", return_value=False):
            script.before_process(processing(img2img=True), script.MODE_PAINT)
        with self.assertRaises(contracts.IrisZError) as raised:
            contracts.raise_pending_error()
        self.assertIn("Depth to Image", str(raised.exception))

    def test_paint_input_inverts_our_depth_maps(self):
        # a map from Image to Depth (near = bright, already spread over the range) comes back as the encoding
        ramp = Image.fromarray(torch.linspace(0, 255, 1024).round().to(torch.uint8).repeat(1024, 1).numpy(), "L")
        given = depth.paint_input(ramp)
        self.assertEqual(given.mode, "RGB")
        self.assertEqual(given.size, (1024, 1024))
        r, g, b = given.getpixel((1023, 10))
        self.assertTrue(r == g == b)
        self.assertLess(r, 8)  # nearest: the encoding's -1
        self.assertGreater(given.getpixel((0, 10))[0], 247)  # farthest: +1

    def test_paint_input_stretches_a_narrow_map(self):
        narrow = Image.new("L", (640, 480), 100)
        narrow.paste(140, (0, 0, 320, 480))
        given = depth.paint_input(narrow)
        values = sorted({given.getpixel((x, 100))[0] for x in (10, 600)})
        self.assertEqual(values, [0, 255])

    def test_selected_checkpoint(self):
        shared = types.SimpleNamespace(opts=types.SimpleNamespace(sd_model_checkpoint="iris-z"))
        sd_models = types.SimpleNamespace(get_closet_checkpoint_match=lambda name: types.SimpleNamespace(
            filename=f"/models/{name}.safetensors"))
        modules = types.ModuleType("modules")
        modules.shared, modules.sd_models = shared, sd_models
        self.selected.stop()
        try:
            with mock.patch.dict(sys.modules, {"modules": modules}), \
                    mock.patch.object(script.checkpoints, "file_is_iris_z", side_effect=lambda f: "iris" in f):
                self.assertTrue(script.selected_is_iris_z(processing()))
                self.assertFalse(script.selected_is_iris_z(processing(override_settings={
                    "sd_model_checkpoint": "klein"})))
        finally:
            self.selected.start()


class TestDepthSizes(unittest.TestCase):
    def test_sdk_working_sizes(self):
        # the SDK's choose_resolution gives the same sizes (checked against iris_z/depth.py)
        self.assertEqual(depth.work_size(640, 480), (576, 432))  # under 0.5 MP: about 512²
        self.assertEqual(depth.work_size(1200, 800), (1200, 800))  # about 1024², never above the source
        self.assertEqual(depth.work_size(512, 768), (416, 624))
        width, height = depth.work_size(4000, 3000)
        self.assertEqual((width % 16, height % 16), (0, 0))
        self.assertAlmostEqual(width / height, 4 / 3, delta=0.02)
        self.assertLess(abs(width * height - 1024 ** 2), 0.2 * 1024 ** 2)

    def test_never_larger_than_the_source(self):
        width, height = depth.work_size(300, 200)
        self.assertLessEqual(width, 300)
        self.assertLessEqual(height, 200)


if __name__ == "__main__":
    unittest.main()
