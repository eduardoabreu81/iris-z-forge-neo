# Iris-Z support for Forge Neo
# Entry point: Forge Neo imports every scripts/*.py at startup, before the UI and before any model loads, which is
# when the model gets registered. Everything runs in the module scope; any failure leaves Forge Neo unchanged and
# says so in the console.

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENABLED = False

try:
    from forge_irisz.native import compat

    _problems = compat.check()
    if _problems:
        compat.report(_problems)
    else:
        from forge_irisz.native import patches

        patches.apply()
        ENABLED = True
except Exception:
    print("[Iris-Z] failed to enable, Forge Neo is unchanged:")
    traceback.print_exc()


if ENABLED:
    from modules import scripts

    from forge_irisz import script as irisz_script

    class IrisZScript(scripts.Script):
        sorting_priority = 10
        create_group = False

        def title(self):
            return "Iris-Z"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            if not is_img2img:
                return []
            import gradio as gr

            with gr.Accordion("Iris-Z", open=False):
                mode = gr.Radio(list(irisz_script.MODES), value=irisz_script.MODE_IMAGE, label="Task",
                                elem_id="iris_z_task_mode")
                gr.Markdown("**Image to Depth**: the depth map of the input image, near = bright, at the input's "
                            "size. It ignores the prompt text, steps, CFG and denoising.")
            return [mode]

        def before_process(self, p, *args):
            irisz_script.before_process(p, *args[:1])

        def process_before_every_sampling(self, p, *args, **kwargs):
            irisz_script.before_sampling(p, kwargs.get("noise"), kwargs.get("x"))

        def postprocess_image(self, p, pp, *args):
            irisz_script.postprocess_image(p, pp)

        def postprocess(self, p, processed, *args):
            irisz_script.postprocess(p)
