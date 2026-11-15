# Iris-Z support for Forge Neo
# Checks that the installed Forge Neo has every piece the backend builds on, before anything is patched.
# Same mechanism as the Iris 3B extension's compat.py.

import importlib
import inspect

# what the backend imports or patches, by module; an older Forge Neo misses some of these
REQUIRED = {
    "backend.attention": ("attention_function",),
    "backend.args": ("dynamic_args",),
    "backend.memory_management": ("load_model_gpu",),
    "backend.loader": ("load_huggingface_component", "possible_models"),
    "backend.nn.flux": ("IntegratedFluxTransformer2DModel",),
    "backend.diffusion_engine.base": ("ForgeDiffusionEngine",),
    "backend.diffusion_engine.flux2": ("Flux2",),
    "modules.sd_schedulers": ("schedulers_map",),
    "modules_forge.presets": ("PresetArch", "SAMPLERS", "SCHEDULERS", "STEPS", "CFG", "register"),
    "huggingface_guess.detection": ("detect_unet_config",),
    "huggingface_guess.model_list": ("models", "Flux2K9B"),
}

# the transformer goes through Forge's FLUX.2 branch of the loader, which builds the class named here
LOADER_SOURCE = ('"Flux2Transformer2DModel"', "IntegratedFluxTransformer2DModel(")
SCHEDULER = "Flux2"


def check() -> list[str]:
    problems: list[str] = []

    for module_name, names in REQUIRED.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            problems.append(f"{module_name} could not be imported ({type(e).__name__}: {e})")
            continue
        missing = [n for n in names if not hasattr(module, n)]
        if missing:
            problems.append(f"{module_name} is missing {', '.join(missing)}")

    if not problems:
        from backend import loader
        from modules import sd_schedulers

        # the whole module, not load_huggingface_component: an extension that loaded first may have put its own
        # wrapper there, and a wrapper's source holds none of Forge's branches
        source = inspect.getsource(loader)
        missing = [s for s in LOADER_SOURCE if s not in source]
        if missing:
            problems.append(f"the model loader has no {', '.join(missing)}")
        if SCHEDULER not in sd_schedulers.schedulers_map:
            problems.append(f"there is no {SCHEDULER} schedule type")

    return problems


def report(problems: list[str]) -> None:
    print("[Iris-Z] this Forge Neo is too old or too new for the extension, so it stays disabled "
          "(Forge Neo itself keeps working as usual):")
    for problem in problems:
        print(f"          {problem}")
