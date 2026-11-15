# Iris-Z support for Forge Neo
# huggingface_guess registration. Iris-Z-small is FLUX.2 [klein] Base 9B underneath: the same text encoder
# (Qwen3 8B), VAE, latent format and sampling, so the guess is Forge's own Klein 9B with its own model name.
#
# huggingface_repo stays Forge's FLUX.2-klein-9B folder: the loader reads the tokenizer, VAE and text encoder
# configurations from it, and the "klein" in its name turns on Forge's Klein behaviour (the Flux2 schedule under
# "Automatic", the one Iris-Z samples with). The extension's loader hook routes only the transformer.

from huggingface_guess import model_list


class IrisZ(model_list.Flux2K9B):
    unet_config = {"image_model": "iris_z"}

    # Forge reserves activation memory from the latent's area; every Iris-Z sequence holds two grids of it (the
    # image and its depth), twice the tokens of a Klein 9B image of the same size
    memory_usage_factor = 2 * model_list.Flux2K9B.memory_usage_factor
