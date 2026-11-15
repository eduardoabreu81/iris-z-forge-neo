# Iris-Z support for Forge Neo
# Recognize an Iris-Z DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized. Every width comes from a weight's rows (its output features) or a norm: 4-bit builds pack two values
# per byte along the input features, so a Linear's column count is not its input width there.

import re

# the FLUX.2 layout (Forge's own flux2 detection keys on double_stream_modulation_img) plus Iris-Z's modality embedding
IDENTITY_KEYS = ("modality_in.weight", "double_stream_modulation_img.lin.weight", "double_blocks.0.img_attn.qkv.weight",
                 "final_layer.linear.weight")
# FLUX.2 [dev]'s width: Iris-Z-big, whose Mistral Small 3.1 text encoder Forge Neo has no engine for
BIG_HIDDEN = 6144

BIG_MESSAGE = ("[Iris-Z] this is Iris-Z-big (FLUX.2 [dev], 32B): it needs FLUX.2 [dev]'s Mistral text encoder, which "
               "Forge Neo does not load. Use Iris-Z-small.")


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_iris_z(names, key_prefix=""):
    return all(f"{key_prefix}{k}" in names for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The IrisZDiT keyword arguments for this checkpoint, or None when it is not an Iris-Z DiT."""
    if not is_iris_z(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    hidden = shape("final_layer.adaLN_modulation.1.weight")[0] // 2
    if hidden == BIG_HIDDEN or f"{key_prefix}guidance_in.in_layer.weight" in state_dict:
        raise ValueError(BIG_MESSAGE)
    head_dim = shape("double_blocks.0.img_attn.norm.query_norm.scale")[0]
    config = {"image_model": "iris_z"}
    config["in_channels"] = shape("final_layer.linear.weight")[0]
    config["hidden_size"] = hidden
    config["num_heads"] = hidden // head_dim
    config["axes_dim"] = [head_dim // 4] * 4
    config["theta"] = 2000
    # the SiLU-gated MLP: its first layer outputs gate and value, 2 * mlp_ratio * hidden rows
    config["mlp_ratio"] = shape("double_blocks.0.img_mlp.0.weight")[0] / 2 / hidden
    config["depth"] = count_blocks(keys, f"{key_prefix}double_blocks.")
    config["depth_single_blocks"] = count_blocks(keys, f"{key_prefix}single_blocks.")
    # the text projection stays in full precision in every published quantized FLUX.2 build
    config["context_in_dim"] = shape("txt_in.weight")[1]
    config["num_modalities"] = shape("modality_in.weight")[0]
    return config
