# Iris-Z support for Forge Neo
# The few Forge Neo runtime pieces the DiT calls, with plain torch fallbacks outside Forge (CPU tests).

import torch

try:
    from backend import attention as _forge_attention
except ImportError:  # outside Forge Neo: eager backend on CPU
    _forge_attention = None

__all__ = ["attention", "fp16_fix"]


def attention(q, k, v, transformer_options=None):
    """q, k, v: [B, H, S, D] -> [B, S, H * D], no mask (FLUX.2 attends to every text token, padding included).

    transformer_options reach Forge's attention wrapper, which applies attention overrides of other extensions. The
    torch fallback (CPU tests) pins the MATH backend, the plain reference implementation.
    """
    if _forge_attention is not None:
        # looked up per call: Forge picks its attention function at import time and other extensions may wrap it
        return _forge_attention.attention_function(q, k, v, q.shape[1], skip_reshape=True,
                                                   transformer_options=transformer_options or {})
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.MATH):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def fp16_fix(x):
    """Forge's guard for float16 compute (GPUs without bf16): FLUX activations can leave the float16 range."""
    if x.dtype == torch.float16:
        return x.nan_to_num(nan=0.0, posinf=65504.0, neginf=-65504.0).clamp(-65504.0, 65504.0)
    return x
