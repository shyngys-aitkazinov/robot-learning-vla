"""PyTorch 2.x compatibility shim for FlowerVLA's attention.

Problem
-------
``external/flower_vla_calvin/flower/models/networks/transformers.py`` (the
``FlowerSelfAttention.forward`` path, ~line 199) calls
``torch.nn.functional.scaled_dot_product_attention`` with BOTH an explicit
``attn_mask`` AND ``is_causal=True``. PyTorch >= 2.x rejects that combination:

    RuntimeError: _scaled_dot_product_attention: Explicit attn_mask should
    not be set when is_causal=True

The FlowerVLA code builds an explicit upper-triangular causal mask, then
*also* passes ``is_causal=True`` — the flag is redundant (the explicit mask
already encodes causality) but newer torch treats the pair as an error.

Why monkey-patch (not edit ``external/``)
-----------------------------------------
``external/flower_vla_calvin`` is a pristine ``git clone`` (see the setup
recipe in ``scripts/run_eval3_flower_deploy.sh``). Editing it directly would
be lost on a fresh clone. Same pattern as ``eval3_lerobot_shim.py`` /
``eval3_concat_patch.py``: work around the dependency at import time.

Fix
---
Wrap ``F.scaled_dot_product_attention`` so that when a caller passes both an
explicit ``attn_mask`` and ``is_causal=True``, we MERGE the causal constraint
into the mask and clear the flag. Merging (rather than just dropping
``is_causal``) keeps the shim correct for any caller whose explicit mask is
not itself causal.

Apply ``apply()`` once, before the FlowerVLA model runs a forward pass.
Idempotent.
"""
from __future__ import annotations

import logging

_APPLIED = False


def apply() -> None:
    """Install the SDPA compat wrapper on torch.nn.functional."""
    global _APPLIED
    if _APPLIED:
        return

    import torch
    import torch.nn.functional as F

    _orig_sdpa = F.scaled_dot_product_attention

    def _sdpa_compat(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        # Only intervene on the illegal combination; every other call is
        # passed straight through untouched.
        if is_causal and attn_mask is not None:
            L = query.size(-2)
            S = key.size(-2)
            # Lower-triangular (incl. diagonal) = position i may attend to j<=i.
            causal = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
            if attn_mask.dtype == torch.bool:
                # Boolean mask: True = attend. Intersect with the causal mask.
                attn_mask = attn_mask & causal
            else:
                # Additive float mask: 0 = attend, -inf = block. Add -inf
                # wherever the causal mask forbids attention.
                neg = torch.zeros(L, S, dtype=attn_mask.dtype, device=query.device)
                neg = neg.masked_fill(~causal, float("-inf"))
                attn_mask = attn_mask + neg
            is_causal = False
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            **kwargs,
        )

    F.scaled_dot_product_attention = _sdpa_compat
    # Also patch the torch.nn.functional module object directly in case some
    # code captured the attribute via `torch.nn.functional` rather than `F`.
    torch.nn.functional.scaled_dot_product_attention = _sdpa_compat

    logging.info(
        "[eval3_flower_sdpa_compat] patched F.scaled_dot_product_attention "
        "(merges is_causal into explicit attn_mask for torch>=2.x)"
    )
    _APPLIED = True


if __name__ == "__main__":
    apply()
    # Quick self-check: the illegal combination must no longer raise.
    import torch
    import torch.nn.functional as F

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    bool_mask = torch.ones(1, 2, 4, 4, dtype=torch.bool).tril()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask, is_causal=True)
    assert out.shape == (1, 2, 4, 8), out.shape
    print("[eval3_flower_sdpa_compat] self-check OK — illegal combo handled")
