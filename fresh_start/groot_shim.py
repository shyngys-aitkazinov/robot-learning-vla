"""GROOT import-time shim — lets ``import lerobot.policies`` succeed.

lerobot 0.5.1 ships ``lerobot/policies/groot/groot_n1.py``, whose
``GR00TN15Config`` is a transformers ``PretrainedConfig`` dataclass with
non-default ``init=False`` fields declared after inherited default fields. As
soon as ``transformers`` is importable, defining that dataclass raises::

    TypeError: non-default argument 'backbone_cfg' follows default argument

``lerobot/policies/__init__.py`` and ``lerobot.policies.factory`` (used by the
training entry point) import the GROOT package eagerly, so on a machine with
``transformers`` installed — which SmolVLA needs — *any* ``import
lerobot.policies`` crashes.

:func:`apply` fixes this by pre-seeding ``sys.modules`` with an inert stub for
``lerobot.policies.groot.groot_n1`` *before* ``lerobot.policies`` is imported.
The broken module body then never runs. This pipeline is SmolVLA-only and
never touches GROOT, so the stub only has to satisfy GROOT's internal
``from ...groot_n1 import GR00TN15`` imports.

This works around lerobot purely at import time — no lerobot source is edited.
Call :func:`apply` before importing anything from ``lerobot.policies``.
"""

from __future__ import annotations

import sys
from types import ModuleType

_GROOT_N1_MODULE = "lerobot.policies.groot.groot_n1"
_applied = False


class _GrootDisabled:
    """Placeholder for every symbol GROOT modules import from ``groot_n1``.

    Subclassing it and reading attributes is harmless; *constructing* it is
    not, because that would mean GROOT is actually being used — unsupported by
    this SmolVLA-only pipeline.
    """

    def __init__(self, *args, **kwargs):  # noqa: D107
        raise RuntimeError(
            "GROOT is disabled by fresh_start.groot_shim — this pipeline only "
            "fine-tunes SmolVLA."
        )


def _stub_getattr(name: str):
    """Resolve any symbol (``GR00TN15``, ``GR00TN15Config``, ...) to the stub."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _GrootDisabled


def apply() -> None:
    """Install the ``groot_n1`` stub in ``sys.modules``. Idempotent."""
    global _applied
    if _applied:
        return
    if _GROOT_N1_MODULE not in sys.modules:
        stub = ModuleType(_GROOT_N1_MODULE)
        stub.__doc__ = "Inert stub injected by fresh_start.groot_shim."
        stub.__getattr__ = _stub_getattr  # PEP 562 module-level __getattr__
        sys.modules[_GROOT_N1_MODULE] = stub
    _applied = True


def selftest() -> None:
    """Apply the shim and prove ``lerobot.policies`` + SmolVLA import cleanly."""
    apply()
    import lerobot.policies  # noqa: F401
    from lerobot.policies.factory import make_policy  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401

    print("groot_shim OK: lerobot.policies + SmolVLAPolicy import cleanly")


if __name__ == "__main__":
    selftest()
