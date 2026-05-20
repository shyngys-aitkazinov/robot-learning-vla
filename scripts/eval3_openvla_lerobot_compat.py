"""Let lerobot's robot/camera/control utilities import under OpenVLA's pinned
transformers==4.40.1.

Problem
-------
OpenVLA's checkpoints (and its ``trust_remote_code`` modeling) target
``transformers==4.40.1`` (see ``integrations/openvla/requirements-min.txt``).
But ``lerobot``'s ``policies/__init__.py`` eagerly imports *every* bundled
policy at package-init time, and the newer ones need model modules that only
exist in much newer transformers:

    lerobot/policies/__init__.py
      -> pi05      -> transformers.models.paligemma   (transformers >= 4.42)
      -> smolvla   -> SmolVLM                          (transformers >= 4.46)
      -> ...

So merely doing ``from lerobot.utils.control_utils import ...`` (which the
OpenVLA deploy needs for the keyboard listener) triggers the whole policy
import cascade and crashes with::

    ModuleNotFoundError: No module named 'transformers.models.paligemma'

Bumping transformers is not an option: lerobot's newest policies need
transformers ~5.x, which is far past what OpenVLA's remote code tolerates.

Fix
---
The OpenVLA deploy adapter (``scripts/eval3_openvla_deploy.py``) never uses a
lerobot *policy* — OpenVLA is not a LeRobot ``PreTrainedPolicy``. It only
needs lerobot's **robot / camera / processor / control** utilities. So before
any lerobot import, we pre-register stub ``lerobot.policies`` /
``lerobot.policies.pretrained`` / ``lerobot.policies.utils`` modules in
``sys.modules``. With those already present, Python never runs
``lerobot/policies/__init__.py`` and the eager pi05/smolvla cascade is
skipped entirely.

``lerobot.utils.control_utils`` imports exactly two names from those modules
(``PreTrainedPolicy`` for a type hint, ``prepare_observation_for_inference``)
— the stubs satisfy both. The OpenVLA deploy calls neither.

Why a separate shim (not ``eval3_lerobot_shim``)
------------------------------------------------
``eval3_lerobot_shim`` is shared by the SmolVLA train/deploy scripts, which
DO need real lerobot policies. This stub is OpenVLA-deploy-specific — only
``eval3_openvla_deploy.py`` applies it.

Apply ``apply()`` once, before importing anything from ``lerobot``.
Idempotent.
"""
from __future__ import annotations

import sys
import types

_APPLIED = False


def apply() -> None:
    """Pre-register stub lerobot.policies modules so lerobot core imports cleanly."""
    global _APPLIED
    if _APPLIED:
        return

    if "lerobot.policies" in sys.modules:
        # lerobot.policies already imported for real (or already stubbed) —
        # nothing safe to do; bail out rather than clobber a real module.
        _APPLIED = True
        return

    # lerobot.policies — empty namespace package (no __init__ body runs).
    policies = types.ModuleType("lerobot.policies")
    policies.__path__ = []  # marks it as a package so submodule lookups work
    sys.modules["lerobot.policies"] = policies

    # lerobot.policies.pretrained — control_utils imports PreTrainedPolicy
    # purely as a type annotation; a bare class satisfies it.
    pretrained = types.ModuleType("lerobot.policies.pretrained")

    class PreTrainedPolicy:  # noqa: D401 - placeholder for type hints only
        """Stub — the OpenVLA deploy never instantiates a lerobot policy."""

    pretrained.PreTrainedPolicy = PreTrainedPolicy
    sys.modules["lerobot.policies.pretrained"] = pretrained
    policies.pretrained = pretrained

    # lerobot.policies.utils — control_utils imports prepare_observation_for_inference
    # at module load. The OpenVLA deploy never calls it (it runs its own
    # OpenVLA inference path), so the stub raises loudly if anything does.
    putils = types.ModuleType("lerobot.policies.utils")

    def prepare_observation_for_inference(*args, **kwargs):  # noqa: ANN001, ANN201
        raise RuntimeError(
            "lerobot.policies.utils.prepare_observation_for_inference is stubbed "
            "for the OpenVLA deploy (eval3_openvla_lerobot_compat). The OpenVLA "
            "adapter must not route through lerobot's policy-inference helpers."
        )

    putils.prepare_observation_for_inference = prepare_observation_for_inference
    sys.modules["lerobot.policies.utils"] = putils
    policies.utils = putils

    _APPLIED = True


if __name__ == "__main__":
    apply()
    # Self-check: every lerobot import the OpenVLA deploy makes must succeed.
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
    from lerobot.configs import parser  # noqa: F401
    from lerobot.processor import make_default_processors  # noqa: F401
    from lerobot.robots import RobotConfig, make_robot_from_config, so_follower  # noqa: F401
    from lerobot.utils.control_utils import init_keyboard_listener  # noqa: F401
    from lerobot.utils.import_utils import register_third_party_plugins  # noqa: F401
    from lerobot.utils.robot_utils import precise_sleep  # noqa: F401
    from lerobot.utils.utils import init_logging, log_say  # noqa: F401

    print("[eval3_openvla_lerobot_compat] self-check OK — lerobot core imports cleanly")
