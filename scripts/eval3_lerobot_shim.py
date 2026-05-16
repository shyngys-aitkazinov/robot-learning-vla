"""Work around LeRobot GROOT + transformers interaction (GR00TN15Config vs HF PretrainedConfig).

Import this and call ``apply()`` before any ``lerobot.policies`` import when SmolVLA needs
``transformers`` installed.

Background: with `transformers` installed, importing `lerobot.policies` cascades into
`lerobot.policies.groot.groot_n1`, whose dataclass inherits from a transformers parent that
breaks Python dataclass field ordering (`non-default argument 'backbone_cfg' follows default
argument`). The naive workaround is to set `lerobot.utils.import_utils._transformers_available
= False` before any policy import, but that also breaks SmolVLA's `TokenizerProcessorStep`
which legitimately needs transformers.

Surgical fix: pre-load the tokenizer processor with the flag True (so transformers is properly
imported and cached in sys.modules), then briefly flip the flag False to load GROOT with stubs
(avoiding the dataclass crash), then restore the flag for the rest of the process. Subsequent
imports of GROOT just reuse the cached stubs from sys.modules; SmolVLA gets real transformers."""

from __future__ import annotations


def apply() -> None:
    import lerobot.utils.import_utils as import_utils

    saved = import_utils._transformers_available

    # 1) Load TokenizerProcessorStep WITH transformers so it captures the real transformers symbols.
    import lerobot.processor.tokenizer_processor  # noqa: F401

    # 2) Disable transformers detection just long enough to import GROOT modules,
    #    so their dataclass inheritance uses stub parents instead of the broken HF PretrainedConfig.
    import_utils._transformers_available = False
    try:
        import lerobot.policies.groot.groot_n1  # noqa: F401
        import lerobot.policies.groot.action_head.flow_matching_action_head  # noqa: F401
        import lerobot.policies.groot.modeling_groot  # noqa: F401
        import lerobot.policies.groot.processor_groot  # noqa: F401
    finally:
        # 3) Restore the flag so later imports (e.g. SmolVLA's processor pipeline) work normally.
        import_utils._transformers_available = saved
