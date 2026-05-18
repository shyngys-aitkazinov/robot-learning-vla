"""OpenVLA instruction formatting (upstream template)."""


def wrap_openvla_prompt(instruction: str) -> str:
    """Match Hugging Face OpenVLA README template."""
    text = instruction.strip()
    return f"In: What action should the robot take to {text}?\nOut:"
