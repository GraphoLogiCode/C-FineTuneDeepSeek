"""Shared detection input format for dataset construction and interactive inference."""

# Preserve the format of the collected dataset and the trained adapter exactly.
DETECT_SYSTEM = (
    "You are a C++ static-analysis assistant for game-engine code. "
    "Given one C++ function, list the clang-tidy checks it triggers.\n"
    'Answer with JSON only: {"checks": [<check names in alphabetical order>]}\n'
    'If no check applies, answer {"checks": []}\n'
)
DETECT_USER = "List the clang-tidy checks this C++ function triggers:\n\n{code}"


def detection_messages(code, context="", use_context=False):
    """Return inputs only. References and diagnostics must never enter this prompt."""
    if not code.strip():
        raise ValueError("Enter a C++ function first.")
    markers = ("### Instruction:", "### Response:", "<|EOT|>", "<｜begin▁of▁sentence｜>", "\x00")
    if any(marker in code or (use_context and marker in context) for marker in markers):
        raise ValueError("Remove chat-template markers from the C++ input.")
    if use_context:
        if not context.strip():
            raise ValueError("Add declarations, or turn off the context experiment.")
        user = (
            "Reference declarations (context only; do not report checks for these):\n"
            f"```cpp\n{context}\n```\n\n"
            "List the clang-tidy checks triggered by this target C++ function only:\n\n"
            f"{code}"
        )
    else:
        user = DETECT_USER.format(code=code)
    return [{"role": "system", "content": DETECT_SYSTEM}, {"role": "user", "content": user}]
