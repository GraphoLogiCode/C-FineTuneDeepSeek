"""The collected-data repair contract: requested checks plus one function -> rewritten function."""
FIX_SYSTEM = (
    "You are a C++ refactoring assistant for game-engine code. "
    "Rewrite the given function so that the listed clang-tidy findings are fixed and nothing else changes.\n"
    "Answer with the complete rewritten function only, with no explanation.\n"
)
FIX_USER = "Fix these clang-tidy findings: {checks}\n\n{code}"


def fixing_messages(code, checks):
    if not code.strip():
        raise ValueError("A target function is required.")
    if not checks or any(not isinstance(c, str) or not c or any(ch.isspace() for ch in c) for c in checks):
        raise ValueError("Provide at least one clang-tidy check name.")
    if any(marker in code for marker in ("### Instruction:", "### Response:", "<|EOT|>", "\x00")):
        raise ValueError("Chat-template markers are not allowed inside the code.")
    return [{"role": "system", "content": FIX_SYSTEM},
            {"role": "user", "content": FIX_USER.format(checks=", ".join(sorted(set(checks))), code=code)}]
