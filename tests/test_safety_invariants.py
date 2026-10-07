"""Static checks for invariants that are easiest to enforce by inspecting source."""

from __future__ import annotations

import re

from .conftest import REPO_ROOT

SRC = REPO_ROOT / "src" / "ai_trader"

# Safety Invariant #5 and #6: no funding/withdrawal, no margin/leverage/derivatives.
# Matches ccxt unified methods and raw Kraken endpoint names when *called*.
_FORBIDDEN_CALL = re.compile(
    r"""\b(
        withdraw\w* | transfer\w* | \w*_?deposit_address\w* |
        set_leverage | set_margin_mode | add_margin | reduce_margin |
        private_post_withdraw\w* | private_post_wallettransfer\w*
    )\s*\(""",
    re.IGNORECASE | re.VERBOSE,
)
_FORBIDDEN_PARAM = re.compile(r"""["'](leverage|reduce_only|reduceOnly|margin)["']\s*:""")


def _source_files():
    return sorted(SRC.rglob("*.py"))


def test_no_withdrawal_transfer_or_margin_calls() -> None:
    offenders = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _FORBIDDEN_CALL.search(line) or _FORBIDDEN_PARAM.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Forbidden capability found:\n" + "\n".join(offenders)


def test_env_is_gitignored() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "/data/" in lines  # anchored: must not ignore src/ai_trader/data/


def test_env_example_has_no_values_for_secrets() -> None:
    secret_keys = {
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
    }
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in secret_keys:
            assert value.split("#")[0].strip() == "", f"{key} must be blank in .env.example"


def test_source_files_are_not_gitignored() -> None:
    import subprocess

    files = [str(p.relative_to(REPO_ROOT)) for p in _source_files()]
    result = subprocess.run(
        ["git", "check-ignore", *files], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.stdout.strip() == "", f"source files ignored by git:\n{result.stdout}"
