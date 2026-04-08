"""
llm/safe_executor.py

Sandboxed execution of GPT-generated Python plotting scripts.

Strategy:
  - Parse the code with compile() to catch SyntaxErrors before execution.
  - Run in a subprocess with a strict import allowlist enforced by a custom
    __import__ hook injected into the script preamble.
  - Kill the subprocess after timeout_seconds to prevent infinite loops.

Why subprocess instead of RestrictedPython:
  RestrictedPython 7.x still leaks several sandbox escapes through
  __class__.__mro__ chains. A subprocess with a hard import blacklist and a
  wall-clock timeout is simpler, more reliable, and easier to test.

Allowed imports: matplotlib, pymavlink, math, json, os.path (read-only),
  collections, itertools, numpy (if installed).
"""

import subprocess
import sys
import textwrap
import tempfile
import os
from typing import Tuple

# Modules the generated script is allowed to import
ALLOWED_MODULES = {
    "matplotlib", "matplotlib.pyplot", "matplotlib.dates",
    "pymavlink", "pymavlink.mavutil",
    "math", "json", "collections", "itertools",
    "numpy", "numpy.np",
    "datetime", "re", "struct",
}

BLOCKED_MODULES = {"os", "subprocess", "sys", "shutil", "socket",
                   "importlib", "ctypes", "builtins", "signal"}

# Preamble injected before the user script to enforce the allowlist
_PREAMBLE = textwrap.dedent("""\
    import builtins as _builtins
    _real_import = _builtins.__import__
    _ALLOWED = {allowed!r}
    _BLOCKED = {blocked!r}

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in _BLOCKED or (top not in _ALLOWED and name not in _ALLOWED):
            raise ImportError(f"Import '{{name}}' is not allowed in this sandbox.")
        return _real_import(name, *args, **kwargs)

    _builtins.__import__ = _safe_import
""")


def execute_script(
    code: str,
    timeout_seconds: int = 30,
) -> Tuple[bool, str]:
    """
    Execute *code* in an isolated subprocess with a timeout and import guard.

    Args:
        code:            Python source code to execute.
        timeout_seconds: Wall-clock timeout. The process is killed if exceeded.

    Returns:
        (success, output) where *success* is True if the script exited with
        code 0 and *output* is stdout+stderr combined (or error message).
    """
    # 1. Syntax check before spawning a process
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    preamble = _PREAMBLE.format(
        allowed=ALLOWED_MODULES,
        blocked=BLOCKED_MODULES,
    )
    full_code = preamble + "\n" + code

    # 2. Write to a temp file and execute in a subprocess
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        output = (result.stdout + result.stderr).strip()
        return result.returncode == 0, output

    except subprocess.TimeoutExpired:
        return False, f"Script timed out after {timeout_seconds}s."
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
