"""
mavpose/safe_executor.py

Sandboxed execution of GPT-generated Python plotting scripts.

Strategy:
  - Compile the code first to catch SyntaxErrors cheaply.
  - Run in a child subprocess so any crash is isolated.
  - Inject a __import__ hook that uses a DENYLIST checked against the
    caller's __file__. This means stdlib modules loading their own
    transitive dependencies (e.g. json -> re -> enum -> sys) are NOT
    blocked — only direct imports from the user script itself are.
  - Kill the subprocess after timeout_seconds to prevent infinite loops.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

# Modules the generated script must NEVER import directly.
BLOCKED_MODULES = {
    "os",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "importlib",
    "ctypes",
    "signal",
    "pty",
    "tty",
    "termios",
    "fcntl",
    "pwd",
    "grp",
    "resource",
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    "urllib",
    "http",
    "ftplib",
    "smtplib",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
}

# Preamble injected at the top of every generated script.
#
# The hook inspects the *caller's* __file__: if the import was triggered
# directly from the user script (caller_file == _SCRIPT) we enforce the
# denylist.  Imports triggered by stdlib loading its own sub-modules
# (e.g. json -> re -> enum -> sys) come from a different __file__ and are
# allowed through, so `import json` / `import math` etc. work correctly.
_PREAMBLE = textwrap.dedent("""\
    import builtins as _builtins
    import sys as _sys
    _real_import = _builtins.__import__
    _BLOCKED = {blocked!r}
    _SCRIPT = __file__

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if top in _BLOCKED:
            frame = _sys._getframe(1)
            caller_file = frame.f_globals.get("__file__", "") or ""
            if caller_file == _SCRIPT:
                raise ImportError(
                    f"Import '{{name}}' is not allowed in this sandbox."
                )
        return _real_import(name, *args, **kwargs)

    _builtins.__import__ = _safe_import
""")


def execute_script(
    code: str,
    timeout_seconds: int = 30,
) -> tuple:
    """
    Execute *code* in an isolated subprocess with a timeout and import guard.

    Args:
        code:            Python source code to execute.
        timeout_seconds: Wall-clock timeout. The process is killed if exceeded.

    Returns:
        (success, output) — success is True if exit code is 0.
    """
    # 1. Syntax check before spawning a process
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    preamble = _PREAMBLE.format(blocked=BLOCKED_MODULES)
    full_code = preamble + "\n" + code

    # 2. Write to a temp file and run in a subprocess
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
        return False, f"script timed out after {timeout_seconds}s."
    except Exception as exc:
        return False, str(exc)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
