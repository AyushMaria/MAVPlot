"""
llm/safe_executor.py

Sandboxed execution of GPT-generated Python plotting scripts.

Strategy:
  - Compile the code first to catch SyntaxErrors cheaply.
  - Run in a child subprocess so any crash is isolated.
  - Inject a __import__ hook that uses a DENYLIST (not an allowlist).
    An allowlist breaks stdlib internals: e.g. `import json` triggers
    `import _io` internally, which would be silently blocked.
  - Kill the subprocess after timeout_seconds to prevent infinite loops.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

# Modules the generated script must NEVER be allowed to import.
# Internal C-extension names (leading underscore) are allowed so stdlib works.
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
    "telnetlib",
    "xmlrpc",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
}

# Preamble injected at the top of every generated script.
_PREAMBLE = textwrap.dedent("""\
    import builtins as _builtins
    _real_import = _builtins.__import__
    _BLOCKED = {blocked!r}

    def _safe_import(name, *args, **kwargs):
        top = name.split(".")[0]
        # Allow internal C-extensions (e.g. _io, _json) needed by stdlib
        if not top.startswith("_") and top in _BLOCKED:
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
