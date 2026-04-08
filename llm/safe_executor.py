"""
safe_executor.py

Provides a sandboxed Python script executor using RestrictedPython.
Only a safe subset of builtins and a whitelist of allowed imports are permitted.
This prevents GPT-generated code from accessing the filesystem, network,
or executing shell commands.
"""

import io
import sys
import traceback
from RestrictedPython import compile_restricted, safe_globals, safe_builtins
from RestrictedPython.Guards import (
    safe_globals as rp_safe_globals,
    guarded_iter_unpack_sequence,
)
from RestrictedPython.PrintCollector import PrintCollector

# Modules that the generated plotting scripts are allowed to import
ALLOWED_MODULES = {
    "pymavlink",
    "pymavlink.mavutil",
    "matplotlib",
    "matplotlib.pyplot",
    "math",
    "os.path",
    "json",
    "re",
    "datetime",
    "collections",
    "numpy",
}


def _safe_import(name, *args, **kwargs):
    """
    A guarded __import__ that only allows modules in ALLOWED_MODULES.
    Raises ImportError for anything outside the whitelist.
    """
    top_level = name.split(".")[0]
    if name not in ALLOWED_MODULES and top_level not in ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed in generated scripts. "
            f"Allowed modules: {', '.join(sorted(ALLOWED_MODULES))}"
        )
    return __import__(name, *args, **kwargs)


def execute_script(script_code: str, timeout_seconds: int = 30) -> tuple[bool, str]:
    """
    Execute a Python script in a RestrictedPython sandbox.

    Args:
        script_code: The Python source code string to execute.
        timeout_seconds: Maximum allowed execution time (enforced via threading).

    Returns:
        A tuple of (success: bool, output_or_error: str).
        On success, output_or_error contains captured stdout.
        On failure, output_or_error contains the error traceback.
    """
    import threading

    result = {"success": False, "output": ""}

    def _run():
        try:
            byte_code = compile_restricted(script_code, filename="<generated>", mode="exec")
        except SyntaxError as e:
            result["output"] = f"SyntaxError in generated script: {e}"
            return

        # Build a restricted globals dict
        restricted_globals = dict(safe_globals)
        restricted_globals["__builtins__"] = dict(safe_builtins)
        restricted_globals["__builtins__"]["__import__"] = _safe_import
        restricted_globals["_getiter_"] = iter
        restricted_globals["_getattr_"] = getattr
        restricted_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        restricted_globals["_print_"] = PrintCollector

        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            exec(byte_code, restricted_globals)  # noqa: S102
            result["success"] = True
            result["output"] = captured_output.getvalue()
        except Exception:  # noqa: BLE001
            result["output"] = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        return False, f"Script execution timed out after {timeout_seconds} seconds."

    return result["success"], result["output"]
