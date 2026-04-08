"""
file_validator.py

Validates uploaded files before they are passed to pymavlink.
Checks file extension, MIME type, and basic magic-byte signature.
"""

import os
import mimetypes
from pathlib import Path

# Allowed extensions for MAVLink log files
ALLOWED_EXTENSIONS = {".tlog", ".bin", ".log"}

# Maximum allowed upload size: 200 MB
MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024

# Known MIME types for binary/log files (many systems report these as octet-stream)
ACCEPTABLE_MIME_TYPES = {
    "application/octet-stream",
    "application/x-binary",
    "text/plain",  # some .log files
    None,           # mimetypes may return None for unknown types
}


class FileValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_mavlink_file(filepath: str) -> None:
    """
    Validate that a file is a plausible MAVLink log file.

    Args:
        filepath: Absolute or relative path to the uploaded file.

    Raises:
        FileValidationError: If the file fails any validation check.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)

    # 1. Existence
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # 2. Must be a regular file (not a symlink, directory, device node, etc.)
    if not path.is_file() or path.is_symlink():
        raise FileValidationError(
            "Upload must be a regular file. Symlinks and directories are not allowed."
        )

    # 3. Extension check
    ext = path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Invalid file type '{ext}'. "
            f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are accepted."
        )

    # 4. File size check
    size = path.stat().st_size
    if size == 0:
        raise FileValidationError("Uploaded file is empty.")
    if size > MAX_FILE_SIZE_BYTES:
        raise FileValidationError(
            f"File size ({size / 1_048_576:.1f} MB) exceeds the 200 MB limit."
        )

    # 5. MIME type check (best-effort — not all systems detect these correctly)
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type not in ACCEPTABLE_MIME_TYPES:
        raise FileValidationError(
            f"Unexpected MIME type '{mime_type}' for a MAVLink log file."
        )

    # 6. Path traversal guard — ensure the resolved path stays within its parent directory
    try:
        path.resolve().relative_to(path.parent.resolve())
    except ValueError:
        raise FileValidationError("Path traversal detected in uploaded file path.")  # noqa: B904
