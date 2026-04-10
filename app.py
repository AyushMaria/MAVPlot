"""
app.py

MAVPose no longer ships a Gradio UI.
The headless CLI entry point is cli.py.

This file is kept so existing references don't 404, and to provide
a helpful error message if someone tries to run it directly.
"""

import sys


def main() -> None:
    print(
        "The Gradio UI has been removed from MAVPose.\n"
        "Use the CLI instead:\n\n"
        "  python cli.py <log_file> [--prompt \"Plot altitude over time\"]\n\n"
        "Run `python cli.py --help` for full usage."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
