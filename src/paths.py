"""
paths.py
------------------------------
Resolves file paths correctly whether AirChord is running normally
from source (python src/app_ui.py) or as a PyInstaller-packaged
standalone .exe.

Why this is needed:
- When PyInstaller builds a --onefile exe, bundled files (like our
  model and default audio) get extracted to a temporary folder at
  runtime (sys._MEIPASS). Plain relative paths like "assets/audio"
  won't find them once packaged.
- Generated/regenerated audio (from the AI generator) needs to be
  saved somewhere the user actually has write permission to -- the
  temp extraction folder is wiped after the app closes, and the
  install location itself may be read-only (e.g. Program Files).
  So we use a per-user data folder for anything the app WRITES,
  separate from the bundled defaults it only READS.
"""

import sys
import os


def resource_path(relative_path):
    """
    Returns the correct path to a bundled, READ-ONLY resource
    (e.g. the hand-tracking model, default audio files).
    """
    if hasattr(sys, "_MEIPASS"):
        # Running as a PyInstaller-frozen exe -- files were extracted here
        base_path = sys._MEIPASS
    else:
        # Running normally from source -- use the project root
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    return os.path.join(base_path, relative_path)


def user_data_path(relative_path):
    """
    Returns a path inside a per-user, WRITABLE data folder for anything
    the app needs to save (like AI-regenerated chord audio), so it
    works correctly regardless of where the app is installed/run from.

    Windows: C:\\Users\\<you>\\AppData\\Roaming\\AirChord\\...
    macOS/Linux: ~/.airchord/...
    """
    if sys.platform == "win32":
        base = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "AirChord")
    else:
        base = os.path.join(os.path.expanduser("~"), ".airchord")

    full_path = os.path.join(base, relative_path)
    os.makedirs(os.path.dirname(full_path) if os.path.splitext(full_path)[1] else full_path, exist_ok=True)
    return full_path