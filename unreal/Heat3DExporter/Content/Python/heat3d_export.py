"""
Headless entry point.

    UnrealEditor-Cmd.exe <project>.uproject ^
      -run=pythonscript -script="<...>/heat3d_export.py --map /Game/Maps/L_Example --out C:/exports/L_Example.glb" ^
      -unattended -nopause -nosplash -nullrhi

The script path and its arguments go inside one quoted `-script=` value; the
Python plugin splits them into `sys.argv`. `run-export.ps1` in the plugin folder
does all of this, including finding the editor and working around plugins that
have no compiled binaries.

Writes the .glb, and a .json beside it recording what was kept, what was dropped
and why, and where the time went.
"""
import json
import os
import sys
import traceback

import unreal

# The plugin's Python folder is on the path when the plugin is mounted, but not
# when this file is run by absolute path from outside it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from heat3d import settings as settings_module  # noqa: E402
from heat3d import exporter  # noqa: E402


def main():
    argv = sys.argv[1:]
    settings = settings_module.from_argv(argv)
    report, _written = exporter.run(settings)

    path = settings.report_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, default=str)
    unreal.log("[heat3d] report written to {}".format(path))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - a commandlet must report, not vanish
        unreal.log_error("[heat3d] export failed:\n{}".format(traceback.format_exc()))
        raise
