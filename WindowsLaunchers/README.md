# Windows launcher

Double-click `Verknuepfungen-installieren.cmd` to create **BiBaZu Drop
Simulation** shortcuts on the Desktop and in Start Menu > BiBaZu. To pin it,
find the Start Menu entry, right-click it and choose **Pin to Start**.

The shortcut starts this repository's `.venv\Scripts\pythonw.exe`, so install
the GUI dependencies with `uv sync --python 3.12 --all-extras` first. The
installer checks the environment and leaves other BiBaZu shortcuts alone.
The simulator uses no hardware lease or hardware devices and can run alongside
the other BiBaZu GUIs.

For a direct launch, double-click `..\DropSimulationGUI.cmd` or run
`..\Start-DropSimulationGUI.ps1`.

```powershell
.\Install-BiBaZuShortcuts.ps1 -CheckOnly
.\Install-BiBaZuShortcuts.ps1 -StartMenuOnly
.\Install-BiBaZuShortcuts.ps1 -DesktopOnly
```

The PNG source and multi-resolution ICO live in `icons`. To rebuild the ICO,
run `python icons\build_icons.py` in an environment with Pillow. Reinstall the
shortcut after moving this repository.
