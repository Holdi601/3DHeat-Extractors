"""
Telling the player something without a window: a Windows notification where it
can be shown, the log either way.

The notification goes through PowerShell's own toast API, so it needs nothing
installed; where that is unavailable (another platform, a locked-down machine)
the message is only logged.
"""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger("heat3d_watch")

_TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$x = $t.GetElementsByTagName('text')
$x.Item(0).AppendChild($t.CreateTextNode($env:HEAT3D_TITLE)) > $null
$x.Item(1).AppendChild($t.CreateTextNode($env:HEAT3D_TEXT)) > $null
$n = [Windows.UI.Notifications.ToastNotification]::new($t)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('3DHeat').Show($n)
"""


def notify(title: str, text: str) -> None:
    log.info("%s: %s", title, text)
    if os.name != "nt" or os.environ.get("HEAT3D_NO_TOAST"):
        return
    try:
        # Title and text go in as environment variables, never into the script:
        # nothing a game or a file name says can become PowerShell code.
        subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST],
            env={**os.environ, "HEAT3D_TITLE": title, "HEAT3D_TEXT": text},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass
