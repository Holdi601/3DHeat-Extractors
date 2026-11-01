# Updates

Every tool in this repository checks GitHub for a newer release when it starts
(at most every six hours, never blocking for more than a few seconds), says so,
and offers to update: a git checkout with `git pull --ff-only` when it has no
local changes, a downloaded copy by replacing its files from the release
archive. The watcher updates itself while no game is running and starts again.
`HEAT3D_NO_UPDATE=1` turns the check off.

The 3DHeat viewer does the same from its own side: `start.bat` brings a clean
checkout up to date before it starts, and the viewer shows a notice with an
"Update now" button when a newer version is out.
