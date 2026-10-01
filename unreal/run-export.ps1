<#
.SYNOPSIS
  Export a level from an Unreal project as a .glb for the 3DHeat viewer.

.DESCRIPTION
  Runs the editor headless with the exporter's Python entry point. Everything
  awkward about doing that from a command line lives here:

    * finding the editor for a source build or an installed one;
    * `-nullrhi`, because the alternative (`-AllowCommandletRendering`) compiles
      global shaders on the way in, which fails in plenty of workspaces and is
      not needed to read geometry;
    * disabling plugins whose modules have no compiled binary. A single one of
      those aborts editor startup in `-unattended` mode with a message dialog
      nobody sees, and a workspace with prebuilt binaries usually has a few.

.EXAMPLE
  .\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example -Out C:\exports\L_Example.glb

.EXAMPLE
  .\run-export.ps1 -Project C:\game\Game.uproject -Map /Game/Maps/L_Example -Out C:\exports\L_Example.glb `
                   -Args '--min-size 6 --budget 3000000 --region -120000 -120000 120000 120000'
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)][string]$Project,
  [Parameter(Mandatory = $true)][string]$Map,
  [Parameter(Mandatory = $true)][string]$Out,
  # Editor executable, or the engine root. Guessed from the project when omitted.
  [string]$Engine,
  # Extra arguments for the exporter itself; see `--help` in heat3d/settings.py.
  # Not named `Args`: that shadows PowerShell's own `$args`.
  [Alias('Args')][string]$Options = '',
  # Where the plugin lives. Defaults to the copy next to this script.
  [string]$Plugin = (Join-Path $PSScriptRoot 'Heat3DExporter'),
  # Run this Python file instead of the exporter's entry point.
  #
  # For the diagnostic probes in .probe/. They need everything this script does —
  # the editor lookup, `-nullrhi`, and above all the broken-plugin list, since a
  # single module without a binary aborts startup with a dialog nobody sees. Run
  # by hand instead, a probe fails after two minutes with a plugin error that has
  # nothing to do with what it was asking.
  [string]$Script,
  [switch]$KeepLog
)

$ErrorActionPreference = 'Stop'

function Find-Editor {
  param([string]$project, [string]$hint)

  if ($hint) {
    if ($hint -like '*.exe') { return $hint }
    $candidate = Join-Path $hint 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
    if (Test-Path $candidate) { return $candidate }
    throw "No UnrealEditor-Cmd.exe under -Engine '$hint'."
  }

  # A source build keeps the engine beside the project, usually one level up.
  $dir = Split-Path -Parent (Resolve-Path $project)
  for ($i = 0; $i -lt 4 -and $dir; $i++) {
    $candidate = Join-Path $dir 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
    if (Test-Path $candidate) { return $candidate }
    $dir = Split-Path -Parent $dir
  }

  # Otherwise an installed engine, matched to the project's association.
  $assoc = (Get-Content $project -Raw | ConvertFrom-Json).EngineAssociation
  if ($assoc) {
    $key = "HKLM:\SOFTWARE\EpicGames\Unreal Engine\$assoc"
    if (Test-Path $key) {
      $root = (Get-ItemProperty $key).InstalledDirectory
      $candidate = Join-Path $root 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
      if (Test-Path $candidate) { return $candidate }
    }
  }
  throw 'Could not find UnrealEditor-Cmd.exe. Pass -Engine with the engine root.'
}

function Get-BrokenPlugins {
  <#
    Plugin modules with no compiled DLL anywhere in the tree.

    Workspaces that download prebuilt binaries routinely miss a few, and the
    editor refuses to start rather than skipping them — in -unattended mode it
    prints a dialog to a console nobody is reading and exits 1.
  #>
  param([string]$projectDir)

  $built = [System.Collections.Generic.HashSet[string]]::new()
  Get-ChildItem $projectDir -Recurse -Directory -Filter 'Win64' -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'Binaries' } |
    ForEach-Object {
      Get-ChildItem $_.FullName -Filter 'UnrealEditor-*.dll' -ErrorAction SilentlyContinue |
        ForEach-Object { [void]$built.Add(($_.BaseName -replace '^UnrealEditor-', '')) }
    }

  $broken = @()
  Get-ChildItem $projectDir -Recurse -Filter '*.uplugin' -ErrorAction SilentlyContinue |
    ForEach-Object {
      $descriptor = $null
      try { $descriptor = Get-Content $_.FullName -Raw | ConvertFrom-Json } catch { return }
      foreach ($module in @($descriptor.Modules)) {
        if ($module -and $module.Name -and -not $built.Contains($module.Name)) {
          $broken += $_.BaseName
        }
      }
    }
  return ($broken | Sort-Object -Unique)
}

$editor = Find-Editor -project $Project -hint $Engine
$projectDir = Split-Path -Parent (Resolve-Path $Project)

# Install the plugin into the project if it is not already there. Copying is the
# only way in: the engine has no "load a plugin from this path" switch.
$installed = Join-Path $projectDir 'Plugins\Heat3DExporter'
if (-not (Test-Path (Join-Path $installed 'Heat3DExporter.uplugin'))) {
  Write-Host "Installing plugin into $installed"
  New-Item -ItemType Directory -Force (Split-Path -Parent $installed) | Out-Null
  Copy-Item -Recurse -Force $Plugin $installed
} else {
  # Keep the installed copy current, so editing the source here is enough.
  Copy-Item -Recurse -Force (Join-Path $Plugin '*') $installed
}

$entry = if ($Script) { (Resolve-Path $Script).Path } else {
  Join-Path $installed 'Content\Python\heat3d_export.py'
}
$broken = Get-BrokenPlugins -projectDir $projectDir
if ($broken) { Write-Host "Disabling plugins with no compiled module: $($broken -join ', ')" }

$outDir = [System.IO.Path]::GetDirectoryName($Out)
New-Item -ItemType Directory -Force $outDir | Out-Null
# One log per run. A fixed name is a nuisance the moment two exports overlap, or
# one is killed and its redirect keeps the handle: the next run then dies on
# "the process cannot access the file" after doing nothing wrong.
$log = Join-Path $outDir ('heat3d-export-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')

# Settings go in a file, not on the command line. Unreal splits the value of
# `-script=` on spaces without honouring quotes inside it, so a path with a
# space in it arrives as two arguments; an out folder called "My Exports" would
# fail at the argument parser after a two-minute editor start.
$config = @{ map = $Map; out = $Out; argv = $Options }
$configPath = Join-Path $outDir 'heat3d-config.json'
$config | ConvertTo-Json -Depth 4 | Set-Content -Path $configPath -Encoding UTF8
$env:HEAT3D_CONFIG = $configPath

$arguments = @(
  $Project,
  '-run=pythonscript',
  "-script=$entry",
  '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities,GeometryScripting',
  '-unattended', '-nopause', '-nosplash', '-nullrhi', '-stdout', '-FullStdOutLogOutput'
)
if ($broken) { $arguments += "-DisablePlugins=$($broken -join ',')" }

if ($Script) { Write-Host "Running $entry against $Map" } else { Write-Host "Exporting $Map -> $Out" }
Write-Host "  editor: $editor"
$started = Get-Date
& $editor @arguments *> $log
$elapsed = [int]((Get-Date) - $started).TotalSeconds

# The editor's exit code is unreliable in commandlet mode — shutdown ensures and
# late errors set it non-zero after a successful run — so the file is the truth.
Select-String -Path $log -Pattern '\[heat3d\]' | ForEach-Object { $_.Line -replace '^\[[^\]]+\]\[[^\]]+\]', '' }

if ($Script) {
  # A probe writes whatever it writes; there is no $Out to check, so the log is
  # the result and it is always kept.
  Write-Host "Finished in ${elapsed}s. Log: $log"
  exit 0
}

if (Test-Path $Out) {
  $mb = [math]::Round((Get-Item $Out).Length / 1MB, 1)
  Write-Host "Done in ${elapsed}s: $Out ($mb MB)"
  if (-not $KeepLog) { Remove-Item $log -ErrorAction SilentlyContinue }
  exit 0
}

Write-Host "Export failed after ${elapsed}s. Log: $log"
Select-String -Path $log -Pattern 'LogPython: Error|Traceback|Fatal error|Assertion failed' |
  Select-Object -First 20 | ForEach-Object { $_.Line }
exit 1
