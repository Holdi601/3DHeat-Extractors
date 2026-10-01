# Licences and third-party components

Everything in this repository that was written for it - the three extractors,
the telemetry recorders, the BeamNG.drive protocol mod and the Trackmania
plugin - is under the MIT licence in [`LICENSE`](LICENSE). Use it, change it,
ship it, commercially or not.

What the MIT licence cannot cover is what the tools are used with: the Python
packages they import, the models game capture runs, and the games they read.
Those carry their own terms, and they are listed here because the answer differs
by part, and the difference decides whether output may be used at work.

Every licence below was read off the package metadata or the model card in
September 2026, not recalled. Where a component's terms changed after that, the
source is what counts, not this file.

---

## The three parts

| Part | What it is | Who uses it | Commercial use of the output |
|---|---|---|---|
| **Unreal exporter** (`unreal/`) | Level → `.glb` from inside the editor | Developers, on their own game | Yes |
| **Game capture** (`gamecapture/`) | Level reconstructed from gameplay video; lap telemetry recorded from games' published interfaces | Players, and anyone without the project files | **Depends on the model chosen** — see below |
| **Game files** (`gamefiles/`) | Courses and levels read out of shipped archives | Players and developers with an installed game | Yes for the tool; the game's own content stays its owner's |
| **CS2** (`cs2/`) | Counter-Strike 2 demos into tables and match summaries; maps into geometry | Players, analysts, teams | Yes for the tool; demos and maps stay their owners' |

A studio developer has the project open and should use the Unreal exporter: it
reads the real geometry and is exact. Someone without the project — a player,
an analyst — has only what the screen shows, and that is what game capture is
for. Someone with the installed game but not the project sits between the two,
and that is what the game-file exporter is for.

---

## Unreal exporter — `unreal/`

Content-only Unreal plugin: Python, no compiled module. It uses the editor's own
scripting API and carries no third-party dependency of its own.

Obligations come from Unreal Engine's own EULA, which applies to anyone running
the editor and is not altered by this plugin.

---

## Game capture and telemetry recorders — `gamecapture/`

### Software

| Package | Licence | Note |
|---|---|---|
| numpy, scipy, scikit-image | BSD-3-Clause | |
| opencv-python | Apache-2.0 | |
| onnxruntime, onnxruntime-directml | MIT | |
| windows-capture | MIT | |
| **PySide6-Essentials** | **LGPL-3.0** | See below |
| pytest | MIT | Development only |
| torch, transformers, onnxscript | BSD-3-Clause / Apache-2.0 | One-off model conversion only; not a runtime dependency |

**PySide6 is the one with conditions.** LGPL-3.0 permits use in proprietary
software, including commercially, provided Qt stays a *replaceable* library: it is
used here as an ordinary dynamically-linked import, which satisfies that, and the
licence notice must travel with any distribution. It is not statically linked and
not modified. If this is ever bundled into a single executable, that packaging has
to keep Qt replaceable or the obligation is not met.

### The depth model — this is the one that matters

Depth Anything V2 is **not licensed alike across its sizes**, and the tool's
output inherits whichever was used:

| Model | Licence | Commercial use |
|---|---|---|
| **Depth-Anything-V2-Small** | **Apache-2.0** | **Permitted** |
| Depth-Anything-V2-Base | CC-BY-NC-4.0 | **Non-commercial only** |
| Depth-Anything-V2-Large | CC-BY-NC-4.0 | **Non-commercial only** |

**Small is the default**, and `tools/export_onnx.py` defaults to it, precisely so
that a scan carries no restriction unless someone deliberately chooses otherwise.
Base was the default until the model cards were actually read, which would have
made every scan a licence problem for anyone using this at work without ever
saying so.

Choosing `--size base` or `--size large` is permitted and produces better depth.
It also means the resulting model, and every reconstruction made with it, is
non-commercial. That is a decision for whoever runs it, not a default to inherit.

### VGGT, which is now used

**VGGT** (`facebook/VGGT-1B`) is a feed-forward model that produces camera poses,
depth and point maps for a *set* of frames in one pass, so every frame constrains
every other one directly rather than through a chain of pairwise estimates. That
is why it is here: incremental visual odometry loses a racetrack, where the road
is a smooth ribbon with little texture and nothing pulls a drifting pose back.

It is **CC-BY-NC-4.0 — non-commercial only, and a reconstruction made with it
inherits that.**

A separately licensed `facebook/VGGT-1B-Commercial` exists. It is a **gated
repository**: access is requested from Meta and granted per account, and without
that it cannot be downloaded at all — the API answers 401. So it cannot be a
default here. `heat3d_capture/pose/feedforward.py` names both, and
`describe_licence()` exists so the interface can state the restriction at the
point of use rather than leave it to be discovered afterwards.

The practical reading: reconstruction from video is the part of this project a
*player* would run, and non-commercial terms fit that. Anyone reconstructing for
commercial work should use an engine export or the game-file exporter, both of
which are exact anyway.

---

## Game files — `gamefiles/`

### Software

| Package | Licence | Note |
|---|---|---|
| numpy | BSD-3-Clause | Mesh decoding |
| cryptography | Apache-2.0 / BSD-3-Clause | Only for archives the user holds a key for |
| lz4 | BSD-3-Clause | Optional; some Unity bundles need it |
| Pillow | MIT-CMU (HPND) | Decoding block-compressed textures, writing JPEGs |
| msgpack | Apache-2.0 | BeamNG's compiled shapes (`.cdae`) |
| zstandard | BSD-3-Clause | The few `.cdae` shapes that are zstd-compressed |
| pytest | MIT | Development only |

All permissive. This exporter carries no restriction of its own.

### Oodle is found, not bundled

Most shipped Unreal content is Oodle-compressed. **Oodle (RAD Game Tools, now
Epic) is proprietary and has no redistributable build**, so `oodle.py` loads a
library already on the machine rather than carrying one — the arrangement the
licence contemplates, and the same one FModel and UModel use. `oo2core_*.dll` is
licensed to the product that ships it; `HEAT3D_OODLE` points at a copy the user
already has.

### The engine formats themselves

`.pak`, Unity's `SerializedFile` and Frostbite's `.toc` are undocumented file
formats. Reading them is reverse engineering of a container, which is what the
section below is about; none of the readers here is derived from another
project's source.

Two formats are not undocumented. BeamNG publishes its `.cdae` shape cache's
layout, and the reader follows that. Unreal 5's cooked property data needs each
class's property list in order to be read; the lists embedded here
(`unversioned.py`) are property names and types only - facts about the format,
as an interoperability mappings file holds them - derived from the Unreal
Engine headers Epic makes available to anyone with a linked account, and
checked against the shipped data. No engine source is
included or required.

---

## CS2 — `cs2/`

### Software

| Package | Licence | Note |
|---|---|---|
| demoparser2 | MIT | Reading the demos |
| pyarrow | Apache-2.0 | Writing Parquet |
| pandas, numpy | BSD-3-Clause | |
| pytest | MIT | Development only |

All permissive. The map export runs **Source2Viewer's command line**
([ValveResourceFormat](https://github.com/ValveResourceFormat/ValveResourceFormat),
MIT) as a separate program the user installs; it is not bundled and nothing
of it is copied here.

### What it reads

A demo is the server's recording of a match; the map is part of the game. The
tool reads files the user has - their own matches, demos they downloaded, the
installed game - and what comes out is for their own analysis. Demos and
data derived from them are not this project's to redistribute, and not the
user's either: Valve has had a dataset built from professional demos taken
down. Sites that offer demos commonly forbid automated downloading in their
terms of use, which is why the tool reads demos from disk and fetches none.

---

## How geometry is obtained, and what that obliges

Not a licence question, but it belongs next to one: both are obligations that are
easy to breach without noticing. Three routes exist to a level's geometry, and
they are constrained by quite different things.

**Nothing here injects into a game process.** Frames come from Windows Graphics
Capture, the public API screen recorders use, and telemetry from feeds games
publish themselves. That is a deliberate constraint: injection into a title with
anti-cheat risks the account of whoever runs it.

**ReShade** was considered. Its depth buffer would make reconstruction far more
accurate, and ReShade itself is broadly tolerated — BattlEye states it bans only
actual cheats. But ReShade **disables depth-buffer access in multiplayer sessions
by design**, because depth access is a wallhack, and that is why it is tolerated
at all. So it is usable for single-player and offline titles, where it would be a
genuine improvement, and unavailable for exactly the always-online case an
analytics capture usually needs.

**Extracting geometry from shipped game files** — `.pak` archives and the like —
is a third route, and it is now built, as `gamefiles/`. An earlier
draft of this document ruled it out wholesale. That was too broad, and the
correction is worth recording because the reasoning matters more than the
conclusion.

A tool is not the same thing as a distribution. Extractors are published openly
and used widely — FModel, UModel, AssetRipper, Noesis — and the obvious
legitimate use is a developer reading *their own* game's build. Nothing about
parsing a container format is itself a circumvention.

The line that does matter is **encryption**, and it falls inside the tool rather
than around it. DMCA §1201(a)(2) reaches technologies, not only acts: a tool
whose purpose is to defeat an access control is in scope even if nobody
distributes a single extracted file. That is why the established extractors are
built the way they are — **they ship no keys and no key-recovery**. FModel cannot
open an encrypted archive until the user supplies an AES key they obtained
themselves. The tool reads containers; it does not break locks.

So the design, which is what was built:

- **Read unencrypted archives directly.** Ordinary parsing, no circumvention
  question arises. Every archive tested so far has been in this category.
- **For encrypted archives, take a key from the user and never recover one.** No
  bundled keys, no key extraction from the executable, no memory scraping.
  `pak.py` takes `aes_key=` and has no code path that obtains one.
- **Extract geometry, not assets to redistribute.** The output wanted here is a
  decimated collision or terrain mesh for spatial context, which is also the
  least of what an archive holds.

One consequence worth stating plainly: Frostbite's obfuscated table form is
*detected and declined*. Its key sits in the file itself, so it is obfuscation
rather than an access control and no key would be needed from anyone — but no
file on hand uses it, and implementing a format with nothing to check it against
produces something that looks like support and is not.

What remains the user's responsibility either way is the EULA of the game they
point it at. Those commonly prohibit reverse engineering and are enforceable
contracts; that obligation sits with whoever runs the tool, on whichever game,
and no design choice here discharges it.

---

## Sources

- [Depth Anything V2 model cards](https://huggingface.co/depth-anything) — licence per size
- [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B) — CC-BY-NC-4.0
- [ReShade — PCGamingWiki](https://www.pcgamingwiki.com/wiki/ReShade) — anti-cheat position
- [ReShade forum, depth buffer detection](https://reshade.me/forum/general-discussion/4083-depth-buffer-detection-modifications?start=1200) — multiplayer depth disabling
- [Reverse engineering and the law](https://ipwatchdog.com/2021/03/27/reverse-engineering-law-understand-restrictions-minimize-risks/) — EULA and DMCA position
- [DMCA §1201 anti-circumvention](https://copyrightalliance.org/education/copyright-law-explained/the-digital-millennium-copyright-act-dmca/section-1201-technology-protection/) — the tools provisions reach technologies, not only acts
- [FModel setup](https://modding.wiki/en/aewff/developers/fmodel) — established extractors require a user-supplied AES key rather than shipping one
- [facebook/VGGT-1B-Commercial](https://huggingface.co/facebook/VGGT-1B-Commercial) — gated; access is granted per account, so it cannot be a default
- [Oodle Data Compression](http://www.radgametools.com/oodle.htm) — proprietary, no redistributable build, hence found rather than bundled
- [Review of Feed-forward 3D Reconstruction: DUSt3R to VGGT](https://arxiv.org/abs/2507.08448)

This document is a record of what the licences say. It is not legal advice, and
the commercial question for any particular use is one for whoever is answerable
for that use.
