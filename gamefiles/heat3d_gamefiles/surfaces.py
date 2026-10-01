"""
The kinds of ground an export tells apart, and how each is shown.

Where each kind comes from matters more than the list, because the obvious
source is the wrong one:

- **road**, **verge** and **terrain** do *not* come from material names. Around a
  circuit 94% of the ground carries a road-shader material - the terrain near
  roads is drawn with it, and asphalt or grass is decided per pixel. Road comes
  from the vertex layout (road strips carry three UV sets, nothing else does);
  within a strip, asphalt and verge are told apart by the tile's own asphalt
  mask. See `forzaterrain.surface_keys` and `courseexport.plan_parts`.
- **markings**, **tyre marks** and **water** *do* come from material names,
  because for those the name is specific: a decal submesh is nothing but its
  decal.

The colours are for flat-shaded parts and for viewers that ignore textures.
They are chosen to read at a glance, not to match the game.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Keeps a part's name in the form the viewer classifies by. Anything not
#: matching `^(ground|structure|water)\\s*[:|]` is classified by a heuristic
#: written for Unreal landscapes, which is not what a Forza circuit looks like.
KIND = "ground"


@dataclass(frozen=True)
class Surface:
    """One kind of ground, and how to show it."""

    key: str
    label: str
    #: Linear RGB, 0..1.
    colour: tuple[float, float, float]
    #: Material names that mean this surface, where a name can mean anything.
    pattern: re.Pattern[str] | None = None


ROAD = Surface("road", "road", (0.22, 0.22, 0.23))
VERGE = Surface("verge", "verge", (0.52, 0.47, 0.30))
TERRAIN = Surface("terrain", "terrain", (0.30, 0.44, 0.18))
# Before `markings`, because a tyre mark ships as `tyremark_decal_dirt` and would
# otherwise be caught by the `decal` in its own name and painted white.
TYREMARKS = Surface("tyremarks", "tyre marks", (0.20, 0.17, 0.16), re.compile(r"tyremark|skid", re.I))
MARKINGS = Surface("markings", "markings", (0.90, 0.90, 0.88), re.compile(r"whiteline|yellowline|decal", re.I))
WATER = Surface("water", "water", (0.16, 0.34, 0.48), re.compile(r"water|lake|river|paddy|ocean|sea", re.I))
OTHER = Surface("other", "other", (0.55, 0.30, 0.55))
# Drivable ground that is not paved - a gravel stage, a dirt road, a sand
# track. Forza's roads are always strips of their own; BeamNG paints them onto
# the terrain, and the ground model under them says which they are.
LOOSE = Surface("loose", "gravel and dirt", (0.50, 0.42, 0.30))

#: The surfaces a material name can decide, in the order they are tried.
BY_MATERIAL = (TYREMARKS, MARKINGS, WATER)

SURFACES = (ROAD, VERGE, TERRAIN, TYREMARKS, MARKINGS, WATER)
BY_KEY = {s.key: s for s in (*SURFACES, LOOSE, OTHER)}


def classify(material: str) -> Surface:
    """
    What a material name alone says about its triangles.

    `other` for everything that is not a marking, a tyre mark or water -
    including every road and terrain material, since for those the name does
    not decide it. `other` is a signal to look elsewhere, not a surface.
    """
    name = material or ""
    for surface in BY_MATERIAL:
        if surface.pattern.search(name):
            return surface
    return OTHER


def part_name(surface: Surface) -> str:
    """The viewer's own class for a surface, which is `water` only for water."""
    kind = "water" if surface.key == "water" else KIND
    return f"{kind}:{surface.label}"
