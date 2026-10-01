"""
Which texture colours a placed model.

A model's materials do not name their textures; they list hashes. The names are
in the track's `AssetManifest.xml`, a plain XML file beside the archives that
lists every texture the track ships with its `SourceHash`:

    <Texture Source="tracks\\<map>\\textures\\roads\\assets\\swatches\\
             road_gen_rum_round_asan_bclr_b2d95eb8-….swatch"
             SourceHash="3130211576" PVSTextureIndex="32031" />

and the swatch's file in the archive is that name with `_quality<n>.pb` in
place of `.swatch`. `forzatech.material_textures` reads the hashes out of a
model; this turns them into the one texture worth drawing it with.

That is the base colour. The standard environment shader reads it from slot
0x88B483AA, and 1,054 of 1,054 sampled materials that have that slot hold a
`_bclr` texture in it. Other shaders - signage, rock, building LODs - put their
colour elsewhere under other slot ids, so for them the texture's own name
decides: its kind is the word before the GUID (`_bclr_`, `_diff_`,
`_colour_`), and the manifest is where that name comes from, not a guess
from the model's name. A model named `bar_rur_armco_crct_02_c` is coloured by
`bar_rur_armco_crct_02_a_xsummerx_bclr`, because that is what its material
says; looking for a texture by the model's own name finds nothing for it, and
the wrong thing for others.
"""

from __future__ import annotations

import re
from pathlib import Path

from .forzatech import SLOT_BASE_COLOUR

MANIFEST = "AssetManifest.xml"
_TEXTURE = re.compile(r'<Texture Source="([^"]+)" SourceHash="(\d+)"')

#: `…_<kind>_<guid>`, or `…_<kind>_<seven characters>` for the short ids some
#: swatches carry instead.
_GUID = re.compile(r"_([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[a-z0-9]{7})$")

#: Texture kinds that hold a colour, in order of preference.
COLOUR_KINDS = ("bclr", "diff", "colour")

#: Colour-kind textures that colour nothing by themselves: terrain tile maps (the
#: ground has its own bake), and tint palettes that a shader multiplies other
#: textures by.
_NOT_A_COLOUR = re.compile(r"^(mainmap|submap|autogloss)|tint|worldtint", re.I)

SEASONS = ("summer", "spring", "autumn", "winter")


class Manifest:
    """Every texture a track ships, by the hash its materials use."""

    def __init__(self, names: dict[int, str]):
        self.names = names

    @classmethod
    def read(cls, folder: str | Path) -> "Manifest | None":
        """
        The manifest beside a track's archives, or None if there is none.

        Only the texture list is read. The file is 90 MB, nearly all of it
        per-instance model records that come after the textures, so reading stops
        at `</Textures>`.
        """
        path = Path(folder) / MANIFEST
        if not path.is_file():
            return None
        names: dict[int, str] = {}
        with path.open(encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                if "</Textures>" in line:
                    break
                found = _TEXTURE.search(line)
                if found:
                    leaf = found.group(1).rsplit("\\", 1)[-1].lower().removesuffix(".swatch")
                    names[int(found.group(2))] = leaf
        return cls(names)

    def hashes(self):
        """Every hash, sorted, which is the form `material_textures` searches."""
        import numpy as np

        return np.unique(np.fromiter(self.names, dtype=np.uint32, count=len(self.names)))

    def name(self, value: int) -> str | None:
        return self.names.get(value)


def kind(name: str) -> str:
    """The word before the GUID: `bclr`, `nrml`, `extra`, `mask`…"""
    bare = _GUID.sub("", name)
    return bare.rsplit("_", 1)[-1]


def strip_guid(name: str) -> str:
    return _GUID.sub("", name)


def season_of(name: str) -> str | None:
    for season in SEASONS:
        if f"x{season}x" in name or f"_{season}_" in name:
            return season
    return None


def colour_texture(slots: list[tuple[int, int]], manifest: Manifest) -> str | None:
    """The base-colour texture one material binds, by name, or None."""
    for slot, value in slots:
        if slot == SLOT_BASE_COLOUR:
            name = manifest.name(value)
            if name and not _NOT_A_COLOUR.search(name):
                return name
    for wanted in COLOUR_KINDS:
        for _slot, value in slots:
            name = manifest.name(value)
            if name and kind(name) == wanted and not _NOT_A_COLOUR.search(name):
                return name
    return None


def in_season(name: str, season: str, available) -> str:
    """
    The same texture for another season, if the track has one.

    Materials come in seasonal pairs - `…_xsummerx_bclr` and `…_xwinterx_bclr`,
    or `…_bclr` and `…_winter_bclr` - with different GUIDs, so the swap is by
    name with the GUID left off, and `available` (a callable taking such a name
    and returning a full one or None) says whether it exists.
    """
    current = season_of(name)
    if current == season or (current is None and season == "summer"):
        return name
    bare = strip_guid(name)
    word = kind(name)
    if current is None:
        # Summer is often the unmarked one: `_bclr` pairs with `_winter_bclr`.
        head = bare[: -len(word) - 1]
        candidates = [f"{head}_{season}_{word}", f"{head}_x{season}x_{word}"]
    else:
        swapped = bare.replace(f"x{current}x", f"x{season}x").replace(f"_{current}_", f"_{season}_")
        candidates = [swapped]
        if season == "summer":
            candidates.append(bare.replace(f"_{current}_", "_"))
    for candidate in candidates:
        found = available(candidate)
        if found:
            return found
    return name
