"""
Editor entry point: adds Tools -> Export level for 3DHeat.

Runs automatically when the plugin is mounted. The menu exports the level that
is currently open, with the defaults, to `Saved/Heat3D/<Level>.glb`; anything
that needs different settings is better done from the command line, where they
can be written down and repeated — see `run-export.ps1`.

Exporting a big World Partition level takes minutes and blocks the editor while
it runs. That is deliberate: a background export would race the editor's own
loading of the actors it is reading.
"""
import os
import traceback

import unreal

MENU = "LevelEditor.MainMenu.Tools"
SECTION = "Heat3D"


def _settings_for_current_level():
    from heat3d import settings as settings_module

    world = unreal.EditorLevelLibrary.get_editor_world()
    if not world:
        raise RuntimeError("no level is open")
    package = world.get_outermost().get_name()  # /Game/Maps/L_Example
    name = package.rsplit("/", 1)[-1]
    out = os.path.join(
        unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir()),
        "Heat3D",
        name + ".glb",
    )
    return settings_module.Settings(map=package, out=out, name=name)


def _summary(settings, report):
    """What the dialog says when it worked.

    More than "done", because the two things that decide whether the result is
    usable are both in the report and neither is the triangle count. The size cut
    says whether buildings kept their parts; the skipped-instance count says how
    much of the map is simply absent. Someone running this from the menu has no
    command line to tune, so the dialog has to tell them a command line is what
    they need next.
    """
    steps = report.get("steps", {})
    parts = report.get("output", {}).get("parts", [])
    lines = [
        "Wrote {}".format(settings.out),
        "",
        "{:,} triangles, {:.1f} MB".format(
            sum(p["triangles"] for p in parts),
            report.get("output", {}).get("bytes", 0) / 1e6,
        ),
        "{:,} objects, everything above {}m across".format(
            steps.get("structures", {}).get("instances", 0),
            steps.get("collect", {}).get("size_cut_m", 0),
        ),
    ]
    if report.get("budget_chosen"):
        lines.append(
            "Budget chosen for this level's size: {:,} triangles.".format(settings.budget)
        )
    if report.get("warnings"):
        lines += [
            "",
            "This level is large for a one-click export. Buildings made of "
            "separate actors may be missing their smaller parts. Export from the "
            "command line with a higher --budget, or a --region crop, to fix it — "
            "see the plugin's README.",
        ]
    return "\n".join(lines)


@unreal.uclass()
class Heat3DExportEntry(unreal.ToolMenuEntryScript):
    @unreal.ufunction(override=True)
    def execute(self, context):
        import json

        from heat3d import exporter

        try:
            settings = _settings_for_current_level()
            unreal.log("[heat3d] exporting {} -> {}".format(settings.map, settings.out))
            report, _written = exporter.run(settings)
            with open(settings.report_path(), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=1, default=str)
            unreal.log("[heat3d] done: {}".format(settings.out))
            unreal.EditorDialog.show_message(
                "3DHeat export", _summary(settings, report), unreal.AppMsgType.OK
            )
        except Exception as err:  # noqa: BLE001 - a menu item must not vanish
            unreal.log_error("[heat3d] export failed:\n{}".format(traceback.format_exc()))
            unreal.EditorDialog.show_message(
                "3DHeat export failed", str(err), unreal.AppMsgType.OK
            )


def register():
    menus = unreal.ToolMenus.get()
    menu = menus.find_menu(MENU)
    if not menu:
        return False
    menu.add_section(SECTION, unreal.Text("3DHeat"))

    entry = Heat3DExportEntry()
    entry.init_entry(
        owner_name=menu.menu_name,
        menu=MENU,
        section=SECTION,
        name="Heat3DExportLevel",
        label="Export level for 3DHeat",
        tool_tip="Write a decimated world-space .glb of this level for the telemetry viewer.",
    )
    menu.add_menu_entry_object(entry)
    menus.refresh_all_widgets()
    return True


# Commandlets mount the plugin too. They have no menu to add to, which
# `find_menu` reports by returning nothing, so no separate check is needed — and
# a failure here must never stop the editor from starting.
try:
    register()
except Exception:  # noqa: BLE001 - never break editor startup over a menu
    unreal.log_warning("[heat3d] could not register the menu:\n{}".format(traceback.format_exc()))
