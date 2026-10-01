// 3DHeat recorder: every frame of a run, written as a lap CSV the 3DHeat Race
// tab reads, one file per run, into this plugin's storage folder
// (OpenplanetNext/PluginStorage/Heat3dRecorder).
//
// It reads only what Openplanet exposes for plugins - the viewed car's
// vehicle state (VehicleState plugin) and the player's race time - and writes
// files; it sends nothing anywhere and changes nothing in the game.
//
// Columns are the viewer's channel names: time, position, speed, the car's
// own forward and sideways speed (from which the viewer takes the drift
// angle), pedals and steering, gear, rpm, per wheel Trackmania's slip
// coefficient, and whether the car is on loose ground or in the air. The
// first line is the run's description as JSON after a '#'.
//
// Unsigned plugins load in Openplanet's developer mode only (which limits
// online play) - see docs/games.md.

const string VERSION = "1";

array<string> rows;
bool recording = false;
int lastTime = -1;
uint respawnsAtStart = 0;
string mapName = "";

// Loose surfaces, by EPlugSurfaceMaterialId: the ground the car slides on.
bool Loose(EPlugSurfaceMaterialId m) {
    return m == EPlugSurfaceMaterialId::Grass
        || m == EPlugSurfaceMaterialId::Ice
        || m == EPlugSurfaceMaterialId::Sand
        || m == EPlugSurfaceMaterialId::Dirt
        || m == EPlugSurfaceMaterialId::Green
        || m == EPlugSurfaceMaterialId::RoadIce;
}

string Header() {
    return "time,x,y,z,speed,longVel,latVel,yaw,throttle,brake,steer,gear,rpm,"
        + "slipIndex.FL,slipIndex.FR,slipIndex.RL,slipIndex.RR,loose,airborne,respawns";
}

string F(float v) { return Text::Format("%.4f", v); }

void Begin() {
    rows.RemoveRange(0, rows.Length);
    recording = true;
}

void Finish(bool complete, int raceTime) {
    if (!recording) return;
    recording = false;
    if (rows.Length < 20) return;
    string meta = "{\"format\":\"heat3d-lap-csv\",\"version\":" + VERSION
        + ",\"game\":\"trackmania\",\"track\":\"" + mapName.Replace("\"", "'") + "\""
        + ",\"complete\":" + (complete ? "true" : "false")
        + ",\"seconds\":" + F(raceTime / 1000.0)
        + ",\"meta\":{\"discipline\":\"arcade\"}}";
    string name = "run_" + Time::FormatString("%Y-%m-%d_%H-%M-%S") + (complete ? "" : "_partial") + ".csv";
    IO::File f(IO::FromStorageFolder(name), IO::FileMode::Write);
    f.Write("# " + meta + "\n" + Header() + "\n");
    for (uint i = 0; i < rows.Length; i++) f.Write(rows[i] + "\n");
    f.Close();
    print("3DHeat: wrote " + name + " (" + rows.Length + " samples)");
}

void Update(float dt) {
    auto app = cast<CTrackMania>(GetApp());
    if (app is null) return;
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) {
        Finish(false, lastTime);
        lastTime = -1;
        return;
    }
    auto terminal = playground.GameTerminals[0];
    auto player = cast<CSmPlayer>(terminal.GUIPlayer);
    if (player is null) return;
    auto script = cast<CSmScriptPlayer>(player.ScriptAPI);
    if (script is null) return;
    auto vis = VehicleState::ViewingPlayerState();
    if (vis is null) return;
    if (app.RootMap !is null) mapName = Text::StripFormatCodes(app.RootMap.MapName);

    int t = script.CurrentRaceTime;
    // Finished: the playground's UI shows the finish sequence (11).
    bool finished = playground.UIConfigs.Length > 0 && int(playground.UIConfigs[0].UISequence) == 11;
    if (finished) {
        Finish(true, lastTime);
        return;
    }
    // Back to the start: the clock runs from the countdown again.
    if (recording && t < lastTime - 500) Finish(false, lastTime);
    if (t < 0) {
        lastTime = t;
        return;
    }
    if (!recording) {
        Begin();
        respawnsAtStart = script.CurrentRaceRespawns;
    }
    lastTime = t;

    float speed = vis.WorldVel.Length();
    float side = VehicleState::GetSideSpeed(vis);
    bool loose = Loose(vis.FLGroundContactMaterial) || Loose(vis.FRGroundContactMaterial)
        || Loose(vis.RLGroundContactMaterial) || Loose(vis.RRGroundContactMaterial);
    rows.InsertLast(
        F(t / 1000.0) + "," + F(vis.Position.x) + "," + F(vis.Position.y) + "," + F(vis.Position.z)
        + "," + F(speed) + "," + F(vis.FrontSpeed) + "," + F(side)
        + "," + F(Math::Atan2(vis.Dir.x, vis.Dir.z))
        + "," + F(vis.InputGasPedal) + "," + (vis.InputIsBraking ? "1" : "0") + "," + F(vis.InputSteer)
        + "," + vis.CurGear + "," + F(VehicleState::GetRPM(vis))
        + "," + F(vis.FLSlipCoef) + "," + F(vis.FRSlipCoef) + "," + F(vis.RLSlipCoef) + "," + F(vis.RRSlipCoef)
        + "," + (loose ? "1" : "0") + "," + (vis.IsGroundContact ? "0" : "1")
        + "," + (script.CurrentRaceRespawns - respawnsAtStart));
}
