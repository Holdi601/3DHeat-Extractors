-- heat3d: the player's car, sent over UDP for the 3DHeat Race tab's recorder.
--
-- A BeamNG "protocol" like the game's own OutGauge and MotionSim ones: the
-- game's protocols.lua packs the struct below and sends it to the address and
-- port given here, from the car the player is driving, when Options > Other >
-- Protocols has "others" switched on. Nothing outside the vehicle's own Lua is
-- touched. See https://documentation.beamng.com/modding/protocols/
--
-- Install: copy this folder (heat3d/) to
--   %LOCALAPPDATA%\BeamNG\BeamNG.drive\current\mods\unpacked\
-- or run `python -m heat3d_capture telemetry --game beamng --install`.
-- Then record with `python -m heat3d_capture telemetry --game beamng`.

local M = {}

-- Where the recorder listens. Change both here and on its command line.
local ADDRESS = "127.0.0.1"
local PORT = 4460
local RATE = 100

local clock = 0
local order = {}

-- The four corner wheels, by where they sit on the car rather than by name
-- (names differ between vehicles): front/rear along the car, left/right across.
-- Vehicle frame: +X left, -Y forwards (documentation.beamng.com, refNodes).
local function findWheels()
  order = {}
  local list = {}
  for _, wd in pairs(wheels.wheels) do
    local p = obj:getNodePosition(wd.node1)
    table.insert(list, { wd = wd, left = p.x, fwd = -p.y })
  end
  if #list == 0 then return end
  local cx, cy = 0, 0
  for _, w in ipairs(list) do cx = cx + w.left; cy = cy + w.fwd end
  cx, cy = cx / #list, cy / #list
  local best = {}
  for _, w in ipairs(list) do
    local key = (w.fwd >= cy and "F" or "R") .. (w.left >= cx and "L" or "R")
    local reach = math.abs(w.left - cx) + math.abs(w.fwd - cy)
    if not best[key] or reach > best[key].reach then best[key] = { wd = w.wd, reach = reach } end
  end
  for i, key in ipairs({ "FL", "FR", "RL", "RR" }) do
    order[i] = best[key] and best[key].wd or nil
  end
end

function M.init() clock = 0; findWheels() end
function M.reset() clock = 0; findWheels() end
function M.getAddress() return ADDRESS end
function M.getPort() return PORT end
function M.getMaxUpdateRate() return RATE end
function M.isPhysicsStepUsed() return true end

function M.getStructDefinition()
  return [[
    char  magic[4];
    float time;
    float px, py, pz;
    float vx, vy, vz;
    float fx, fy, fz;
    float ux, uy, uz;
    float yawRate;
    float throttle, brake, clutch, handbrake, steer, rpm, gear, abs, tcs;
    float wheelSpeed[4];
    float wheelAngular[4];
    float wheelLoad[4];
    float wheelMaterial[4];
  ]]
end

function M.fillStruct(o, dtSim)
  local e = electrics.values
  clock = clock + (dtSim or 0)
  o.magic = "H3D1"
  o.time = clock
  o.px, o.py, o.pz = obj:getPositionXYZ()
  o.vx, o.vy, o.vz = obj:getVelocityXYZ()
  local f = obj:getDirectionVector()
  local u = obj:getDirectionVectorUp()
  o.fx, o.fy, o.fz = f.x, f.y, f.z
  o.ux, o.uy, o.uz = u.x, u.y, u.z
  local _, _, yawRate = obj:getRollPitchYawAngularVelocity()
  o.yawRate = yawRate or 0
  o.throttle = e.throttle or 0
  o.brake = e.brake or 0
  o.clutch = e.clutch or 0
  o.handbrake = e.parkingbrake or 0
  o.steer = e.steering_input or 0
  o.rpm = e.rpm or 0
  o.gear = e.gearIndex or 0
  -- The aids' lamps, where the car has them.
  o.abs = (e.absActive and e.absActive ~= 0) and 1 or 0
  o.tcs = (e.tcsActive and e.tcsActive ~= 0) and 1 or 0
  for i = 1, 4 do
    local wd = order[i]
    if wd then
      o.wheelSpeed[i - 1] = wd.wheelSpeed or 0
      o.wheelAngular[i - 1] = (wd.angularVelocity or 0) * (wd.wheelDir or 1)
      o.wheelLoad[i - 1] = wd.downForce or 0
      o.wheelMaterial[i - 1] = wd.contactMaterialID1 or -1
    else
      o.wheelSpeed[i - 1] = 0
      o.wheelAngular[i - 1] = 0
      o.wheelLoad[i - 1] = 0
      o.wheelMaterial[i - 1] = -1
    end
  end
end

return M
