#!/usr/bin/env bash
# Start PX4 SITL (Gazebo Harmonic, x500) with the parameters this project needs, applied on EVERY start.
#
#   tools/run_sitl.sh              # GUI
#   HEADLESS=1 tools/run_sitl.sh   # no Gazebo window
#   PX4_DIR=/path/to/PX4-Autopilot tools/run_sitl.sh
#
# Why not just `param set` + `param save`? PX4's SITL startup (ROMFS/.../init.d-posix/rcS) resets ALL saved parameters
# whenever the airframe being booted differs from the one in the saved parameter file. If another project boots a
# different airframe from the same PX4 tree (e.g. ws_skie2's 4050_gz_skie_2), our saved values are wiped the next
# time we boot the x500 (4001). PX4_PARAM_<NAME>=<value> environment variables are applied AFTER that reset.
set -euo pipefail

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

# SITL has no ground-control station: without this, arming is denied ("No connection to the GCS")
export PX4_PARAM_NAV_DLL_ACT=0

cd "$PX4_DIR"
exec make px4_sitl gz_x500 "$@"
