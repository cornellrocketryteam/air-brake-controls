# Controller v2 Development Log

## Overview
Rewrote the Rust airbrake controller to work with real flight CSV data containing State/Phase columns, added sensor calibration routines, and fixed physics modeling to match reality.

## Changes Made

### CSV Phase Parsing (main.rs)
- Controller reads `State` column from CSV: `Pad`, `Boost`, `Coast`
- Added `Phase::Pad` variant to the controller's phase enum
- `seen_coast` sticky flag: once a coast row is seen, all subsequent rows are treated as coast
- Phase detection lives in `main.rs`, not the controller — the controller only receives phase via `SensorData`

### Gyro Drift Calibration (gyro_calibration.rs)
- New module that computes gyro drift rates from 40 pad-phase readings
- Calculates average rate of change per axis (x, y) over consecutive stationary samples
- Drift is subtracted from all subsequent gyro readings before passing to the controller

### Initial Tilt Measurement (measure_tilt.rs)
- New module that computes initial rocket tilt from accelerometer data during pad phase
- Averages all 40 pad accelerometer readings, then: `tilt_x = atan2(ax, az)`, `tilt_y = atan2(ay, az)`
- Seeds the controller's gyro integration so subsequent tilt tracking starts from the correct offset

### Altitude-Based Interface
- Controller accepts altitude directly (meters), not pressure
- Ground pressure back-calculated from first altitude reading: `P = P0 * (1 - alt/44330)^(1/0.1903)`
- Removed all pressure-to-altitude conversions from the controller

### Physics Fixes (main.rs, rocket_sim.rs)
- Added body drag to main.rs coast simulation (was missing, only had airbrake drag)
- Fixed mass: 51.26 kg (113 lb), was incorrectly 113 kg
- Fixed body Cd: 0.5 (was 0.4)
- Fixed airbrake Cd: 0.3 (was 0.4)
- Body diameter: 6 inches (0.1524 m)
- Both main.rs physics and rocket_sim.rs now use identical drag model:
  ```
  dynamic_pressure = 0.5 * rho * v_axial^2
  fd_body = dynamic_pressure * BODY_CD * BODY_AREA
  fd_brake = dynamic_pressure * AIRBRAKE_CD * airbrake_area
  ```

### Velocity Smoothing (controller.rs)
- `SensorBuffer::get_velocity()` changed from 2-point finite difference to least-squares linear fit
- Buffer size increased from 3 to 10 points for effective noise reduction
- 2-point differences on noisy 0.05s barometer data caused wild oscillations (~189/286 m/s alternating); least-squares over 10 points gives stable velocity

### Apogee Detection Fix (controller.rs)
- Velocity going negative at coast entry caused false apogee retraction
- Added `coast_initialized` guard: retraction only triggers after the sensor buffer has filled with valid coast data
- `is_ready()` requires buffer to be full (`len >= size`) before coast logic activates

### Flight Replay Improvements (main.rs)
- Flight replay loop now prints controller deployment %, predicted apogee, and error during coast
- Malformed CSV rows (empty fields) are skipped gracefully instead of panicking

## Validation Results

### Full Flight Replay (full_flight.csv, no-brakes real data)
- Real apogee: ~3632 m (11,909 ft)
- Controller correctly commands 100% deployment throughout coast
- Predicted apogee near actual apogee tracks closely (~3630 m predicted vs 3632 m real)

### Closed-Loop Simulation (test_25.csv, pad+boost only)
- Target: 3048 m (10,000 ft)
- Simulated apogee with controller: 3112.3 m
- Error: +64.3 m (+2.11%)
- Controller maxes deployment at 100% — airbrake area is the limiting factor

## File Structure
```
src/
  main.rs              - Simulation harness (CSV replay + coast physics)
  controller.rs        - AirbrakeController, SensorBuffer, Phase enum
  rocket_sim.rs        - Forward Euler apogee predictor
  gyro_calibration.rs  - Pad-phase gyro drift computation
  measure_tilt.rs      - Pad-phase accelerometer tilt measurement
```

## Constants
| Parameter | Value | Notes |
|-----------|-------|-------|
| Mass | 51.26 kg | 113 lb |
| Body Cd | 0.5 | |
| Body diameter | 0.1524 m | 6 inches |
| Airbrake Cd | 0.3 | |
| Airbrake area | 0.001848 - 0.021935 m^2 | 2.86 - 34 in^2 |
| Target apogee | 3048 m | 10,000 ft |
| Sensor buffer | 10 points | Least-squares velocity fit |
| Pad calibration | 40 samples | Gyro drift + tilt |
| Baro noise (sim) | 0.02 m RMS | |
| Gyro noise (sim) | 0.07 deg/s RMS | |
