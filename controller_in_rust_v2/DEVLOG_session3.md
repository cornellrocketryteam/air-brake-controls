# Session 3 — Flight Software Interface, Testing, Multi-Flight Validation

## Changes

### ControllerOutput Struct (controller.rs)
- `step()` now returns `ControllerOutput { deployment, predicted_apogee, error }` instead of bare `f64`
- Controller computes predicted apogee internally via `rocket_sim()` at every coast step
- Controller prints `[CTRL]` line each coast timestep with deployment %, predicted apogee, and error
- `rocket_sim()` is no longer called from main.rs — all prediction lives inside the controller

### Flight Software Interface — airbrakes.rs (NEW)
- `AirbrakeSystem` wraps the full pipeline: calibration, drift correction, and controller stepping
- Single function `execute()` called once per sensor reading by flight software
- Inputs: time, altitude, gyro_x, gyro_y, accel_x, accel_y, accel_z, phase
- Outputs: `AirbrakeOutput { deployment, predicted_apogee, error }`
- Calibration (gyro drift + tilt) happens automatically during first 40 pad readings
- Drift correction applied to all non-pad gyro readings
- Flight software loop:
  ```rust
  let mut airbrakes = AirbrakeSystem::new();
  loop {
      let out = airbrakes.execute(time, alt, gx, gy, ax, ay, az, phase);
      set_brake_position(out.deployment);
  }
  ```

### Library Crate (lib.rs)
- Added `lib.rs` to expose all modules for integration testing
- Crate now builds as both binary (simulation) and library (for flight software / tests)

### Integration Tests (tests/test_airbrakes.rs)
Four tests validating `AirbrakeSystem::execute()`:
1. **test_airbrakes_engages_on_overshoot** — Replays comp_24 CSV through execute() + coast physics sim, verifies apogee in expected range (2700–3000 m)
2. **test_airbrakes_retracts_when_below_target** — Synthetic low-energy data, verifies deployment stays at 0 when predicted apogee is below target
3. **test_airbrakes_calibration_happens_once** — Feeds 50 pad readings (more than the 40 needed), verifies no panic and clean transition to boost
4. **test_full_flight_replay** — Replays full_comp_24 through execute(), verifies brakes stay retracted when real apogee (~2800 m) is below default target (3048 m)

## Test Results

### test_24 / full_test_24
| Run | Target | Apogee | Error | Notes |
|-----|--------|--------|-------|-------|
| full_test_24 (replay) | 3048 m | 4080 m | +1032 m | Real no-brakes data |
| test_24 (sim, 3048 m) | 3048 m | 3990 m | +942 m | Brakes maxed, only shave 90 m |
| test_24 (sim, 4050 m) | 4050 m | 4047 m | -2.7 m (-0.07%) | Achievable target |

### comp_24 / full_comp_24
| Run | Target | Apogee | Error |
|-----|--------|--------|-------|
| full_comp_24 (replay) | 2750 m | 2800 m | +30 m |
| comp_24 (sim) | 2750 m | 2762 m | +12 m (+0.43%) |

### Airbrake Effectiveness Analysis
test_24 brakes only removed 90 m vs test_25's 520 m because:
- Higher burnout altitude (1653 m vs 1495 m) and velocity (257 m/s vs 202 m/s)
- More coast time spent in thin air where drag is less effective
- Airbrake CdA (0.00658) only adds 72% on top of body CdA (0.00912) — physical sizing limitation

## File Structure
```
src/
  airbrakes.rs         - Flight software interface (AirbrakeSystem::execute)
  controller.rs        - Core controller, returns ControllerOutput
  rocket_sim.rs        - Forward Euler apogee predictor
  gyro_calibration.rs  - Pad-phase gyro drift computation
  measure_tilt.rs      - Pad-phase accelerometer tilt measurement
  lib.rs               - Library crate root (exposes all modules)
  main.rs              - Simulation harness (CSV replay + coast physics)
tests/
  test_airbrakes.rs    - Integration tests for AirbrakeSystem
```
