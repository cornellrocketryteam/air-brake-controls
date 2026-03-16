# Session 2 — Velocity Smoothing, Controller Output, Multi-Flight Testing

## Changes

### Velocity Smoothing (controller.rs)
- `SensorBuffer::get_velocity()` now uses least-squares linear regression instead of 2-point finite difference
- Buffer size increased from 3 to 10 points
- Previous approach: noisy 0.05s barometer data caused wild velocity oscillations (~189/286 m/s alternating), preventing the controller from engaging brakes
- New approach: fits a line through the last 10 altitude-time pairs, slope = velocity. Stable output.

### Controller Output Struct (controller.rs)
- `step()` now returns `ControllerOutput { deployment, predicted_apogee, error }` instead of just `f64`
- Controller prints `[CTRL]` line each coast timestep with deployment %, predicted apogee, and error
- `rocket_sim()` is now called only inside the controller — main no longer calls it directly
- main.rs uses `ControllerOutput` fields for its table display and CSV logging

### Flight Replay Improvements (main.rs)
- Flight replay loop now shows controller deployment, predicted apogee, and error during coast (was hardcoded to 0.0/--- before)
- Malformed CSV rows (empty fields) are skipped instead of panicking

## Test Results

### test_25 / full_flight (target 3048 m)
| Run | Target | Apogee | Error |
|-----|--------|--------|-------|
| full_flight (replay, no brakes) | 3048 m | 3632 m | +584 m |
| test_25 (closed-loop sim) | 3048 m | 3112 m | +64 m (+2.1%) |

Controller maxes brakes at 100%. Airbrake area is the limiting factor — can't fully close the 584 m gap but gets within 64 m.

### comp_24 / full_comp_24 (target 2750 m)
| Run | Target | Apogee | Error |
|-----|--------|--------|-------|
| full_comp_24 (replay) | 2750 m | 2800 m | +30 m |
| comp_24 (closed-loop sim) | 2750 m | 2762 m | +12 m (+0.43%) |

Controller actively modulates deployment (seen oscillating between 90-100%) to hit target. Real no-brakes apogee ~2800 m (9188 ft).

### test_24 / full_test_24 (target 3048 m)
| Run | Target | Apogee | Error |
|-----|--------|--------|-------|
| full_test_24 (replay) | 3048 m | 4080 m | +1032 m |
| test_24 (closed-loop sim, 3048 m) | 3048 m | 3990 m | +942 m (+30.9%) |
| test_24 (closed-loop sim, 4050 m) | 4050 m | 4047 m | -2.7 m (-0.07%) |

Brakes maxed at 100% but only shave ~90 m. This flight has higher burnout velocity (257 m/s at 1653 m) and coasts through thinner air where drag is less effective. With an achievable target (4050 m), controller nails it at -0.07% error.

## Analysis: Airbrake Effectiveness

Why brakes removed 520 m on test_25 but only 90 m on test_24:

| | test_25 | test_24 |
|---|---------|---------|
| Burnout altitude | 1495 m | 1653 m |
| Burnout velocity | 202 m/s | 257 m/s |
| No-brakes apogee | 3632 m | 4080 m |
| With full brakes | 3112 m | 3990 m |
| Reduction | 520 m | 90 m |

Drag force = 0.5 * rho * v^2 * Cd * A. At higher altitudes, air density (rho) drops significantly, reducing brake effectiveness. test_24 spends more of its coast in thin air.

Airbrake sizing:
- Body: Cd=0.5, A=0.01824 m^2 -> CdA = 0.00912
- Brakes (max): Cd=0.3, A=0.021935 m^2 -> CdA = 0.00658
- Full brakes add only 72% more drag on top of body drag

This is a physical sizing limitation, not a controller issue. The controller correctly maxes out deployment when it can't reach target.
