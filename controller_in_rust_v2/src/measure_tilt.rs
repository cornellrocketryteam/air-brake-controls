/// Initial tilt measurement from accelerometer pad-phase readings.
///
/// While the rocket is stationary on the pad, the accelerometer reads only
/// gravity.  Any deviation from (0, 0, 1g) means the rocket is tilted from
/// vertical.  We average all 40 pad samples to reduce noise, then decompose
/// the tilt into its x and y components:
///
///   tilt_x = atan2(ax, az)
///   tilt_y = atan2(ay, az)

pub struct BeginningTilt {
    pub x_deg: f64,
    pub y_deg: f64,
}

/// `readings` is a slice of (accel_x, accel_y, accel_z) in g's.
/// Returns the x and y tilt components in degrees.
pub fn measure_tilt(readings: &[(f64, f64, f64)]) -> BeginningTilt {
    assert!(!readings.is_empty(), "measure_tilt requires at least one reading");

    let n = readings.len() as f64;
    let ax = readings.iter().map(|r| r.0).sum::<f64>() / n;
    let ay = readings.iter().map(|r| r.1).sum::<f64>() / n;
    let az = readings.iter().map(|r| r.2).sum::<f64>() / n;

    BeginningTilt {
        x_deg: ax.atan2(az).to_degrees(),
        y_deg: ay.atan2(az).to_degrees(),
    }
}
