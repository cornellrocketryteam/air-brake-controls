use std::collections::VecDeque;
use crate::rocket_sim::rocket_sim;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
pub const DT: f64 = 0.01;
pub const TARGET_APOGEE: f64 = 3048.0;
pub const R: f64 = 287.05;
pub const G: f64 = 9.80665;
pub const L: f64 = 0.0065;
pub const GROUND_TEMP_K: f64 = 288.15;
pub const AIRBRAKE_MIN: f64 = 0.0;
pub const AIRBRAKE_MAX: f64 = 1.0;
pub const AIRBRAKE_CD: f64 = 0.3;
pub const AIRBRAKE_AREA_MIN: f64 = 0.001848;   // 2.86479 in²
pub const AIRBRAKE_AREA_MAX: f64 = 0.021935;   // 34 in²
pub const MAX_TILT_DEG: f64 = 50.0;
pub const SEA_LEVEL_PRESSURE_PA: f64 = 101325.0;

// -----------------------------------------------------------------------------
// Flight phase — passed in by the flight computer / simulator
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    Pad,
    Boost,
    Coast,
}

// -----------------------------------------------------------------------------
// Sensor data packet (barometer + gyroscope)
// -----------------------------------------------------------------------------
pub struct SensorData {
    pub time: f64,
    pub altitude: f64,
    pub gyro_x: f64,
    pub gyro_y: f64,
    pub phase: Phase,
}

// -----------------------------------------------------------------------------
// Sensor buffer — rolling window of last 3 readings
// -----------------------------------------------------------------------------
pub struct SensorBuffer {
    altitudes: VecDeque<f64>,
    timestamps: VecDeque<f64>,
    size: usize,
}

impl SensorBuffer {
    pub fn new(size: usize) -> Self {
        SensorBuffer {
            altitudes: VecDeque::with_capacity(size + 1),
            timestamps: VecDeque::with_capacity(size + 1),
            size,
        }
    }

    pub fn add(&mut self, altitude: f64, timestamp: f64) {
        if self.altitudes.len() >= self.size {
            self.altitudes.pop_front();
            self.timestamps.pop_front();
        }
        self.altitudes.push_back(altitude);
        self.timestamps.push_back(timestamp);
    }

    pub fn is_ready(&self) -> bool {
        self.altitudes.len() >= self.size
    }

    /// Velocity via least-squares linear fit over all buffered points.
    /// slope = Σ((t_i - t̄)(h_i - h̄)) / Σ((t_i - t̄)²)
    pub fn get_velocity(&self) -> f64 {
        let n = self.altitudes.len();
        if n < 2 {
            return 0.0;
        }

        let n_f = n as f64;
        let t_mean: f64 = self.timestamps.iter().sum::<f64>() / n_f;
        let h_mean: f64 = self.altitudes.iter().sum::<f64>() / n_f;

        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let dt = self.timestamps[i] - t_mean;
            num += dt * (self.altitudes[i] - h_mean);
            den += dt * dt;
        }

        if den > 0.0 { num / den } else { 0.0 }
    }

    pub fn last_altitude(&self) -> f64 {
        self.altitudes.back().copied().unwrap_or(0.0)
    }
}

// -----------------------------------------------------------------------------
// Aerodynamic helpers
// -----------------------------------------------------------------------------
pub fn air_density(altitude: f64, p0: f64, t0: f64) -> f64 {
    let t = (t0 - L * altitude).max(1.0);
    let p = p0 * (t / t0).powf(G / (R * L));
    (p / (R * t)).max(0.001)
}

pub fn deployment_to_area(deployment: f64) -> f64 {
    AIRBRAKE_AREA_MIN + (AIRBRAKE_AREA_MAX - AIRBRAKE_AREA_MIN) * deployment
}

// -----------------------------------------------------------------------------
// Controller output — returned every step
// -----------------------------------------------------------------------------
pub struct ControllerOutput {
    pub deployment: f64,
    pub predicted_apogee: f64,
    pub error: f64,
}

// -----------------------------------------------------------------------------
// Main controller
// -----------------------------------------------------------------------------
pub struct AirbrakeController {
    pub target_apogee: f64,
    pub ground_pressure: f64,
    pub ground_temp: f64,
    ground_pressure_calibrated: bool,
    pub sensor_buffer: SensorBuffer,
    pub current_airbrake: f64,
    pub beginning_tilt: f64,
    pub integrated_tilt_x: f64,
    pub integrated_tilt_y: f64,
    pub integrated_tilt: f64,
    pub coast_initialized: bool,
    previous_time: Option<f64>,
    pub burnout_velocity: f64,
}

impl AirbrakeController {
    pub fn new(target_apogee: f64, ground_temp: f64) -> Self {
        AirbrakeController {
            target_apogee,
            ground_pressure: 0.0,
            ground_temp,
            ground_pressure_calibrated: false,
            sensor_buffer: SensorBuffer::new(10),
            current_airbrake: 0.0,
            beginning_tilt: 0.0,
            integrated_tilt_x: 0.0,
            integrated_tilt_y: 0.0,
            integrated_tilt: 0.0,
            coast_initialized: false,
            previous_time: None,
            burnout_velocity: 0.0,
        }
    }

    pub fn set_beginning_tilt(&mut self, tilt_x_deg: f64, tilt_y_deg: f64) {
        self.integrated_tilt_x = tilt_x_deg;
        self.integrated_tilt_y = tilt_y_deg;    
        self.integrated_tilt = (tilt_x_deg * tilt_x_deg + tilt_y_deg * tilt_y_deg).sqrt();
        self.beginning_tilt = self.integrated_tilt;
    }

    fn integrate_gyroscope(&mut self, gyro_x: f64, gyro_y: f64, dt: f64) {
        self.integrated_tilt_x += gyro_x * dt;
        self.integrated_tilt_y += gyro_y * dt;
        self.integrated_tilt = (self.integrated_tilt_x * self.integrated_tilt_x
            + self.integrated_tilt_y * self.integrated_tilt_y)
            .sqrt();
    }

    /// Binary search (20 iters) for deployment that hits target apogee.
    fn find_optimal_deployment(&self, height: f64, velocity: f64, tilt_deg: f64) -> f64 {
        let predicted_no_brakes = rocket_sim(height, velocity, tilt_deg, 0.0, self.ground_pressure);
        if predicted_no_brakes <= self.target_apogee {
            return AIRBRAKE_MIN;
        }

        let mut lo = AIRBRAKE_MIN;
        let mut hi = AIRBRAKE_MAX;
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            let predicted = rocket_sim(height, velocity, tilt_deg, mid, self.ground_pressure);
            if predicted >= self.target_apogee {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo.min(AIRBRAKE_MAX)
    }

    pub fn step(&mut self, sensor_data: &SensorData) -> ControllerOutput {
        let current_time = sensor_data.time;

        if !self.ground_pressure_calibrated {
            self.ground_pressure = SEA_LEVEL_PRESSURE_PA
                * (1.0 - sensor_data.altitude / 44330.0).powf(1.0 / 0.1903);
            self.ground_pressure_calibrated = true;
            println!(
                "[{:.2}s] Ground pressure calibrated: {:.1} Pa (from altitude {:.1} m)",
                current_time, self.ground_pressure, sensor_data.altitude
            );
        }

        let dt = self.previous_time.map_or(DT, |pt| current_time - pt);
        self.previous_time = Some(current_time);

        self.sensor_buffer.add(sensor_data.altitude, current_time);
        self.integrate_gyroscope(sensor_data.gyro_x, sensor_data.gyro_y, dt);

        let mut predicted_apogee = 0.0;

        match sensor_data.phase {
            Phase::Pad => {}

            Phase::Boost => {
                if self.sensor_buffer.is_ready() {
                    self.burnout_velocity = self.sensor_buffer.get_velocity();
                }
            }

            Phase::Coast => {
                let height = sensor_data.altitude;
                let velocity = if !self.coast_initialized {
                    self.burnout_velocity
                } else {
                    self.sensor_buffer.get_velocity()
                };
                let tilt = self.integrated_tilt;

                if !self.coast_initialized && self.sensor_buffer.is_ready() {
                    self.coast_initialized = true;
                }

                if self.coast_initialized && velocity <= 0.0 {
                    self.current_airbrake = AIRBRAKE_MIN;
                    predicted_apogee = height;
                    let error = predicted_apogee - self.target_apogee;
                    println!(
                        "[CTRL] t={:.2}s  deploy={:.1}%  pred={:.1}m  err={:+.1}m",
                        current_time, self.current_airbrake * 100.0, predicted_apogee, error
                    );
                    return ControllerOutput {
                        deployment: self.current_airbrake,
                        predicted_apogee,
                        error,
                    };
                }

                if tilt > MAX_TILT_DEG {
                    self.current_airbrake = AIRBRAKE_MIN;
                    predicted_apogee = rocket_sim(height, velocity, tilt, self.current_airbrake, self.ground_pressure);
                    let error = predicted_apogee - self.target_apogee;
                    println!(
                        "[CTRL] t={:.2}s  FAILSAFE tilt={:.1}°  deploy={:.1}%  pred={:.1}m  err={:+.1}m",
                        current_time, tilt, self.current_airbrake * 100.0, predicted_apogee, error
                    );
                    return ControllerOutput {
                        deployment: self.current_airbrake,
                        predicted_apogee,
                        error,
                    };
                }

                self.current_airbrake = self.find_optimal_deployment(height, velocity, tilt);
                predicted_apogee = rocket_sim(height, velocity, tilt, self.current_airbrake, self.ground_pressure);
                let error = predicted_apogee - self.target_apogee;
                println!(
                    "[CTRL] t={:.2}s  deploy={:.1}%  pred={:.1}m  err={:+.1}m",
                    current_time, self.current_airbrake * 100.0, predicted_apogee, error
                );
            }
        }

        let error = predicted_apogee - self.target_apogee;
        ControllerOutput {
            deployment: self.current_airbrake,
            predicted_apogee,
            error,
        }
    }
}
