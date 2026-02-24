use std::collections::VecDeque;
use crate::pid::Pid;
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
pub const GROUND_PRESSURE_PA: f64 = 101325.0;
pub const AIRBRAKE_MIN: f64 = 0.0;
pub const AIRBRAKE_MAX: f64 = 1.0;
pub const AIRBRAKE_CD: f64 = 0.3;
pub const AIRBRAKE_AREA_MIN: f64 = 0.01;
pub const AIRBRAKE_AREA_MAX: f64 = 0.05;
pub const MAX_TILT_DEG: f64 = 50.0;
pub const LAUNCH_ACCEL_THRESHOLD: f64 = 5.0;
pub const KP: f64 = 0.008;
pub const KI: f64 = 0.0002;
pub const KD: f64 = 0.001;
pub const DRAG_SCALE_FACTOR: f64 = 5.0;

// -----------------------------------------------------------------------------
// Flight phase
// -----------------------------------------------------------------------------
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    PreLaunch,
    Launch,
    CoastInit,
    CoastActive,
}

// -----------------------------------------------------------------------------
// Sensor data packet
// -----------------------------------------------------------------------------
pub struct SensorData {
    pub time: f64,
    pub pressure: f64,
    pub gyro_x: f64,
    pub gyro_y: f64,
    pub gyro_z: f64,
}

// -----------------------------------------------------------------------------
// Sensor buffer — rolling window of last 3 readings
// -----------------------------------------------------------------------------
pub struct SensorBuffer {
    altitudes: VecDeque<f64>,
    gyro_readings: VecDeque<(f64, f64, f64)>,
    timestamps: VecDeque<f64>,
    size: usize,
}

impl SensorBuffer {
    pub fn new(size: usize) -> Self {
        SensorBuffer {
            altitudes: VecDeque::with_capacity(size + 1),
            gyro_readings: VecDeque::with_capacity(size + 1),
            timestamps: VecDeque::with_capacity(size + 1),
            size,
        }
    }

    pub fn add(&mut self, altitude: f64, gyro: (f64, f64, f64), timestamp: f64) {
        if self.altitudes.len() >= self.size {
            self.altitudes.pop_front();
            self.gyro_readings.pop_front();
            self.timestamps.pop_front();
        }
        self.altitudes.push_back(altitude);
        self.gyro_readings.push_back(gyro);
        self.timestamps.push_back(timestamp);
    }

    pub fn is_ready(&self) -> bool {
        self.altitudes.len() >= 3
    }

    pub fn get_velocity(&self) -> f64 {
        let n = self.altitudes.len();
        if n < 2 {
            return 0.0;
        }
        let dh = self.altitudes[n - 1] - self.altitudes[n - 2];
        let dt = self.timestamps[n - 1] - self.timestamps[n - 2];
        if dt > 0.0 { dh / dt } else { 0.0 }
    }

    pub fn get_acceleration(&self) -> f64 {
        let n = self.altitudes.len();
        if n < 3 {
            return 0.0;
        }
        let dt1 = self.timestamps[n - 2] - self.timestamps[n - 3];
        let dt2 = self.timestamps[n - 1] - self.timestamps[n - 2];
        if dt1 <= 0.0 || dt2 <= 0.0 {
            return 0.0;
        }
        let v1 = (self.altitudes[n - 2] - self.altitudes[n - 3]) / dt1;
        let v2 = (self.altitudes[n - 1] - self.altitudes[n - 2]) / dt2;
        let dt_avg = (dt1 + dt2) / 2.0;
        (v2 - v1) / dt_avg
    }

    pub fn last_altitude(&self) -> f64 {
        self.altitudes.back().copied().unwrap_or(0.0)
    }
}

// -----------------------------------------------------------------------------
// Aerodynamic helpers
// -----------------------------------------------------------------------------
pub fn pressure_to_altitude(pressure_pa: f64, p0: f64, t0: f64) -> f64 {
    if pressure_pa <= 0.0 {
        return 0.0;
    }
    let exponent = (R * L) / G;
    let altitude = (t0 / L) * (1.0 - (pressure_pa / p0).powf(exponent));
    altitude.max(0.0)
}

pub fn altitude_to_temperature(altitude: f64, t0: f64) -> f64 {
    (t0 - L * altitude).max(1.0)
}

pub fn air_density(altitude: f64, p0: f64, t0: f64) -> f64 {
    let t = altitude_to_temperature(altitude, t0);
    let p = p0 * (t / t0).powf(G / (R * L));
    (p / (R * t)).max(0.001)
}

pub fn deployment_to_area(deployment: f64) -> f64 {
    AIRBRAKE_AREA_MIN + (AIRBRAKE_AREA_MAX - AIRBRAKE_AREA_MIN) * deployment
}

pub fn deployment_to_drag(deployment: f64, velocity: f64, altitude: f64) -> f64 {
    let rho = air_density(altitude, GROUND_PRESSURE_PA, GROUND_TEMP_K);
    let a = deployment_to_area(deployment);
    0.5 * rho * velocity * velocity * AIRBRAKE_CD * a
}

pub fn drag_force_to_deployment(drag_force_n: f64, velocity_mps: f64, altitude_m: f64) -> f64 {
    if velocity_mps <= 0.0 {
        return AIRBRAKE_MIN;
    }
    let rho = air_density(altitude_m, GROUND_PRESSURE_PA, GROUND_TEMP_K);
    let k = 0.5 * rho * velocity_mps * velocity_mps;
    if k <= 0.0 {
        return AIRBRAKE_MIN;
    }
    let deployment =
        (drag_force_n / (k * AIRBRAKE_CD) - AIRBRAKE_AREA_MIN) / (AIRBRAKE_AREA_MAX - AIRBRAKE_AREA_MIN);
    deployment.clamp(AIRBRAKE_MIN, AIRBRAKE_MAX)
}

// -----------------------------------------------------------------------------
// Main controller
// -----------------------------------------------------------------------------
pub struct AirbrakeController {
    pub target_apogee: f64,
    pub ground_pressure: f64,
    pub ground_temp: f64,
    pub sensor_buffer: SensorBuffer,
    pid: Pid,
    pub current_airbrake: f64,
    pub integrated_tilt: f64,
    pub phase: Phase,
    pub launched: bool,
    pub connection_lost: bool,
    pub last_data_time: f64,
    previous_time: Option<f64>,
}

impl AirbrakeController {
    pub fn new(target_apogee: f64, ground_pressure: f64, ground_temp: f64) -> Self {
        AirbrakeController {
            target_apogee,
            ground_pressure,
            ground_temp,
            sensor_buffer: SensorBuffer::new(3),
            pid: Pid::new(KP, KI, KD, -AIRBRAKE_MAX, AIRBRAKE_MAX),
            current_airbrake: 0.0,
            integrated_tilt: 0.0,
            phase: Phase::PreLaunch,
            launched: false,
            connection_lost: false,
            last_data_time: 0.0,
            previous_time: None,
        }
    }

    fn process_sensors(&self, sensor_data: &SensorData) -> (f64, f64, f64, f64) {
        let altitude =
            pressure_to_altitude(sensor_data.pressure, self.ground_pressure, self.ground_temp);
        (altitude, sensor_data.gyro_x, sensor_data.gyro_y, sensor_data.gyro_z)
    }

    fn integrate_gyroscope(&mut self, gyro_x: f64, gyro_y: f64, dt: f64) {
        let tilt_rate = (gyro_x * gyro_x + gyro_y * gyro_y).sqrt();
        self.integrated_tilt += tilt_rate * dt;
    }

    fn check_failsafes(&self, current_velocity: f64) -> Option<String> {
        if self.integrated_tilt > MAX_TILT_DEG {
            return Some(format!(
                "FAILSAFE: Tilt {:.1}° exceeds {:.1}°. Fully retracting.",
                self.integrated_tilt, MAX_TILT_DEG
            ));
        }
        if self.connection_lost && current_velocity > 0.0 {
            return Some("FAILSAFE: Connection lost. Fully retracting.".to_string());
        }
        None
    }

    fn airbrake_adjustment_loop(&self, height: f64, velocity: f64, tilt: f64) -> f64 {
        let mut lo = AIRBRAKE_MIN;
        let mut hi = AIRBRAKE_MAX;
        for _ in 0..20 {
            let mid = (lo + hi) / 2.0;
            let predicted = rocket_sim(height, velocity, tilt, mid);
            if predicted >= self.target_apogee {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo.min(AIRBRAKE_MAX)
    }

    fn calculate_drag_adjustment(
        &self,
        pid_output: f64,
        current_velocity: f64,
        current_altitude: f64,
        current_deployment: f64,
    ) -> f64 {
        let current_drag =
            deployment_to_drag(current_deployment, current_velocity, current_altitude);
        let target_drag = (current_drag - pid_output * DRAG_SCALE_FACTOR).max(0.0);
        drag_force_to_deployment(target_drag, current_velocity, current_altitude)
    }

    fn command_airbrakes(&mut self, deployment: f64) {
        self.current_airbrake = deployment.clamp(AIRBRAKE_MIN, AIRBRAKE_MAX);
    }

    /// Process one sensor reading through all phase logic. Returns current deployment (0–1).
    pub fn step(&mut self, sensor_data: &SensorData) -> f64 {
        let current_time = sensor_data.time;
        let dt = self.previous_time.map_or(DT, |pt| current_time - pt);
        self.previous_time = Some(current_time);
        self.last_data_time = current_time;

        let (altitude, gyro_x, gyro_y, gyro_z) = self.process_sensors(sensor_data);
        self.sensor_buffer
            .add(altitude, (gyro_x, gyro_y, gyro_z), current_time);

        if self.launched {
            self.integrate_gyroscope(gyro_x, gyro_y, dt);
        }

        let calculated_accel = self.sensor_buffer.get_acceleration();

        match self.phase {
            Phase::PreLaunch => {
                if self.sensor_buffer.is_ready() && calculated_accel > LAUNCH_ACCEL_THRESHOLD {
                    self.phase = Phase::Launch;
                    self.launched = true;
                    println!(
                        "[{:.2}s] Launch detected! (accel={:.1} m/s²)",
                        current_time, calculated_accel
                    );
                }
            }

            Phase::Launch => {
                if calculated_accel < 0.0 {
                    self.phase = Phase::CoastInit;
                    println!(
                        "[{:.2}s] Coast phase detected. Initializing control.",
                        current_time
                    );
                }
            }

            Phase::CoastInit => {
                if self.sensor_buffer.is_ready() {
                    self.phase = Phase::CoastActive;
                    let height = altitude;
                    let velocity = self.sensor_buffer.get_velocity();
                    let tilt = self.integrated_tilt;

                    let predicted = rocket_sim(height, velocity, tilt, 0.0);
                    println!(
                        "[{:.2}s] Predicted apogee (no brakes): {:.1} m",
                        current_time, predicted
                    );

                    if predicted <= self.target_apogee {
                        println!(
                            "[{:.2}s] Predicted apogee too low. Not deploying brakes.",
                            current_time
                        );
                        self.current_airbrake = AIRBRAKE_MIN;
                    } else {
                        self.current_airbrake =
                            self.airbrake_adjustment_loop(height, velocity, tilt);
                        println!(
                            "[{:.2}s] Initial airbrake deployment: {:.1}%",
                            current_time,
                            self.current_airbrake * 100.0
                        );
                        let d = self.current_airbrake;
                        self.command_airbrakes(d);
                    }
                }
            }

            Phase::CoastActive => {
                let height = altitude;
                let velocity = self.sensor_buffer.get_velocity();
                let tilt = self.integrated_tilt;

                if velocity <= 0.0 {
                    println!(
                        "[{:.2}s] Apogee detected by controller at {:.1} m. Retracting.",
                        current_time, height
                    );
                    self.current_airbrake = AIRBRAKE_MIN;
                    self.command_airbrakes(AIRBRAKE_MIN);
                    return self.current_airbrake;
                }

                if let Some(msg) = self.check_failsafes(velocity) {
                    println!("[{:.2}s] {}", current_time, msg);
                    self.current_airbrake = AIRBRAKE_MIN;
                    self.command_airbrakes(AIRBRAKE_MIN);
                    return self.current_airbrake;
                }

                let predicted = rocket_sim(height, velocity, tilt, self.current_airbrake);
                let pid_output = self.pid.update(self.target_apogee, predicted, dt);
                let new_deployment = self.calculate_drag_adjustment(
                    pid_output,
                    velocity,
                    height,
                    self.current_airbrake,
                );
                self.command_airbrakes(new_deployment);
            }
        }

        self.current_airbrake
    }
}
