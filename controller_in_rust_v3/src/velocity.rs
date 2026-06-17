//! Upward velocity estimator.
//!
//! Uses a rolling least-squares slope of the last 5 (altitude, time) samples.
//! Applies a physics-based scaler function to correct for the phase lag of the buffer,
//! accounting for the deceleration due to gravity and aerodynamic drag.

use heapless::Deque;

const BARO_BUFFER_SIZE: usize = 5;

pub struct VelocityEstimator {
    altitudes: Deque<f32, BARO_BUFFER_SIZE>,
    times: Deque<f32, BARO_BUFFER_SIZE>,
    last: f32,
}

impl VelocityEstimator {
    pub fn new() -> Self {
        Self {
            altitudes: Deque::new(),
            times: Deque::new(),
            last: 0.0,
        }
    }

    /// Push a new (altitude, time) sample and return the current upward-velocity estimate.
    pub fn update(&mut self, altitude: f32, time: f32) -> f32 {
        if self.altitudes.is_full() {
            let _ = self.altitudes.pop_front();
            let _ = self.times.pop_front();
        }
        let _ = self.altitudes.push_back(altitude);
        let _ = self.times.push_back(time);

        self.last = self.baro_slope().unwrap_or(self.last);
        self.last
    }

    fn baro_slope(&self) -> Option<f32> {
        let n = self.altitudes.len();
        if n < 2 {
            return None;
        }
        let n_f = n as f32;
        let t_mean: f32 = self.times.iter().sum::<f32>() / n_f;
        let h_mean: f32 = self.altitudes.iter().sum::<f32>() / n_f;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (t, a) in self.times.iter().zip(self.altitudes.iter()) {
            let dt = t - t_mean;
            num += dt * (a - h_mean);
            den += dt * dt;
        }
        if den > 0.0 {
            let v_slope = num / den;
            
            let latest_t = *self.times.back().unwrap();
            let dt_lag = (latest_t - t_mean).max(0.0);
            
            // Fast-forwards the slope to the present to account for gravity and drag deceleration
            // 9.81 = gravity, 0.000099 = drag factor ((1.1 * 0.52 * Area) / (2 * Mass))
            let v_current = v_slope - (9.81 + 0.000099 * v_slope * v_slope) * dt_lag;
            
            Some(v_current)
        } else {
            None
        }
    }
}

impl Default for VelocityEstimator {
    fn default() -> Self {
        Self::new()
    }
}
