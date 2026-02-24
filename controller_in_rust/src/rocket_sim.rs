/// Simulates coast phase from given conditions and returns predicted apogee (m).
pub fn rocket_sim(x: f64, v: f64, tilt_deg: f64, airbrake: f64) -> f64 {
    const DT: f64 = 0.01;
    const G: f64 = 9.80665;
    const R: f64 = 287.05;
    const L: f64 = 0.0065;
    const MASS: f64 = 16.0;
    const T0: f64 = 288.15;
    const P0_HPA: f64 = 1013.25;

    let altitude_from_pressure = |p: f64| -> f64 {
        (T0 / L) * ((P0_HPA / p).powf((R * L) / G) - 1.0)
    };

    let air_density = |h: f64| -> f64 {
        let p0_pa = P0_HPA * 100.0;
        (p0_pa / (R * T0)) * (1.0 - (L * h / T0)).powf(G / (R * L) - 1.0)
    };

    // Cd constant at 0.3; area varies linearly with deployment
    let drag_coeff_area = |deploy: f64| -> (f64, f64) {
        (0.3, 0.01 + 0.04 * deploy)
    };

    let mut x = x;
    let mut v = v;

    loop {
        let p = P0_HPA * (-x / 8500.0_f64).exp();
        let h = altitude_from_pressure(p);
        let rho = air_density(h);

        // Airspeed along rocket axis (vertical velocity is one leg, axial is hypotenuse)
        let v_air = v.abs() / tilt_deg.abs().to_radians().cos().max(1e-6);

        let (cd, a) = drag_coeff_area(airbrake);
        let fd = 0.5 * rho * v_air * v_air * cd * a;

        let acc = -G - fd / MASS;
        let v_next = v + acc * DT;
        let x_next = x + v * DT + 0.5 * acc * DT * DT;

        if v_next < 0.0 {
            return x_next;
        }

        v = v_next;
        x = x_next;
    }
}
