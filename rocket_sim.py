import math
def rocket_sim(x, v, tilt_deg,airbrake):
    # --------------------
    # Constants
    # --------------------
    dt = 0.01          # time step (s)
    g = 9.80665        # gravity (m/s^2)
    R = 287.05         # J/(kg*K)
    L = 0.0065         # K/m
    mass = 16.0        # kg

    # --------------------
    # Initial conditions
    # --------------------
    x = x            # height (m)
    v = v         # vertical velocity (m/s)
    time = 0.0

    # --------------------
    # Ground conditions
    # --------------------
    T0 = 288.15        # K
    P0_hpa = 1013.25   # hPa

    # --------------------
    # Inputs (example)
    # --------------------
    tilt_deg = tilt_deg     # degrees off vertical (DIRECT INPUT)
    airbrake = airbrake     # 0.0 – 1.0

    # --------------------
    # Helper functions
    # --------------------
    def altitude_from_pressure(P, P0, T0):
        return (T0 / L) * ((P0 / P)**((R * L) / g) - 1.0)


    def air_density(h, P0_pa, T0):
        return (P0_pa / (R * T0)) * (1 - (L * h / T0))**((g / (R * L)) - 1)


    def drag_coeff_area(deploy):
        Cd = 0.5 + 1.5 * deploy
        A = 0.01 + 0.04 * deploy
        return Cd, A


    # --------------------
    # Simulation loop
    # --------------------
    while True:
        # --- Pressure model (for simulation only) ---
        P = P0_hpa * math.exp(-x / 8500.0)

        # --- Barometric altitude ---
        h = altitude_from_pressure(P, P0_hpa, T0)

        # --- Air density ---
        rho = air_density(h, P0_hpa * 100.0, T0)

        # --- Airspeed along rocket axis ---
        v_air = abs(v) / max(math.cos(math.radians(abs(tilt_deg))), 1e-6)

        # --- Drag ---
        Cd, A = drag_coeff_area(airbrake)
        Fd = 0.5 * rho * v_air**2 * Cd * A

        # --- Acceleration ---
        a = -g - Fd / mass

        # --- Kinematics ---
        v_next = v + a * dt
        x_next = x + v * dt + 0.5 * a * dt**2

        # --- Apogee condition ---
        if v_next < 0:
            print(f"Apogee reached at {x:.2f} m after {time:.2f} s")
            break

        # --- Update ---
        v = v_next
        x = x_next
        time += dt
    return x_next