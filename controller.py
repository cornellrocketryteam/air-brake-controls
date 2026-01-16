"""
Airbrake Controller Architecture
Reads raw sensor data from BMP390 barometer and ICM-42688-P gyroscope via CSV.
Manages airbrake deployment using PID control and rocket simulation for apogee prediction.
"""

import csv
import math
from collections import deque
from PID import PID
from rocket_sim import rocket_sim

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DT = 0.01  # Time step (s)
TARGET_APOGEE = 3048.0  # Target apogee in meters (adjust as needed)

# Atmospheric constants
R = 287.05  # Specific gas constant for dry air (J/(kg·K))
G = 9.80665  # Gravity (m/s^2)
L = 0.0065  # Temperature lapse rate (K/m) - Metric for how T changes with Height

# Ground conditions (calibrate before launch)
GROUND_TEMP_K = 288.15  # Ground temperature (K) - standard sea level
GROUND_PRESSURE_PA = 101325.0  # Ground pressure (Pa) - standard sea level

# Airbrake limits
AIRBRAKE_MIN = 0.0
AIRBRAKE_MAX = 1.0
AIRBRAKE_RETRACT_STEP = 0.05  # Small step for adjustment loop

# Airbrake aerodynamic characteristics (linear interpolation)
# Deployment 0.0 -> Area = 1 m², Cd = 0.2
# Deployment 0.5 -> Area = 2 m², Cd = 0.3
# Deployment 1.0 -> Area = 3 m², Cd = 0.4
AIRBRAKE_AREA_MIN = 1.0      # Area at 0% deployment (m²)
AIRBRAKE_AREA_MAX = 3.0      # Area at 100% deployment (m²)
AIRBRAKE_CD_MIN = 0.2        # Cd at 0% deployment
AIRBRAKE_CD_MAX = 0.4        # Cd at 100% deployment

# Failsafe thresholds
MAX_TILT_DEG = 50.0

# Launch detection threshold
LAUNCH_ACCEL_THRESHOLD = 5.0  # m/s^2 - acceleration to detect launch

# PID tuning parameters (adjust based on testing)
KP = 0.001
KI = 0.0001
KD = 0.0005


# -----------------------------------------------------------------------------
# Sensor Conversion Functions
# -----------------------------------------------------------------------------

def pressure_to_altitude(pressure_pa, P0=GROUND_PRESSURE_PA, T0=GROUND_TEMP_K):
    """
    Convert BMP390 pressure reading to altitude using barometric formula.

    Args:
        pressure_pa: Pressure in Pascals from BMP390
        P0: Ground-level pressure in Pascals (calibrate before launch)
        T0: Ground-level temperature in Kelvin

    Returns:
        Altitude in meters above ground level
    """
    if pressure_pa <= 0:
        return 0.0

    # Barometric formula: h = (T0/L) * (1 - (P/P0)^(R*L/g))
    exponent = (R * L) / G
    altitude = (T0 / L) * (1.0 - (pressure_pa / P0) ** exponent)
    return max(0.0, altitude)


def gyro_to_angular_rate(gyro_raw, scale_factor=131.0):
    """
    Convert ICM-42688-P raw gyroscope reading to degrees per second.

    The ICM-42688-P outputs raw values that need scaling based on configured range.
    Default scale factors for different ranges:
        ±250 dps:  131 LSB/dps
        ±500 dps:  65.5 LSB/dps
        ±1000 dps: 32.8 LSB/dps
        ±2000 dps: 16.4 LSB/dps

    Args:
        gyro_raw: Raw gyroscope value from sensor
        scale_factor: LSB per degree/second (depends on configured range)

    Returns:
        Angular rate in degrees per second
    """
    return gyro_raw / scale_factor


# -----------------------------------------------------------------------------
# Aerodynamic Functions
# -----------------------------------------------------------------------------

def altitude_to_temperature(altitude, T0=GROUND_TEMP_K):
    """
    Calculate temperature at altitude using ISA lapse rate.

    Args:
        altitude: Altitude in meters
        T0: Ground-level temperature in Kelvin

    Returns:
        Temperature in Kelvin
    """
    T = T0 - L * altitude
    return max(T, 1.0)  # Prevent non-positive temperature


def air_density(altitude, P0=GROUND_PRESSURE_PA, T0=GROUND_TEMP_K):
    """
    Calculate air density at a given altitude using ISA model.

    Args:
        altitude: Altitude in meters
        P0: Ground-level pressure in Pascals
        T0: Ground-level temperature in Kelvin

    Returns:
        Air density in kg/m³
    """
    T = altitude_to_temperature(altitude, T0)

    # Pressure at altitude
    P = P0 * (T / T0) ** (G / (R * L))

    # Density from ideal gas law: rho = P / (R * T)
    rho = P / (R * T)
    return max(rho, 0.001)  # Minimum density to prevent issues


def deployment_to_area(deployment):
    """
    Calculate airbrake frontal area as function of deployment.

    Linear interpolation:
        0% deployment -> 1 m²
        100% deployment -> 3 m²

    Args:
        deployment: Airbrake deployment (0.0 to 1.0)

    Returns:
        Frontal area in m²
    """
    # A(d) = A_min + (A_max - A_min) * d = 1 + 2*d
    return AIRBRAKE_AREA_MIN + (AIRBRAKE_AREA_MAX - AIRBRAKE_AREA_MIN) * deployment


def deployment_to_cd(deployment):
    """
    Calculate drag coefficient as function of deployment.

    Linear interpolation:
        0% deployment -> Cd = 0.2
        100% deployment -> Cd = 0.4

    Args:
        deployment: Airbrake deployment (0.0 to 1.0)

    Returns:
        Drag coefficient (dimensionless)
    """
    # Cd(d) = Cd_min + (Cd_max - Cd_min) * d = 0.2 + 0.2*d
    return AIRBRAKE_CD_MIN + (AIRBRAKE_CD_MAX - AIRBRAKE_CD_MIN) * deployment


def deployment_to_drag(deployment, velocity, altitude):
    """
    Calculate drag force for a given deployment, velocity, and altitude.

    Drag equation: F_d = 0.5 * rho * v² * Cd * A

    Args:
        deployment: Airbrake deployment (0.0 to 1.0)
        velocity: Velocity in m/s
        altitude: Altitude in meters

    Returns:
        Drag force in Newtons
    """
    rho = air_density(altitude)
    Cd = deployment_to_cd(deployment)
    A = deployment_to_area(deployment)

    return 0.5 * rho * velocity**2 * Cd * A


def drag_force_to_deployment(drag_force_n, velocity_mps, altitude_m):
    """
    Convert desired drag force to airbrake deployment percentage.

    Solves the drag equation for deployment:
        F_d = 0.5 * rho * v² * Cd(d) * A(d)

    Where:
        Cd(d) = 0.2 + 0.2*d
        A(d) = 1 + 2*d

    Expanding: F_d = k * (0.4*d² + 0.6*d + 0.2) where k = 0.5 * rho * v²

    Solving quadratic: d = (-0.6 + sqrt(0.04 + 1.6*F_d/k)) / 0.8

    Args:
        drag_force_n: Desired drag force in Newtons
        velocity_mps: Current velocity in m/s
        altitude_m: Current altitude in meters

    Returns:
        Required deployment percentage (0.0 to 1.0), clamped to valid range
    """
    if velocity_mps <= 0:
        return AIRBRAKE_MIN

    rho = air_density(altitude_m)
    k = 0.5 * rho * velocity_mps**2

    if k <= 0:
        return AIRBRAKE_MIN

    # Quadratic coefficients for: 0.4*d² + 0.6*d + (0.2 - F_d/k) = 0
    # Using: d = (-0.6 + sqrt(0.04 + 1.6*F_d/k)) / 0.8
    discriminant = 0.04 + 1.6 * drag_force_n / k

    if discriminant < 0:
        # No real solution - drag force too low, return minimum deployment
        return AIRBRAKE_MIN

    deployment = (-0.6 + math.sqrt(discriminant)) / 0.8

    # Clamp to valid range
    return max(AIRBRAKE_MIN, min(AIRBRAKE_MAX, deployment))


# -----------------------------------------------------------------------------
# Sensor Buffer
# -----------------------------------------------------------------------------

class SensorBuffer:
    """Keeps last N sensor readings for calculating derivatives."""

    def __init__(self, size=3):
        self.altitudes = deque(maxlen=size)
        self.gyro_readings = deque(maxlen=size)
        self.timestamps = deque(maxlen=size)

    def add(self, altitude, gyro, timestamp):
        self.altitudes.append(altitude)
        self.gyro_readings.append(gyro)
        self.timestamps.append(timestamp)

    def is_ready(self):
        """Check if we have enough data points (3 for acceleration calc)."""
        return len(self.altitudes) >= 3

    def get_velocity(self):
        """Calculate velocity from altitude differences."""
        if len(self.altitudes) < 2:
            return 0.0
        # Use most recent two points
        dh = self.altitudes[-1] - self.altitudes[-2]
        dt = self.timestamps[-1] - self.timestamps[-2]
        return dh / dt if dt > 0 else 0.0

    def get_acceleration(self):
        """Calculate acceleration from velocity differences (second derivative of altitude)."""
        if len(self.altitudes) < 3:
            return 0.0

        dt1 = self.timestamps[-2] - self.timestamps[-3]
        dt2 = self.timestamps[-1] - self.timestamps[-2]

        if dt1 <= 0 or dt2 <= 0:
            return 0.0

        # v1 between points 0 and 1
        v1 = (self.altitudes[-2] - self.altitudes[-3]) / dt1
        # v2 between points 1 and 2
        v2 = (self.altitudes[-1] - self.altitudes[-2]) / dt2

        # Average dt for acceleration calculation
        dt_avg = (dt1 + dt2) / 2.0
        return (v2 - v1) / dt_avg


# -----------------------------------------------------------------------------
# Main Controller
# -----------------------------------------------------------------------------

class AirbrakeController:
    """
    Main controller for airbrake system.

    Phases:
    1. Pre-launch: Wait for launch detection (calculated accel > threshold)
    2. Launch: Integrate gyroscope data for tilt, wait for coast
    3. Coast (accel < 0): Active control with PID and simulation
    """

    def __init__(self, target_apogee=TARGET_APOGEE, ground_pressure=GROUND_PRESSURE_PA,
                 ground_temp=GROUND_TEMP_K):
        self.target_apogee = target_apogee
        self.ground_pressure = ground_pressure
        self.ground_temp = ground_temp
        self.sensor_buffer = SensorBuffer(size=3)

        # PID controller: output limits correspond to airbrake deployment range
        self.pid = PID(KP, KI, KD, output_limits=(AIRBRAKE_MIN, AIRBRAKE_MAX))

        # State variables
        self.current_airbrake = 0.0
        self.integrated_tilt = 0.0  # Accumulated tilt from gyroscope (degrees)
        self.phase = "pre_launch"
        self.launched = False
        self.coast_started = False
        self.connection_lost = False
        self.last_data_time = 0.0

    def read_csv_data(self, csv_path):
        """
        Generator that yields sensor data from CSV file.

        Expected CSV columns (raw sensor data):
            time        - Timestamp in seconds
            pressure    - BMP390 pressure reading in Pascals
            gyro_x      - ICM-42688-P X-axis angular rate (raw or deg/s)
            gyro_y      - ICM-42688-P Y-axis angular rate (raw or deg/s)
            gyro_z      - ICM-42688-P Z-axis angular rate (raw or deg/s)
        """
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield {
                    'time': float(row.get('time', 0)),
                    'pressure': float(row.get('pressure', GROUND_PRESSURE_PA)),
                    'gyro_x': float(row.get('gyro_x', 0)),
                    'gyro_y': float(row.get('gyro_y', 0)),
                    'gyro_z': float(row.get('gyro_z', 0))
                }

    def process_sensors(self, sensor_data):
        """
        Convert raw sensor data to usable values.

        Returns:
            dict with 'altitude' and gyro rates in deg/s
        """
        # Convert pressure to altitude
        altitude = pressure_to_altitude(
            sensor_data['pressure'],
            P0=self.ground_pressure,
            T0=self.ground_temp
        )

        # Gyro data - assuming already in deg/s from CSV
        # If raw, uncomment the gyro_to_angular_rate conversion
        gyro_x = sensor_data['gyro_x']  # gyro_to_angular_rate(sensor_data['gyro_x'])
        gyro_y = sensor_data['gyro_y']  # gyro_to_angular_rate(sensor_data['gyro_y'])
        gyro_z = sensor_data['gyro_z']  # gyro_to_angular_rate(sensor_data['gyro_z'])

        return {
            'altitude': altitude,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z
        }

    def integrate_gyroscope(self, gyro_x, gyro_y, dt):
        """
        Integrate gyroscope readings to find tilt off-axis.
        Combines X and Y rotation into total tilt angle.

        Args:
            gyro_x: Angular rate around X-axis (deg/s)
            gyro_y: Angular rate around Y-axis (deg/s)
            dt: Time step in seconds

        Returns:
            Integrated tilt angle in degrees
        """
        # Integrate angular rates to get angle change
        # Total tilt rate is the magnitude of X and Y components
        tilt_rate = math.sqrt(gyro_x**2 + gyro_y**2)
        delta_angle = tilt_rate * dt
        self.integrated_tilt += delta_angle
        return self.integrated_tilt

    def check_failsafes(self, current_velocity):
        """
        Check failsafe conditions.

        Returns: (should_retract, message)
        """
        # Failsafe 1: Excessive tilt
        if self.integrated_tilt > MAX_TILT_DEG:
            return True, f"FAILSAFE: Tilt {self.integrated_tilt:.1f}° exceeds {MAX_TILT_DEG}°. Fully retracting."

        # Failsafe 2: Connection lost (no new data but velocity > 0)
        if self.connection_lost and current_velocity > 0:
            return True, "FAILSAFE: Connection lost with positive velocity. Fully retracting."

        return False, None

    def airbrake_adjustment_loop(self, height, velocity, tilt):
        """
        Initial airbrake adjustment loop.
        Starts with max deployment and retracts until predicted apogee < target.
        Finds the deployment level where target can be reached with constant deployment.

        Returns the optimal airbrake deployment level.
        """
        airbrake = AIRBRAKE_MAX

        while airbrake > AIRBRAKE_MIN:
            # Predict apogee with candidate deployment
            airbrake_candidate = airbrake - AIRBRAKE_RETRACT_STEP
            airbrake_candidate = max(airbrake_candidate, AIRBRAKE_MIN)

            predicted_apogee = rocket_sim(height, velocity, tilt, airbrake_candidate)

            if predicted_apogee >= self.target_apogee:
                # Safe to retract further
                airbrake = airbrake_candidate
            else:
                # Stop retraction; this is the optimal point
                break

        return airbrake

    def calculate_drag_adjustment(self, pid_output, current_velocity, current_altitude, current_deployment):
        """
        Convert PID output to airbrake deployment using drag force model.

        The PID outputs a value based on apogee error:
        - Positive output = predicted apogee < target, need less drag (retract)
        - Negative output = predicted apogee > target, need more drag (deploy)

        This method:
        1. Calculates current drag force from current deployment
        2. Adjusts drag force based on PID output (scaled appropriately)
        3. Converts target drag force back to deployment percentage

        Args:
            pid_output: Output from PID controller
            current_velocity: Current velocity in m/s
            current_altitude: Current altitude in m
            current_deployment: Current airbrake deployment (0.0 to 1.0)

        Returns:
            New deployment percentage (0.0 to 1.0)
        """
        # Calculate current drag force
        current_drag = deployment_to_drag(current_deployment, current_velocity, current_altitude)

        # Scale PID output to drag force adjustment
        # PID output is based on apogee error (meters)
        # We need to convert this to a drag force change
        # Positive PID = apogee too low = reduce drag
        # Negative PID = apogee too high = increase drag
        drag_scale_factor = 10.0  # Tunable: converts apogee error to drag force (N per meter error)
        target_drag = current_drag - (pid_output * drag_scale_factor)

        # Ensure target drag is non-negative
        target_drag = max(0.0, target_drag)

        # Convert target drag force to deployment percentage
        new_deployment = drag_force_to_deployment(target_drag, current_velocity, current_altitude)

        return new_deployment

    def run(self, csv_path):
        """
        Main control loop. Reads sensor data from CSV and controls airbrakes.
        """
        print("Airbrake Controller Starting...")
        print(f"Target Apogee: {self.target_apogee} m")
        print(f"Ground Pressure: {self.ground_pressure} Pa")
        print(f"Ground Temperature: {self.ground_temp} K")

        previous_time = None

        for raw_sensor_data in self.read_csv_data(csv_path):
            current_time = raw_sensor_data['time']

            # Calculate dt
            if previous_time is not None:
                dt = current_time - previous_time
            else:
                dt = DT
            previous_time = current_time

            # Update last data time for connection monitoring
            self.last_data_time = current_time

            # Process raw sensor data
            sensor_data = self.process_sensors(raw_sensor_data)
            altitude = sensor_data['altitude']

            # Add to sensor buffer
            self.sensor_buffer.add(
                altitude,
                (sensor_data['gyro_x'], sensor_data['gyro_y'], sensor_data['gyro_z']),
                current_time
            )

            # Integrate gyroscope data (during all phases after launch)
            if self.launched:
                self.integrate_gyroscope(
                    sensor_data['gyro_x'],
                    sensor_data['gyro_y'],
                    dt
                )

            # Get calculated acceleration (from altitude derivatives)
            calculated_accel = self.sensor_buffer.get_acceleration()

            # Phase: Pre-launch - wait for launch detection
            if self.phase == "pre_launch":
                if self.sensor_buffer.is_ready() and calculated_accel > LAUNCH_ACCEL_THRESHOLD:
                    self.phase = "launch"
                    self.launched = True
                    print(f"[{current_time:.2f}s] Launch detected! (accel={calculated_accel:.1f} m/s²)")
                    print(f"[{current_time:.2f}s] Beginning gyroscope integration.")
                continue

            # Phase: Launch - integrate gyroscope, wait for coast phase
            if self.phase == "launch":
                if calculated_accel < 0:  # Negative acceleration = coast
                    self.phase = "coast_init"
                    print(f"[{current_time:.2f}s] Coast phase detected (accel={calculated_accel:.1f} m/s²). Initializing control.")
                continue

            # Phase: Coast initialization - gather 3 data points after coast start
            if self.phase == "coast_init":
                if self.sensor_buffer.is_ready():
                    self.phase = "coast_active"
                    print(f"[{current_time:.2f}s] Sensor buffer ready. Starting active control.")

                    # Get current state
                    height = altitude
                    velocity = self.sensor_buffer.get_velocity()
                    tilt = self.integrated_tilt

                    print(f"[{current_time:.2f}s] State: h={height:.1f}m, v={velocity:.1f}m/s, tilt={tilt:.1f}°")

                    # Run initial simulation with no brakes
                    predicted_apogee = rocket_sim(height, velocity, tilt, 0.0)
                    print(f"[{current_time:.2f}s] Predicted apogee (no brakes): {predicted_apogee:.2f} m")

                    if predicted_apogee <= self.target_apogee:
                        print(f"[{current_time:.2f}s] Predicted apogee too low. Not deploying brakes.")
                        self.current_airbrake = AIRBRAKE_MIN
                    else:
                        print(f"[{current_time:.2f}s] Predicted apogee > target. Running adjustment loop.")
                        # Max out airbrakes first, then run adjustment loop
                        self.current_airbrake = AIRBRAKE_MAX
                        self.current_airbrake = self.airbrake_adjustment_loop(
                            height, velocity, tilt
                        )
                        print(f"[{current_time:.2f}s] Initial airbrake deployment: {self.current_airbrake:.1%}")
                        self.command_airbrakes(self.current_airbrake)
                continue

            # Phase: Active coast control (Main Loop)
            if self.phase == "coast_active":
                # Get current state from sensor buffer
                height = altitude
                velocity = self.sensor_buffer.get_velocity()
                acceleration = self.sensor_buffer.get_acceleration()
                tilt = self.integrated_tilt

                # Check for apogee (velocity <= 0)
                if velocity <= 0:
                    print(f"[{current_time:.2f}s] Apogee reached at {height:.2f} m. Retracting airbrakes.")
                    self.current_airbrake = AIRBRAKE_MIN
                    self.command_airbrakes(self.current_airbrake)
                    break

                # Check failsafes
                should_retract, message = self.check_failsafes(velocity)
                if should_retract:
                    print(f"[{current_time:.2f}s] {message}")
                    self.current_airbrake = AIRBRAKE_MIN
                    self.command_airbrakes(self.current_airbrake)
                    continue

                # Predict apogee with current deployment
                predicted_apogee = rocket_sim(height, velocity, tilt, self.current_airbrake)

                # PID control: setpoint is target, measurement is predicted apogee
                pid_output = self.pid.update(self.target_apogee, predicted_apogee, dt)

                # Calculate new airbrake deployment based on PID output
                self.current_airbrake = self.calculate_drag_adjustment(
                    pid_output, velocity, height, self.current_airbrake
                )

                # Command airbrakes
                self.command_airbrakes(self.current_airbrake)

                # Log status (including drag force)
                current_drag = deployment_to_drag(self.current_airbrake, velocity, height)
                print(f"[{current_time:.2f}s] h={height:.1f}m v={velocity:.1f}m/s a={acceleration:.1f}m/s² "
                      f"pred={predicted_apogee:.1f}m brake={self.current_airbrake:.1%} drag={current_drag:.1f}N tilt={tilt:.1f}°")

        print("Controller finished.")
        return self.current_airbrake

    def command_airbrakes(self, deployment):
        """
        Send deployment command to airbrake actuators.
        Override this method for actual hardware interface.

        Args:
            deployment: Deployment percentage (0.0 to 1.0)
        """
        # Clamp value
        deployment = max(AIRBRAKE_MIN, min(AIRBRAKE_MAX, deployment))
        self.current_airbrake = deployment
        # In actual implementation: send command to actuators


# -----------------------------------------------------------------------------
# Entry Points
# -----------------------------------------------------------------------------

def run_from_csv(csv_path, target_apogee=TARGET_APOGEE,
                 ground_pressure=GROUND_PRESSURE_PA, ground_temp=GROUND_TEMP_K):
    """
    Convenience function to run controller from CSV file.

    Args:
        csv_path: Path to sensor data CSV
        target_apogee: Target apogee in meters
        ground_pressure: Calibrated ground pressure in Pascals
        ground_temp: Ground temperature in Kelvin
    """
    controller = AirbrakeController(
        target_apogee=target_apogee,
        ground_pressure=ground_pressure,
        ground_temp=ground_temp
    )
    return controller.run(csv_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python controller.py <sensor_data.csv> [target_apogee] [ground_pressure] [ground_temp]")
        print()
        print("Expected CSV columns (raw sensor data):")
        print("  time     - Timestamp in seconds")
        print("  pressure - BMP390 pressure in Pascals")
        print("  gyro_x   - ICM-42688-P X-axis angular rate (deg/s)")
        print("  gyro_y   - ICM-42688-P Y-axis angular rate (deg/s)")
        print("  gyro_z   - ICM-42688-P Z-axis angular rate (deg/s)")
        sys.exit(1)

    csv_file = sys.argv[1]
    target = float(sys.argv[2]) if len(sys.argv) > 2 else TARGET_APOGEE
    pressure = float(sys.argv[3]) if len(sys.argv) > 3 else GROUND_PRESSURE_PA
    temp = float(sys.argv[4]) if len(sys.argv) > 4 else GROUND_TEMP_K

    run_from_csv(csv_file, target, pressure, temp)
