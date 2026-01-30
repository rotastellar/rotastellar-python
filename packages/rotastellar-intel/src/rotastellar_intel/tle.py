"""
RotaStellar Intel - TLE Parsing and Propagation

Two-Line Element set parsing and orbit propagation using SGP4/SDP4.

subhadipmitra@: TLEs are the de facto standard for satellite orbit data, published
by Space-Track (18th Space Defense Squadron). They're updated ~daily for most objects.

Important caveats:
- TLEs degrade over time (accurate to ~1km after a few days for LEO)
- They're mean elements, not osculating - direct conversion to Keplerian is wrong
- Must use SGP4/SDP4 for propagation, not simple two-body mechanics

The SGP4 algorithm handles:
- J2 (Earth oblateness)
- Atmospheric drag (via B* term)
- Solar/lunar perturbations (for deep space - SDP4)

For precision applications (rendezvous, formation flying), use ephemeris data instead.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List
import math
import re

from rotastellar import Position, Orbit, ValidationError

# TODO(subhadipmitra): Add support for 3LE format (includes satellite name)
# TODO: Implement SDP4 for deep space objects (period > 225 min)
# NOTE: Using AFSPC compatibility mode for SGP4 constants

# Earth constants
EARTH_RADIUS_KM = 6378.137
EARTH_MU = 398600.4418  # km^3/s^2
MINUTES_PER_DAY = 1440.0
SECONDS_PER_DAY = 86400.0


@dataclass
class TLE:
    """Two-Line Element set for satellite orbit determination.

    A TLE contains orbital elements that describe a satellite's orbit at a
    specific epoch time. These can be propagated forward or backward in time
    using SGP4/SDP4 algorithms.

    Attributes:
        name: Satellite name (line 0)
        norad_id: NORAD catalog number
        classification: Classification (U=unclassified, C=classified, S=secret)
        intl_designator: International designator (launch year, number, piece)
        epoch_year: Epoch year (2-digit)
        epoch_day: Epoch day of year (fractional)
        mean_motion_dot: First derivative of mean motion (rev/day^2)
        mean_motion_ddot: Second derivative of mean motion (rev/day^3)
        bstar: BSTAR drag term
        element_set_type: Element set type
        element_number: Element set number
        inclination: Inclination in degrees
        raan: Right ascension of ascending node in degrees
        eccentricity: Eccentricity (decimal point assumed)
        arg_perigee: Argument of perigee in degrees
        mean_anomaly: Mean anomaly in degrees
        mean_motion: Mean motion in revolutions per day
        rev_number: Revolution number at epoch

    Example:
        >>> tle_lines = [
        ...     "ISS (ZARYA)",
        ...     "1 25544U 98067A   21275.52243902  .00001082  00000-0  27450-4 0  9999",
        ...     "2 25544  51.6443 208.5943 0003631 355.3422 144.3824 15.48919755304818"
        ... ]
        >>> tle = TLE.parse(tle_lines)
        >>> print(f"ISS inclination: {tle.inclination}°")
    """

    name: str
    norad_id: int
    classification: str
    intl_designator: str
    epoch_year: int
    epoch_day: float
    mean_motion_dot: float
    mean_motion_ddot: float
    bstar: float
    element_set_type: int
    element_number: int
    inclination: float
    raan: float
    eccentricity: float
    arg_perigee: float
    mean_anomaly: float
    mean_motion: float
    rev_number: int

    @classmethod
    def parse(cls, lines: List[str]) -> "TLE":
        """Parse a TLE from its text representation.

        Args:
            lines: List of 2 or 3 strings (name optional, then line 1, line 2)

        Returns:
            Parsed TLE object

        Raises:
            ValidationError: If TLE format is invalid
        """
        if len(lines) == 2:
            name = "UNKNOWN"
            line1, line2 = lines
        elif len(lines) == 3:
            name = lines[0].strip()
            line1, line2 = lines[1], lines[2]
        else:
            raise ValidationError("lines", "TLE must have 2 or 3 lines")

        # Validate line numbers
        if not line1.startswith("1 "):
            raise ValidationError("line1", "Line 1 must start with '1 '")
        if not line2.startswith("2 "):
            raise ValidationError("line2", "Line 2 must start with '2 '")

        try:
            # Parse line 1
            norad_id = int(line1[2:7])
            classification = line1[7]
            intl_designator = line1[9:17].strip()
            epoch_year = int(line1[18:20])
            epoch_day = float(line1[20:32])
            mean_motion_dot = float(line1[33:43])

            # Parse mean_motion_ddot (scientific notation without 'E')
            mmddot_str = line1[44:52].strip()
            if mmddot_str:
                mantissa = float(f"0.{mmddot_str[:-2].replace(' ', '0').replace('-', '').replace('+', '')}")
                exponent = int(mmddot_str[-2:])
                sign = -1 if '-' in mmddot_str[:-2] else 1
                mean_motion_ddot = sign * mantissa * (10 ** exponent)
            else:
                mean_motion_ddot = 0.0

            # Parse BSTAR (scientific notation without 'E')
            bstar_str = line1[53:61].strip()
            if bstar_str:
                mantissa = float(f"0.{bstar_str[:-2].replace(' ', '0').replace('-', '').replace('+', '')}")
                exponent = int(bstar_str[-2:])
                sign = -1 if '-' in bstar_str[:-2] else 1
                bstar = sign * mantissa * (10 ** exponent)
            else:
                bstar = 0.0

            element_set_type = int(line1[62]) if line1[62].strip() else 0
            element_number = int(line1[64:68])

            # Parse line 2
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float(f"0.{line2[26:33]}")
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            rev_number = int(line2[63:68])

            return cls(
                name=name,
                norad_id=norad_id,
                classification=classification,
                intl_designator=intl_designator,
                epoch_year=epoch_year,
                epoch_day=epoch_day,
                mean_motion_dot=mean_motion_dot,
                mean_motion_ddot=mean_motion_ddot,
                bstar=bstar,
                element_set_type=element_set_type,
                element_number=element_number,
                inclination=inclination,
                raan=raan,
                eccentricity=eccentricity,
                arg_perigee=arg_perigee,
                mean_anomaly=mean_anomaly,
                mean_motion=mean_motion,
                rev_number=rev_number,
            )

        except (ValueError, IndexError) as e:
            raise ValidationError("tle", f"Failed to parse TLE: {e}")

    @property
    def epoch(self) -> datetime:
        """Get the epoch as a datetime object."""
        # Convert 2-digit year to 4-digit
        if self.epoch_year < 57:
            year = 2000 + self.epoch_year
        else:
            year = 1900 + self.epoch_year

        # Convert day of year to datetime
        from datetime import timedelta
        jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
        return jan1 + timedelta(days=self.epoch_day - 1)

    @property
    def semi_major_axis_km(self) -> float:
        """Calculate semi-major axis from mean motion."""
        # n = sqrt(mu / a^3), so a = (mu / n^2)^(1/3)
        n_rad_per_sec = self.mean_motion * 2 * math.pi / SECONDS_PER_DAY
        return (EARTH_MU / (n_rad_per_sec ** 2)) ** (1/3)

    @property
    def orbital_period_minutes(self) -> float:
        """Calculate orbital period in minutes."""
        return MINUTES_PER_DAY / self.mean_motion

    @property
    def apogee_km(self) -> float:
        """Calculate apogee altitude in km."""
        a = self.semi_major_axis_km
        return a * (1 + self.eccentricity) - EARTH_RADIUS_KM

    @property
    def perigee_km(self) -> float:
        """Calculate perigee altitude in km."""
        a = self.semi_major_axis_km
        return a * (1 - self.eccentricity) - EARTH_RADIUS_KM

    def to_orbit(self) -> Orbit:
        """Convert TLE to Orbit object.

        Note: This uses osculating elements at epoch. For accurate
        propagation, use SGP4/SDP4.
        """
        return Orbit(
            semi_major_axis_km=self.semi_major_axis_km,
            eccentricity=self.eccentricity,
            inclination_deg=self.inclination,
            raan_deg=self.raan,
            arg_periapsis_deg=self.arg_perigee,
            true_anomaly_deg=self.mean_anomaly,  # Approximation
        )

    def propagate(self, dt: datetime) -> Position:
        """Propagate the orbit to a given time.

        This is a simplified propagation. For accurate results,
        install the `sgp4` package and use `propagate_sgp4()`.

        Args:
            dt: Target datetime (UTC)

        Returns:
            Estimated position at the given time
        """
        # Simplified propagation - just use mean motion
        # For accurate results, use SGP4
        minutes_since_epoch = (dt - self.epoch).total_seconds() / 60
        revolutions = minutes_since_epoch / self.orbital_period_minutes

        # Simple circular orbit approximation
        mean_anomaly_rad = math.radians(self.mean_anomaly)
        new_anomaly = mean_anomaly_rad + (revolutions * 2 * math.pi)

        # Convert to lat/lon (very simplified)
        lat = math.degrees(math.asin(
            math.sin(math.radians(self.inclination)) * math.sin(new_anomaly)
        ))
        lon = math.degrees(new_anomaly) - 180
        while lon < -180:
            lon += 360
        while lon > 180:
            lon -= 360

        alt = (self.apogee_km + self.perigee_km) / 2

        return Position(latitude=lat, longitude=lon, altitude_km=alt)

    def propagate_sgp4(self, dt: datetime) -> Position:
        """Propagate using SGP4/SDP4 algorithm.

        Requires the `sgp4` package to be installed.

        Args:
            dt: Target datetime (UTC)

        Returns:
            Accurate position at the given time

        Raises:
            ImportError: If sgp4 package is not installed
        """
        try:
            from sgp4.api import Satrec, jday
        except ImportError:
            raise ImportError(
                "SGP4 propagation requires the 'sgp4' package. "
                "Install with: pip install rotastellar-intel[sgp4]"
            )

        # Create satellite record
        line1 = self._to_line1()
        line2 = self._to_line2()
        satellite = Satrec.twoline2rv(line1, line2)

        # Get Julian date
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        # Propagate
        error_code, position, velocity = satellite.sgp4(jd, fr)

        if error_code != 0:
            raise ValidationError(
                "propagation",
                f"SGP4 propagation error code: {error_code}"
            )

        # Convert ECI to geodetic
        # position is in km from Earth center
        x, y, z = position

        # Simple ECI to geodetic conversion
        r = math.sqrt(x**2 + y**2 + z**2)
        lat = math.degrees(math.asin(z / r))

        # Longitude needs Earth rotation angle (simplified)
        gmst = self._greenwich_sidereal_time(jd + fr)
        lon = math.degrees(math.atan2(y, x)) - gmst
        while lon < -180:
            lon += 360
        while lon > 180:
            lon -= 360

        alt = r - EARTH_RADIUS_KM

        return Position(latitude=lat, longitude=lon, altitude_km=alt)

    def _greenwich_sidereal_time(self, jd: float) -> float:
        """Calculate Greenwich Mean Sidereal Time in degrees."""
        # Simplified formula
        T = (jd - 2451545.0) / 36525.0
        gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0)
        gmst = gmst % 360
        return gmst

    def _to_line1(self) -> str:
        """Reconstruct TLE line 1."""
        # Simplified reconstruction for SGP4 compatibility
        return f"1 {self.norad_id:05d}U {self.intl_designator:8s} {self.epoch_year:02d}{self.epoch_day:012.8f}  .00000000  00000-0  00000-0 0  0000"

    def _to_line2(self) -> str:
        """Reconstruct TLE line 2."""
        ecc_str = f"{self.eccentricity:.7f}"[2:]  # Remove "0."
        return f"2 {self.norad_id:05d} {self.inclination:8.4f} {self.raan:8.4f} {ecc_str} {self.arg_perigee:8.4f} {self.mean_anomaly:8.4f} {self.mean_motion:11.8f}00000"


def parse_tle(text: str) -> List[TLE]:
    """Parse multiple TLEs from text.

    Args:
        text: Text containing one or more TLEs

    Returns:
        List of parsed TLE objects
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    tles = []

    i = 0
    while i < len(lines):
        # Check if this is a name line or line 1
        if lines[i].startswith("1 "):
            # No name, just lines 1 and 2
            if i + 1 < len(lines) and lines[i + 1].startswith("2 "):
                tles.append(TLE.parse([lines[i], lines[i + 1]]))
                i += 2
            else:
                i += 1
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 "):
            # Name + line 1 + line 2
            tles.append(TLE.parse([lines[i], lines[i + 1], lines[i + 2]]))
            i += 3
        else:
            i += 1

    return tles
