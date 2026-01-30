"""
RotaStellar Distributed - Sync Scheduler

Schedule data synchronization across ground station passes.

subhadipmitra@: This module handles the scheduling of data transfers during
ground station contact windows. LEO satellites typically get:
- 10-15 minute contact windows per pass
- 4-6 passes per day over a single ground station
- ~100-200 Mbps downlink (varies with elevation angle)

The priority queue ensures critical gradient updates get transmitted first,
while lower-priority telemetry can wait for the next pass if needed.

For production, integrate with:
- STK/GMAT for precise pass predictions
- AWS Ground Station / Azure Orbital / KSAT for antenna booking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List
import heapq
import math

# TODO(subhadipmitra): Add actual pass prediction using skyfield or pyorbital
# TODO: Implement bandwidth estimation based on elevation angle
# NOTE: Using simplified orbital period formula (assumes circular orbit)


class Priority(Enum):
    """Priority level for sync operations."""
    CRITICAL = 0  # Must sync in next pass
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class GroundStation:
    """Configuration for a ground station.

    Attributes:
        name: Station name/identifier
        latitude: Latitude in degrees
        longitude: Longitude in degrees
        elevation_m: Elevation above sea level in meters
        bandwidth_mbps: Available bandwidth in Mbps
        min_elevation_deg: Minimum elevation angle for contact
    """
    name: str
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    bandwidth_mbps: float = 100.0
    min_elevation_deg: float = 5.0

    @classmethod
    def svalbard(cls) -> "GroundStation":
        """Svalbard Satellite Station (high Arctic coverage)."""
        return cls("Svalbard", 78.2306, 15.3894, 450.0, 200.0)

    @classmethod
    def kourou(cls) -> "GroundStation":
        """Kourou, French Guiana (equatorial)."""
        return cls("Kourou", 5.2378, -52.7683, 0.0, 150.0)

    @classmethod
    def perth(cls) -> "GroundStation":
        """Perth, Australia."""
        return cls("Perth", -31.9474, 115.8648, 30.0, 100.0)

    @classmethod
    def fairbanks(cls) -> "GroundStation":
        """Fairbanks, Alaska (polar coverage)."""
        return cls("Fairbanks", 64.8401, -147.7200, 135.0, 150.0)

    @classmethod
    def default_network(cls) -> List["GroundStation"]:
        """Default global ground station network."""
        return [cls.svalbard(), cls.kourou(), cls.perth(), cls.fairbanks()]


@dataclass
class ContactWindow:
    """A scheduled contact window with a ground station.

    Attributes:
        station: Ground station for this contact
        start_time: Start of contact window
        end_time: End of contact window
        max_elevation_deg: Maximum elevation during pass
        available_bandwidth_mbps: Bandwidth available during this window
    """
    station: GroundStation
    start_time: datetime
    end_time: datetime
    max_elevation_deg: float
    available_bandwidth_mbps: float

    @property
    def duration_seconds(self) -> float:
        """Duration of the contact window in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def duration_minutes(self) -> float:
        """Duration of the contact window in minutes."""
        return self.duration_seconds / 60.0

    @property
    def capacity_mb(self) -> float:
        """Total data capacity in MB for this window."""
        return (self.available_bandwidth_mbps * self.duration_seconds) / 8


@dataclass(order=True)
class SyncTask:
    """A task queued for synchronization.

    Uses priority ordering for the priority queue.
    """
    priority: int  # Lower = higher priority
    deadline: datetime
    node_id: str = field(compare=False)
    data_size_bytes: int = field(compare=False)
    task_id: str = field(compare=False)
    description: str = field(compare=False, default="")
    created_at: datetime = field(compare=False, default_factory=datetime.now)


class PriorityQueue:
    """Priority queue for bandwidth-aware sync operations.

    Manages sync tasks with priority, deadline, and size considerations.

    Example:
        >>> queue = PriorityQueue()
        >>> queue.add_task("node-1", 1024*1024, Priority.HIGH, "Upload gradients")
        >>> queue.add_task("node-2", 512*1024, Priority.NORMAL, "Sync checkpoints")
        >>> while not queue.is_empty():
        ...     task = queue.pop_task()
        ...     print(f"Processing: {task.description}")
    """

    def __init__(self):
        self._heap: List[SyncTask] = []
        self._task_counter = 0

    def add_task(
        self,
        node_id: str,
        data_size_bytes: int,
        priority: Priority = Priority.NORMAL,
        description: str = "",
        deadline: Optional[datetime] = None
    ) -> str:
        """Add a sync task to the queue."""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        if deadline is None:
            # Default deadline based on priority
            hours = {Priority.CRITICAL: 1, Priority.HIGH: 4, Priority.NORMAL: 12, Priority.LOW: 48}
            deadline = datetime.now() + timedelta(hours=hours[priority])

        task = SyncTask(
            priority=priority.value,
            deadline=deadline,
            node_id=node_id,
            data_size_bytes=data_size_bytes,
            task_id=task_id,
            description=description
        )

        heapq.heappush(self._heap, task)
        return task_id

    def pop_task(self) -> Optional[SyncTask]:
        """Get and remove the highest priority task."""
        if self._heap:
            return heapq.heappop(self._heap)
        return None

    def peek_task(self) -> Optional[SyncTask]:
        """Look at the highest priority task without removing it."""
        if self._heap:
            return self._heap[0]
        return None

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._heap) == 0

    @property
    def size(self) -> int:
        """Number of tasks in queue."""
        return len(self._heap)

    @property
    def total_bytes_pending(self) -> int:
        """Total bytes across all pending tasks."""
        return sum(t.data_size_bytes for t in self._heap)

    def get_tasks_for_window(self, capacity_bytes: int) -> List[SyncTask]:
        """Get tasks that fit within a bandwidth window."""
        tasks = []
        remaining = capacity_bytes

        # Sort by priority and deadline
        candidates = sorted(self._heap, key=lambda t: (t.priority, t.deadline))

        for task in candidates:
            if task.data_size_bytes <= remaining:
                tasks.append(task)
                remaining -= task.data_size_bytes

        # Remove selected tasks from heap
        for task in tasks:
            self._heap.remove(task)
        heapq.heapify(self._heap)

        return tasks


class SyncScheduler:
    """Schedule data synchronization across ground station passes.

    Optimizes sync operations to maximize bandwidth utilization while
    respecting priorities and deadlines.

    Example:
        >>> stations = GroundStation.default_network()
        >>> scheduler = SyncScheduler(stations, orbit_altitude_km=550.0)
        >>> windows = scheduler.get_contact_windows(hours=24)
        >>> scheduler.schedule_sync("orbital-1", 10_000_000, Priority.HIGH)
        >>> plan = scheduler.optimize()
    """

    EARTH_RADIUS_KM = 6371.0
    EARTH_MU = 398600.4418

    def __init__(
        self,
        ground_stations: Optional[List[GroundStation]] = None,
        orbit_altitude_km: float = 550.0,
        orbit_inclination_deg: float = 51.6
    ):
        self.ground_stations = ground_stations or GroundStation.default_network()
        self.orbit_altitude_km = orbit_altitude_km
        self.orbit_inclination_deg = orbit_inclination_deg
        self.queue = PriorityQueue()
        self._schedule: List[tuple[ContactWindow, List[SyncTask]]] = []

    @property
    def orbital_period_minutes(self) -> float:
        """Orbital period in minutes."""
        a = self.EARTH_RADIUS_KM + self.orbit_altitude_km
        period_s = 2 * math.pi * math.sqrt(a**3 / self.EARTH_MU)
        return period_s / 60.0

    @property
    def orbits_per_day(self) -> float:
        """Number of orbits per day."""
        return (24.0 * 60.0) / self.orbital_period_minutes

    def get_contact_windows(
        self,
        start_time: Optional[datetime] = None,
        hours: int = 24
    ) -> List[ContactWindow]:
        """Get predicted contact windows for the time period.

        This is a simplified simulation - production would use SGP4 propagation.
        """
        windows = []
        start = start_time or datetime.now()
        end = start + timedelta(hours=hours)

        # Simulate contact windows for each station
        for station in self.ground_stations:
            # Simplified: estimate based on orbital mechanics
            passes_per_day = self._estimate_passes_per_day(station.latitude)
            pass_duration_min = self._estimate_pass_duration(station.latitude)

            # Generate windows
            interval = timedelta(hours=24.0 / passes_per_day) if passes_per_day > 0 else timedelta(hours=24)
            current = start

            while current < end:
                if passes_per_day > 0:
                    window_start = current
                    window_end = current + timedelta(minutes=pass_duration_min)

                    # Estimate max elevation based on latitude
                    max_elevation = self._estimate_max_elevation(station.latitude)

                    if max_elevation > station.min_elevation_deg:
                        windows.append(ContactWindow(
                            station=station,
                            start_time=window_start,
                            end_time=window_end,
                            max_elevation_deg=max_elevation,
                            available_bandwidth_mbps=station.bandwidth_mbps
                        ))

                current += interval

        # Sort by start time
        windows.sort(key=lambda w: w.start_time)
        return windows

    def schedule_sync(
        self,
        node_id: str,
        data_size_bytes: int,
        priority: Priority = Priority.NORMAL,
        description: str = "",
        deadline: Optional[datetime] = None
    ) -> str:
        """Schedule a sync operation."""
        return self.queue.add_task(
            node_id=node_id,
            data_size_bytes=data_size_bytes,
            priority=priority,
            description=description,
            deadline=deadline
        )

    def optimize(self, hours: int = 24) -> List[tuple[ContactWindow, List[SyncTask]]]:
        """Optimize sync schedule for upcoming contact windows.

        Returns list of (window, tasks) tuples representing the schedule.
        """
        windows = self.get_contact_windows(hours=hours)
        schedule = []

        for window in windows:
            capacity_bytes = int(window.capacity_mb * 1024 * 1024)
            tasks = self.queue.get_tasks_for_window(capacity_bytes)

            if tasks:
                schedule.append((window, tasks))

        self._schedule = schedule
        return schedule

    def get_schedule_summary(self) -> Dict:
        """Get summary of current schedule."""
        total_data = sum(
            sum(t.data_size_bytes for t in tasks)
            for _, tasks in self._schedule
        )
        total_tasks = sum(len(tasks) for _, tasks in self._schedule)

        return {
            "scheduled_windows": len(self._schedule),
            "total_tasks_scheduled": total_tasks,
            "total_data_mb": round(total_data / (1024 * 1024), 2),
            "pending_tasks": self.queue.size,
            "pending_data_mb": round(self.queue.total_bytes_pending / (1024 * 1024), 2),
        }

    def _estimate_passes_per_day(self, station_lat: float) -> float:
        """Estimate number of passes per day for a station."""
        # Simplified model based on inclination coverage
        abs_lat = abs(station_lat)
        if abs_lat > self.orbit_inclination_deg + 10:
            return 0  # Station not covered

        # Higher latitude stations get more polar passes
        if abs_lat > self.orbit_inclination_deg - 10:
            return self.orbits_per_day * 0.3  # Many passes for high-lat
        else:
            return self.orbits_per_day * 0.15  # Fewer passes for mid-lat

    def _estimate_pass_duration(self, station_lat: float) -> float:
        """Estimate average pass duration in minutes."""
        # Simplified: 5-12 minutes typical for LEO
        base_duration = 8.0
        # Higher elevation = longer visible
        if abs(station_lat) > self.orbit_inclination_deg - 5:
            return base_duration * 1.2
        return base_duration

    def _estimate_max_elevation(self, station_lat: float) -> float:
        """Estimate typical maximum elevation for a station."""
        lat_diff = abs(abs(station_lat) - self.orbit_inclination_deg)
        if lat_diff < 5:
            return 70.0  # Near-overhead passes
        elif lat_diff < 15:
            return 45.0
        elif lat_diff < 25:
            return 25.0
        else:
            return 10.0
