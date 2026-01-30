"""
RotaStellar Distributed - Space Mesh

Inter-satellite link (ISL) routing for orbital node communication.

subhadipmitra@: The mesh topology here is modeled after Starlink's approach -
each satellite maintains 4 ISL connections (2 intra-plane, 2 inter-plane).
We simplify by using distance-based connectivity rather than fixed neighbor counts.

The routing uses Dijkstra which is fine for constellations up to ~1000 nodes.
For larger meshes, consider hierarchical routing or pre-computed tables.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple
import math
import heapq

# NOTE(subhadipmitra): Speed of light in vacuum. In fiber it's ~0.67c but ISLs are free-space.
# TODO: Add atmospheric drag effects on link budget at lower altitudes


class LinkType(Enum):
    """Type of communication link."""
    OPTICAL = "optical"  # Laser inter-satellite link
    RF = "rf"  # Radio frequency link
    HYBRID = "hybrid"


@dataclass
class OrbitalNode:
    """An orbital compute node in the mesh.

    Attributes:
        node_id: Unique identifier
        orbit_altitude_km: Orbital altitude
        orbit_inclination_deg: Orbital inclination
        raan_deg: Right ascension of ascending node
        mean_anomaly_deg: Mean anomaly (position in orbit)
        isl_range_km: Maximum ISL range
        isl_bandwidth_gbps: ISL bandwidth capacity
        compute_tflops: Compute capacity
    """
    node_id: str
    orbit_altitude_km: float = 550.0
    orbit_inclination_deg: float = 51.6
    raan_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    isl_range_km: float = 5000.0
    isl_bandwidth_gbps: float = 10.0
    compute_tflops: float = 10.0


@dataclass
class ISLLink:
    """Inter-satellite link between two nodes.

    Attributes:
        source_id: Source node ID
        target_id: Target node ID
        distance_km: Current distance
        bandwidth_gbps: Available bandwidth
        latency_ms: One-way latency
        link_type: Type of link (optical/RF)
        active: Whether link is currently active
    """
    source_id: str
    target_id: str
    distance_km: float
    bandwidth_gbps: float
    latency_ms: float
    link_type: LinkType = LinkType.OPTICAL
    active: bool = True


@dataclass
class Route:
    """A route through the mesh between two nodes.

    Attributes:
        source_id: Starting node
        destination_id: Ending node
        path: List of node IDs in the path
        total_distance_km: Total distance
        total_latency_ms: Total latency
        min_bandwidth_gbps: Bottleneck bandwidth
        num_hops: Number of ISL hops
    """
    source_id: str
    destination_id: str
    path: List[str]
    total_distance_km: float
    total_latency_ms: float
    min_bandwidth_gbps: float
    num_hops: int

    @property
    def is_valid(self) -> bool:
        """Check if route is valid (has path)."""
        return len(self.path) >= 2


class SpaceMesh:
    """ISL routing mesh for orbital node communication.

    Manages the mesh network topology and provides routing between
    orbital nodes using Dijkstra's algorithm for optimal paths.

    Example:
        >>> mesh = SpaceMesh()
        >>> mesh.add_node(OrbitalNode("sat-1", mean_anomaly_deg=0))
        >>> mesh.add_node(OrbitalNode("sat-2", mean_anomaly_deg=30))
        >>> mesh.add_node(OrbitalNode("sat-3", mean_anomaly_deg=60))
        >>> mesh.update_topology()
        >>> route = mesh.find_route("sat-1", "sat-3")
        >>> print(f"Route: {route.path}, Latency: {route.total_latency_ms:.1f}ms")
    """

    SPEED_OF_LIGHT_KM_S = 299792.458
    EARTH_RADIUS_KM = 6371.0

    def __init__(self, default_isl_range_km: float = 5000.0):
        self.default_isl_range_km = default_isl_range_km
        self.nodes: Dict[str, OrbitalNode] = {}
        self.links: Dict[Tuple[str, str], ISLLink] = {}
        self._adjacency: Dict[str, Set[str]] = {}

    def add_node(self, node: OrbitalNode) -> None:
        """Add an orbital node to the mesh."""
        self.nodes[node.node_id] = node
        self._adjacency[node.node_id] = set()

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the mesh."""
        if node_id in self.nodes:
            # Remove all links involving this node
            links_to_remove = [
                key for key in self.links
                if node_id in key
            ]
            for key in links_to_remove:
                del self.links[key]

            # Update adjacency
            for neighbors in self._adjacency.values():
                neighbors.discard(node_id)
            del self._adjacency[node_id]
            del self.nodes[node_id]

    def update_topology(self) -> None:
        """Update the mesh topology based on current node positions.

        Recalculates which nodes can communicate via ISL based on
        distance and line-of-sight constraints.
        """
        # Clear existing links
        self.links.clear()
        for node_id in self._adjacency:
            self._adjacency[node_id].clear()

        # Calculate new links
        node_ids = list(self.nodes.keys())
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i + 1:]:
                node1 = self.nodes[id1]
                node2 = self.nodes[id2]

                distance = self._calculate_distance(node1, node2)
                max_range = min(node1.isl_range_km, node2.isl_range_km)

                if distance <= max_range and self._has_line_of_sight(node1, node2):
                    # Create bidirectional link
                    bandwidth = min(node1.isl_bandwidth_gbps, node2.isl_bandwidth_gbps)
                    latency = (distance / self.SPEED_OF_LIGHT_KM_S) * 1000

                    link = ISLLink(
                        source_id=id1,
                        target_id=id2,
                        distance_km=distance,
                        bandwidth_gbps=bandwidth,
                        latency_ms=latency
                    )

                    self.links[(id1, id2)] = link
                    self.links[(id2, id1)] = ISLLink(
                        source_id=id2,
                        target_id=id1,
                        distance_km=distance,
                        bandwidth_gbps=bandwidth,
                        latency_ms=latency
                    )

                    self._adjacency[id1].add(id2)
                    self._adjacency[id2].add(id1)

    def find_route(
        self,
        source_id: str,
        destination_id: str,
        optimize_for: str = "latency"
    ) -> Route:
        """Find optimal route between two nodes using Dijkstra's algorithm.

        Args:
            source_id: Starting node
            destination_id: Destination node
            optimize_for: "latency" or "bandwidth"

        Returns:
            Route object with path and metrics
        """
        if source_id not in self.nodes or destination_id not in self.nodes:
            return Route(source_id, destination_id, [], 0, 0, 0, 0)

        if source_id == destination_id:
            return Route(source_id, destination_id, [source_id], 0, 0, float('inf'), 0)

        # Dijkstra's algorithm
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[source_id] = 0
        predecessors: Dict[str, Optional[str]] = {node_id: None for node_id in self.nodes}

        # Priority queue: (distance, node_id)
        pq = [(0, source_id)]
        visited = set()

        while pq:
            current_dist, current_id = heapq.heappop(pq)

            if current_id in visited:
                continue
            visited.add(current_id)

            if current_id == destination_id:
                break

            for neighbor_id in self._adjacency.get(current_id, set()):
                if neighbor_id in visited:
                    continue

                link = self.links.get((current_id, neighbor_id))
                if link is None or not link.active:
                    continue

                # Calculate edge weight based on optimization target
                if optimize_for == "latency":
                    weight = link.latency_ms
                else:  # bandwidth - use inverse for maximization
                    weight = 1.0 / link.bandwidth_gbps

                new_dist = distances[current_id] + weight
                if new_dist < distances[neighbor_id]:
                    distances[neighbor_id] = new_dist
                    predecessors[neighbor_id] = current_id
                    heapq.heappush(pq, (new_dist, neighbor_id))

        # Reconstruct path
        if distances[destination_id] == float('inf'):
            return Route(source_id, destination_id, [], 0, 0, 0, 0)

        path = []
        current = destination_id
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()

        # Calculate route metrics
        total_distance = 0.0
        total_latency = 0.0
        min_bandwidth = float('inf')

        for i in range(len(path) - 1):
            link = self.links.get((path[i], path[i + 1]))
            if link:
                total_distance += link.distance_km
                total_latency += link.latency_ms
                min_bandwidth = min(min_bandwidth, link.bandwidth_gbps)

        return Route(
            source_id=source_id,
            destination_id=destination_id,
            path=path,
            total_distance_km=round(total_distance, 2),
            total_latency_ms=round(total_latency, 3),
            min_bandwidth_gbps=min_bandwidth if min_bandwidth != float('inf') else 0,
            num_hops=len(path) - 1
        )

    def get_all_routes_from(self, source_id: str) -> Dict[str, Route]:
        """Get optimal routes from a node to all other nodes."""
        return {
            dest_id: self.find_route(source_id, dest_id)
            for dest_id in self.nodes
            if dest_id != source_id
        }

    def get_mesh_stats(self) -> Dict:
        """Get statistics about the mesh network."""
        active_links = sum(1 for link in self.links.values() if link.active) // 2
        avg_links_per_node = (2 * active_links / len(self.nodes)) if self.nodes else 0

        total_bandwidth = sum(link.bandwidth_gbps for link in self.links.values()) / 2
        avg_distance = sum(link.distance_km for link in self.links.values()) / (2 * active_links) if active_links else 0

        return {
            "total_nodes": len(self.nodes),
            "active_links": active_links,
            "avg_links_per_node": round(avg_links_per_node, 2),
            "total_bandwidth_gbps": round(total_bandwidth, 2),
            "avg_link_distance_km": round(avg_distance, 2),
        }

    def _calculate_distance(self, node1: OrbitalNode, node2: OrbitalNode) -> float:
        """Calculate 3D distance between two orbital nodes."""
        # Convert orbital elements to Cartesian (simplified)
        r1 = self.EARTH_RADIUS_KM + node1.orbit_altitude_km
        r2 = self.EARTH_RADIUS_KM + node2.orbit_altitude_km

        # Position based on mean anomaly (simplified circular orbit)
        theta1 = math.radians(node1.mean_anomaly_deg)
        theta2 = math.radians(node2.mean_anomaly_deg)

        inc1 = math.radians(node1.orbit_inclination_deg)
        inc2 = math.radians(node2.orbit_inclination_deg)

        raan1 = math.radians(node1.raan_deg)
        raan2 = math.radians(node2.raan_deg)

        # Simplified 3D position
        x1 = r1 * (math.cos(raan1) * math.cos(theta1) - math.sin(raan1) * math.sin(theta1) * math.cos(inc1))
        y1 = r1 * (math.sin(raan1) * math.cos(theta1) + math.cos(raan1) * math.sin(theta1) * math.cos(inc1))
        z1 = r1 * math.sin(theta1) * math.sin(inc1)

        x2 = r2 * (math.cos(raan2) * math.cos(theta2) - math.sin(raan2) * math.sin(theta2) * math.cos(inc2))
        y2 = r2 * (math.sin(raan2) * math.cos(theta2) + math.cos(raan2) * math.sin(theta2) * math.cos(inc2))
        z2 = r2 * math.sin(theta2) * math.sin(inc2)

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def _has_line_of_sight(self, node1: OrbitalNode, node2: OrbitalNode) -> bool:
        """Check if two nodes have unobstructed line of sight (Earth not blocking).

        subhadipmitra@: This is a simplified geometric check. The proper way is to:
        1. Parameterize the line between two satellites
        2. Find the closest point to Earth's center
        3. Check if that point is inside Earth's radius

        But for Walker constellations at similar altitudes, this approximation
        works well and is much faster. Revisit if we support multi-shell constellations.
        """
        # Simplified check - assumes LOS if both are at similar altitudes
        r1 = self.EARTH_RADIUS_KM + node1.orbit_altitude_km
        r2 = self.EARTH_RADIUS_KM + node2.orbit_altitude_km

        distance = self._calculate_distance(node1, node2)

        # Check if chord between satellites passes through Earth
        min_altitude = min(node1.orbit_altitude_km, node2.orbit_altitude_km)

        # Maximum distance before Earth blocks (horizon distance formula)
        # subhadipmitra@: Derived from Pythagorean theorem on the Earth-satellite-horizon triangle
        max_los_distance = 2 * math.sqrt(
            (self.EARTH_RADIUS_KM + min_altitude) ** 2 - self.EARTH_RADIUS_KM ** 2
        )

        return distance <= max_los_distance


def create_constellation(
    name: str,
    num_planes: int,
    sats_per_plane: int,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    isl_range_km: float = 5000.0
) -> SpaceMesh:
    """Create a Walker constellation mesh.

    Example:
        >>> mesh = create_constellation("test", num_planes=4, sats_per_plane=10)
        >>> print(mesh.get_mesh_stats())
    """
    mesh = SpaceMesh(default_isl_range_km=isl_range_km)

    for plane in range(num_planes):
        raan = (360.0 / num_planes) * plane

        for sat in range(sats_per_plane):
            mean_anomaly = (360.0 / sats_per_plane) * sat
            # Add phase offset between planes
            mean_anomaly += (360.0 / (num_planes * sats_per_plane)) * plane

            node_id = f"{name}_P{plane}_S{sat}"
            node = OrbitalNode(
                node_id=node_id,
                orbit_altitude_km=altitude_km,
                orbit_inclination_deg=inclination_deg,
                raan_deg=raan,
                mean_anomaly_deg=mean_anomaly,
                isl_range_km=isl_range_km
            )
            mesh.add_node(node)

    mesh.update_topology()
    return mesh
