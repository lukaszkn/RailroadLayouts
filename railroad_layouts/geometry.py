"""Rigid track geometry in board millimetres, with Y increasing down the drawing."""

from collections import defaultdict
import json
from dataclasses import dataclass, replace
from functools import cached_property
from math import ceil, copysign, cos, fsum, hypot, inf, isfinite, radians, sin
from pathlib import Path

from shapely.geometry import LineString

from .profiles import ROCO, TILLIG

TILLIG_SYSTEM = "Tillig TT Modellgleis"
TILLIG_INFO = dict(system=TILLIG_SYSTEM, scale="1:120", gauge_mm=12, rail_height_mm=2.07)


@dataclass(frozen=True)
class Pose:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0

    @property
    def xy(self):
        return self.x, self.y

    def local(self, other, mirror=1):
        a = radians(self.heading)
        return Pose(self.x + other.x * cos(a) - mirror * other.y * sin(a),
                    self.y + other.x * sin(a) + mirror * other.y * cos(a),
                    (self.heading + mirror * other.heading) % 360)

    def turned(self, angle):
        return Pose(self.x, self.y, (self.heading + angle) % 360)


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


def from_json(data):
    return Pose(data["x_mm"], data["y_mm"], data["heading_deg"])


@dataclass(frozen=True)
class Segment:
    length: float = 0
    radius: float = 0
    sweep: float = 0

    @property
    def distance(self):
        return self.radius * radians(abs(self.sweep)) if self.radius else self.length

    def advance(self, start, fraction=1.0):
        if not self.radius:
            return start.local(Pose(self.length * fraction, 0))
        sweep = self.sweep * fraction
        a = radians(sweep)
        r = self.radius if self.sweep > 0 else -self.radius
        return start.local(Pose(r * sin(a), r * (1 - cos(a)), sweep))


@dataclass(frozen=True)
class Route:
    start: str
    end: str
    segments: tuple[Segment, ...]

    @property
    def length(self):
        return sum(segment.distance for segment in self.segments)


@dataclass(frozen=True)
class Part:
    code: str
    data: dict
    ports: dict[str, Pose]
    routes: dict[str, Route]
    stop: Pose | None
    mounted_track: "Part | None" = None

    @property
    def kind(self):
        return self.data["kind"]

    @property
    def component_codes(self):
        if self.mounted_track:
            return self.code, self.mounted_track.code
        return (self.code,)

    @cached_property
    def radius(self):
        return self.data.get("clearance_radius_mm", min(
            (s.radius for route in self.routes.values() for s in route.segments if s.radius),
            default=float("inf"),
        ))

    @cached_property
    def track_length_mm(self):
        if self.mounted_track:
            return self.mounted_track.track_length_mm
        length = fsum(route.length for route in self.routes.values())
        if len(self.routes) < 2:
            return length

        # Coincident segments share a supporting line or circle. Crossings at
        # a single point have no length, even when their movements conflict.
        intervals = defaultdict(list)
        for route in self.routes.values():
            start = self.ports[route.start].turned(180)
            for segment in route.segments:
                if segment.radius:
                    side = copysign(1, segment.sweep)
                    center = start.local(Pose(0, side * segment.radius))
                    key = ("arc", round(center.x, 7), round(center.y, 7), segment.radius)
                    angle = start.heading - side * 90 + min(0, segment.sweep)
                    low = angle % 360
                    high = low + abs(segment.sweep)
                    intervals[key].append((radians(low) * segment.radius,
                                           radians(min(high, 360)) * segment.radius))
                    if high > 360:
                        intervals[key].append((0, radians(high - 360) * segment.radius))
                else:
                    angle = radians(round(start.heading % 180, 7) % 180)
                    key = ("line", angle, round(-start.x * sin(angle) + start.y * cos(angle), 7))
                    low = start.x * cos(angle) + start.y * sin(angle)
                    high = low + copysign(segment.length, cos(radians(start.heading) - angle))
                    intervals[key].append((min(low, high), max(low, high)))
                start = segment.advance(start)

        for ranges in intervals.values():
            end = -inf
            for low, high in sorted(ranges):
                length -= max(0, min(end, high) - low)
                end = max(end, high)
        return length


class Catalog(dict):
    """Part lookup with the metadata shared by every piece in a run."""

    def __init__(self, parts, info):
        super().__init__(parts)
        self.info = info

    def copy(self):
        return Catalog(self, self.info.copy())


def catalog_info(catalog):
    return getattr(catalog, "info", TILLIG_INFO)


def track_profile(catalog):
    return ROCO if catalog_info(catalog)["gauge_mm"] == 16.5 else TILLIG


def load_catalog(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    info = {key: raw[key] for key in TILLIG_INFO}
    if (info["scale"], info["gauge_mm"]) not in (("1:120", 12), ("1:87", 16.5)):
        raise ValueError("Supported profiles are 1:120 TT (12 mm) and 1:87 H0 (16.5 mm)")
    system, height = info["system"], info["rail_height_mm"]
    if (not isinstance(system, str) or not system.strip() or type(height) not in (int, float)
            or not isfinite(height) or height <= 0):
        raise ValueError("Catalog needs a system name and a positive rail height")
    catalog = Catalog({}, info)
    for code, data in raw["parts"].items():
        price = data.get("unit_price_eur_cents")
        if price is not None and (type(price) is not int or price < 0):
            raise ValueError(f"{code} unit_price_eur_cents must be a non-negative integer")
        ports = {key: from_json(value) for key, value in data.get("connectors", {}).items()}
        routes = {}
        stop = from_json(data["stop"]) if "stop" in data else None
        for item in data.get("routes", []):
            a, b = item["from"], item["to"]
            segments = tuple(Segment(v.get("length_mm", 0), v.get("radius_mm", 0),
                                     v.get("sweep_deg", 0)) for v in item["segments"])
            if not segments or not all(_valid_segment(s) for s in segments):
                raise ValueError(f"Invalid centerline for {code} {a}-{b}")
            p = ports[a].turned(180)
            for segment in segments:
                p = segment.advance(p)
            q = stop if b == "stop" else ports[b]
            if hypot(p.x - q.x, p.y - q.y) > 0.00001 or abs(angle_delta(p.heading, q.heading)) > 0.00001:
                raise ValueError(f"Centerline disagrees with connector for {code} {a}-{b}")
            routes[f"{a}-{b}"] = Route(a, b, segments)
        if data["kind"] != "reference" and not routes and "mount_track" not in data:
            raise ValueError(f"Missing geometry for {code}")
        if any(a not in routes or b not in routes for a, b in data.get("route_conflicts", [])):
            raise ValueError(f"Unknown conflicting route for {code}")
        catalog[code] = Part(code, data, ports, routes, stop)
    for code, part in catalog.items():
        if "mount_track" in part.data:
            catalog[code] = _mount_buffer(part, catalog)
    return catalog


def _mount_buffer(part, catalog):
    track = catalog.get(part.data["mount_track"])
    setback = part.data["stop_setback_mm"]
    route = track.routes.get("a-b") if track and track.kind == "straight" else None
    if (part.kind != "buffer" or part.ports or part.routes or part.stop or not route
            or len(route.segments) != 1 or route.segments[0].radius or not 0 < setback < route.length):
        raise ValueError(f"Invalid buffer mounting geometry for {part.code}")
    start = track.ports["a"]
    segment = Segment(length=route.length - setback)
    return replace(part, ports={"a": start}, stop=segment.advance(start.turned(180)),
                   routes={"a-stop": Route("a", "stop", (segment,))}, mounted_track=track)


def _valid_segment(segment):
    if not all(isfinite(v) for v in (segment.length, segment.radius, segment.sweep)):
        return False
    if segment.radius:
        return segment.length == 0 and segment.radius > 0 and 0 < abs(segment.sweep) <= 180
    return segment.length > 0 and segment.sweep == 0


@dataclass(frozen=True)
class Step:
    code: str
    route: str = "a-b"
    mirror: int = 1


@dataclass(frozen=True)
class Piece:
    part: Part
    origin: Pose
    mirror: int = 1

    @cached_property
    def ports(self):
        return {key: self.origin.local(value, self.mirror) for key, value in self.part.ports.items()}

    @property
    def stop(self):
        if self.part.stop:
            return self.origin.local(self.part.stop, self.mirror)
        return None

    @cached_property
    def lines(self):
        lines = {}
        for key, route in self.part.routes.items():
            start = self.ports[route.start].turned(180)
            points = [start.xy]
            for segment in route.segments:
                placed = Segment(segment.length, segment.radius, segment.sweep * self.mirror)
                # A 2 mm chord bounds the arc approximation below 0.002 mm.
                count = max(1, ceil(placed.distance / 2)) if placed.radius else 1
                points.extend(placed.advance(start, k / count).xy for k in range(1, count + 1))
                start = placed.advance(start)
            lines[key] = LineString(points)
        return lines

    @cached_property
    def bounds(self):
        bounds = [line.bounds for line in self.rail_lines]
        return (min(b[0] for b in bounds), min(b[1] for b in bounds),
                max(b[2] for b in bounds), max(b[3] for b in bounds))

    @cached_property
    def rail_lines(self):
        if self.part.mounted_track:
            far = self.origin.local(self.part.mounted_track.ports["b"])
            return (LineString((self.ports["a"].xy, far.xy)),)
        return tuple(self.lines.values())


def align(part, connector, joint, mirror=1):
    if mirror != 1 and (part.kind != "curve" or mirror != -1):
        raise ValueError(f"{part.code} cannot be reflected")
    port = part.ports[connector]
    origin = Pose(joint.x, joint.y, joint.heading + 180 - mirror * port.heading)
    return Piece(part, origin.local(Pose(-port.x, -port.y), mirror), mirror)


def residual(a, b):
    x, y = b.x - a.x, b.y - a.y
    t = radians(a.heading)
    return (x * cos(t) + y * sin(t), -x * sin(t) + y * cos(t),
            angle_delta(b.heading, a.heading + 180))


def joint_fits(a, b):
    longitudinal, lateral, angle = residual(a, b)
    return (-1e-6 <= longitudinal <= 0.500001 and abs(lateral) <= 0.150001
            and abs(angle) <= 0.100001)


def min_radius(group, gauge_mm=12):
    return {"A": 21.5, "B": 25, "C": 30}[group] * gauge_mm


def spacing(group, radius, gauge_mm=12):
    # NEM 112, page 2. Use the station spacing as the straight-track floor.
    tt = {
        "A": [(250, 40), (275, 39), (300, 38), (325, 37), (350, 36), (400, 35), (450, 34)],
        "B": [(300, 42), (325, 41), (350, 40), (400, 38), (450, 37), (500, 36), (550, 35), (600, 34)],
        "C": [(350, 43), (400, 41), (450, 40), (500, 38), (550, 37), (600, 36), (700, 35), (800, 34)],
    }
    h0 = {
        "A": [(350, 55), (400, 53), (450, 51)],
        "B": [(400, 59), (450, 57), (500, 55), (550, 53), (600, 52)],
        "C": [(450, 61), (500, 59), (550, 57), (600, 55), (700, 52)],
    }
    table, floor = (h0, 52) if gauge_mm == 16.5 else (tt, 38)
    rows = table[group]
    for (r0, d0), (r1, d1) in zip(rows, rows[1:]):
        if radius <= r1:
            return max(floor, d0 + (d1 - d0) * (max(radius, r0) - r0) / (r1 - r0))
    return float(floor)
