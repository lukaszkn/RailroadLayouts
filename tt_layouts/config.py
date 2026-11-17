"""Read and validate layout inputs and per-SKU inventory rules."""

from dataclasses import dataclass, field
import json
import math

MODES = ("single", "passing", "double", "mixed")
STRATEGIES = ("balanced", "longest", "longest_tracks", "shunting", "fewest_parts")


@dataclass(frozen=True)
class Limit:
    enabled: bool = True
    min_count: int = 0
    max_count: int | None = None

    def allows(self, count):
        return (self.min_count <= count
                and (self.max_count is None or count <= self.max_count)
                and (self.enabled or count == 0))


@dataclass(frozen=True)
class Config:
    width_cm: float = 160
    height_cm: float = 120
    count: int = 10
    mode: str = "mixed"
    strategy: str = "balanced"
    train_length_cm: float = 25
    seed: int = 42
    vehicle_group: str = "B"
    edge_margin_mm: float = 35
    parts: dict[str, Limit] = field(default_factory=dict)
    require_connected_tracks: bool = True
    min_turnouts: int = 1
    max_turnouts: int = 10
    ignore_similar_percentage: float = 0.0
    optimize_track_parts: bool = True
    require_forward_loop_return: bool = False
    min_station_tracks: int = 0
    max_station_tracks: int | None = None
    min_station_track_length_cm: float | None = None
    interior_style: str = "any"
    min_interior_sidings: int = 0

    @property
    def width(self):
        return self.width_cm * 10

    @property
    def height(self):
        return self.height_cm * 10

    @property
    def train_length(self):
        return self.train_length_cm * 10

    @property
    def require_station(self):
        return (self.min_station_tracks > 0 or self.max_station_tracks is not None
                or self.min_station_track_length_cm is not None)

    @property
    def station_track_length(self):
        return max(self.train_length, (self.min_station_track_length_cm or 0) * 10)

    @property
    def require_interior(self):
        return self.interior_style != "any" or self.min_interior_sidings > 0

    def available(self, code):
        limit = self.parts.get(code, Limit())
        return limit.enabled and limit.max_count != 0


def read_config(path, catalog, overrides=None):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("The layout config must be an object")
    data.update(overrides or {})
    extra = set(data) - Config.__dataclass_fields__.keys()
    if extra:
        raise ValueError(f"Unknown settings: {', '.join(sorted(extra))}")
    parts = data.pop("parts", {})
    if not isinstance(parts, dict):
        raise ValueError("parts must be an object keyed by Tillig SKU")
    limits = {}
    for code, rule in parts.items():
        if code not in catalog:
            raise ValueError(f"Unknown Tillig part {code}")
        if not isinstance(rule, dict) or set(rule) - Limit.__dataclass_fields__.keys():
            raise ValueError(f"Use enabled, min_count and max_count for {code}")
        limit = Limit(**rule)
        if (type(limit.enabled) is not bool or type(limit.min_count) is not int
                or limit.min_count < 0 or (limit.max_count is not None and (
                    type(limit.max_count) is not int or limit.max_count < limit.min_count))
                or (limit.min_count and not limit.enabled)):
            raise ValueError(f"Invalid inventory rule for {code}")
        if catalog[code].kind == "reference" and (limit.min_count or rule.get("enabled")):
            raise ValueError(f"{code} is reference-only. {catalog[code].data['notes']}")
        limits[code] = limit
    result = Config(**data, parts=limits)
    for key in ("width_cm", "height_cm", "train_length_cm", "edge_margin_mm"):
        n = getattr(result, key)
        if type(n) not in (int, float) or not math.isfinite(n) or n <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if result.edge_margin_mm < 12:
        raise ValueError("edge_margin_mm must leave at least 12 mm from centerline to the board edge")
    if type(result.count) is not int or result.count < 1 or type(result.seed) is not int:
        raise ValueError("count must be a positive integer and seed must be an integer")
    for key in ("require_connected_tracks", "optimize_track_parts", "require_forward_loop_return"):
        if type(getattr(result, key)) is not bool:
            raise ValueError(f"{key} must be a boolean")
    tolerance = result.ignore_similar_percentage
    if type(tolerance) not in (int, float) or not 0 <= tolerance <= 1:
        raise ValueError("ignore_similar_percentage must be a number between 0 and 1")
    for key in ("min_turnouts", "max_turnouts", "min_station_tracks", "min_interior_sidings"):
        count = getattr(result, key)
        if type(count) is not int or count < 0:
            raise ValueError(f"{key} must be a non-negative integer")
    if result.max_turnouts < result.min_turnouts:
        raise ValueError("max_turnouts must be at least min_turnouts")
    station_max = result.max_station_tracks
    if station_max is not None and (type(station_max) is not int or station_max < 2
                                    or station_max < result.min_station_tracks):
        raise ValueError("max_station_tracks must be null or an integer at least 2 and at least min_station_tracks")
    station_length = result.min_station_track_length_cm
    if station_length is not None and (type(station_length) not in (int, float)
                                       or not math.isfinite(station_length) or station_length <= 0):
        raise ValueError("min_station_track_length_cm must be null or a finite positive number")
    if result.mode not in MODES or result.strategy not in STRATEGIES or result.vehicle_group not in ("A", "B", "C"):
        raise ValueError("Check mode, strategy and vehicle_group against the README")
    if result.require_station and result.mode not in ("double", "mixed"):
        raise ValueError("Station track settings require double or mixed mode")
    if result.interior_style not in ("any", "curved_yard"):
        raise ValueError("interior_style must be any or curved_yard")
    return result
