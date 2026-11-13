"""Assemble pieces and validate the resulting physical and operational routes."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, groupby, pairwise

import numpy as np
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .geometry import Piece, Pose, align, joint_fits, min_radius, residual, spacing


@dataclass(frozen=True)
class Move:
    piece: int
    entry: str
    exit: str

    def reversed(self):
        return Move(self.piece, self.exit, self.entry)


@dataclass(frozen=True)
class Track:
    name: str
    moves: tuple[Move, ...]
    kind: str
    parent: int | None = None


class Layout:
    def __init__(self, catalog, mode, family, title, instructions):
        self.catalog = catalog
        self.mode = mode
        self.family = family
        self.title = title
        self.instructions = instructions
        self.pieces = []
        self.tracks = []
        self.joints = []
        self.lengths = {}
        self.parallel_mm = 0

    def route(self, move):
        part = self.pieces[move.piece].part
        return part.routes.get(f"{move.entry}-{move.exit}") or part.routes[f"{move.exit}-{move.entry}"]

    def line(self, move):
        route = self.route(move)
        line = self.pieces[move.piece].lines[f"{route.start}-{route.end}"]
        return line if move.entry == route.start else line.reverse()

    def port(self, move, end=False):
        piece = self.pieces[move.piece]
        key = move.exit if end else move.entry
        return piece.stop if key == "stop" else piece.ports[key]

    def lay(self, start, steps):
        moves = []
        for step in steps:
            a, b = step.route.split("-")
            piece = align(self.catalog[step.code], a, start, step.mirror)
            moves.append(Move(len(self.pieces), a, b))
            self.pieces.append(piece)
            start = self.port(moves[-1], end=True)
        return tuple(moves)

    def branch(self, move, entry="a", exit="c"):
        branch = Move(move.piece, entry, exit)
        self.route(branch)
        return branch

    def add(self, name, moves, kind, parent=None):
        self.tracks.append(Track(name, tuple(moves), kind, parent))
        return len(self.tracks) - 1

    def line_for_track(self, track):
        # Remove floating-point duplicates while retaining real residuals.
        coords = ((round(x, 8), round(y, 8)) for move in track.moves
                  for x, y in self.line(move).coords)
        return LineString([xy for xy, _ in groupby(coords)])

    def length_for_track(self, track):
        return sum(self.route(move).length for move in track.moves)

    @property
    def inventory(self):
        return Counter(piece.part.code for piece in self.pieces)

    @property
    def lap(self):
        main = self.lengths.get("lap", 0)
        # The passing route can replace the measured segment between station points.
        return max(main, self.lengths.get("passing_lap", 0), self.lengths.get("inner_lap", 0))

    def fit(self, cfg):
        x0, y0, x1, y1 = self.bounds()
        if x1 - x0 + 2 * cfg.edge_margin_mm <= cfg.width + 1e-6 and y1 - y0 + 2 * cfg.edge_margin_mm <= cfg.height + 1e-6:
            transform = Pose((cfg.width - x1 - x0) / 2, (cfg.height - y1 - y0) / 2)
        elif x1 - x0 + 2 * cfg.edge_margin_mm <= cfg.height + 1e-6 and y1 - y0 + 2 * cfg.edge_margin_mm <= cfg.width + 1e-6:
            transform = Pose((cfg.width + y1 + y0) / 2, (cfg.height - x1 - x0) / 2, 90)
        else:
            return False
        self.pieces = [Piece(piece.part, transform.local(piece.origin), piece.mirror) for piece in self.pieces]
        return True

    def bounds(self):
        boxes = [p.bounds for p in self.pieces]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))

    def centered_tracks(self):
        x0, y0, x1, y1 = self.bounds()
        return tuple((track.kind, translate(self.line_for_track(track).simplify(0.01),
                       -(x0 + x1) / 2, -(y0 + y1) / 2)) for track in self.tracks)


def equivalent(first, second):
    """Compare centered track geometry after rotation, independent of rail joints."""
    if len(first) != len(second):
        return False
    for angle in (0, 90, 180, 270):
        unmatched = list(second)
        for kind, line in first:
            geometry = rotate(line, angle, origin=(0, 0)) if angle else line
            for index, (other_kind, other) in enumerate(unmatched):
                if (kind == other_kind and abs(geometry.length - other.length) <= 1
                        and all(abs(a - b) <= 0.15 for a, b in zip(geometry.bounds, other.bounds))
                        and geometry.hausdorff_distance(other) <= 0.15):
                    unmatched.pop(index)
                    break
            else:
                break
        if not unmatched:
            return True
    return False


def _check_joints(layout):
    layout.joints.clear()
    joints = {}
    used = set()
    for track in layout.tracks:
        transitions = list(pairwise(track.moves))
        if track.kind == "loop":
            transitions.append((track.moves[-1], track.moves[0]))
        for move in track.moves:
            layout.route(move)
            used.add((move.piece, move.entry))
            used.add((move.piece, move.exit))
        for a, b in transitions:
            first, second = (a.piece, a.exit), (b.piece, b.entry)
            if first == second:
                return "zero-length joint"
            if (first in joints and joints[first] != second) or (second in joints and joints[second] != first):
                return "connector has multiple mates"
            if not joint_fits(layout.port(a, True), layout.port(b)):
                return "factory-length joint does not close"
            if first not in joints:
                layout.joints.append((first, second, residual(layout.port(a, True), layout.port(b))))
            joints[first], joints[second] = second, first
    for i, piece in enumerate(layout.pieces):
        for port in piece.ports:
            if (i, port) not in joints:
                return f"unconnected {piece.part.code} connector"
        if piece.stop and (i, "stop") not in used:
            return "unreachable buffer stop"
    return None


def _tracks_connected(layout):
    neighbors = defaultdict(set)
    for i, piece in enumerate(layout.pieces):
        for port in piece.ports:
            neighbors.setdefault((i, port), set())
        if piece.part.stop:
            neighbors.setdefault((i, "stop"), set())
        # Include switchable routes even when the named tracks use only the through routes.
        for route in piece.part.routes.values():
            first, second = (i, route.start), (i, route.end)
            neighbors[first].add(second)
            neighbors[second].add(first)
    for first, second, _ in layout.joints:
        neighbors[first].add(second)
        neighbors[second].add(first)

    pending = [next(iter(neighbors))] if neighbors else []
    visited = set(pending)
    while pending:
        reachable = neighbors[pending.pop()] - visited
        visited.update(reachable)
        pending.extend(reachable)
    return len(visited) == len(neighbors)


def _intersections(layout):
    lines, owners = [], []
    for i, piece in enumerate(layout.pieces):
        for line in piece.lines.values():
            lines.append(line)
            owners.append(i)
    allowed = {}
    for (a, pa), (b, _), _ in layout.joints:
        allowed.setdefault(tuple(sorted((a, b))), []).append(Point(layout.pieces[a].ports[pa].xy))
    for i, j in STRtree(lines).query(lines, predicate="intersects").T:
        if j <= i or owners[i] == owners[j]:
            continue
        intersection = lines[i].intersection(lines[j])
        if intersection.is_empty:
            continue
        key = owners[i], owners[j]
        if any(point.buffer(0.55).covers(intersection) for point in allowed.get(key, ())):
            continue
        return f"unplanned intersection ({layout.pieces[key[0]].part.code} and {layout.pieces[key[1]].part.code})"
    return None


def _points(line):
    ds = np.linspace(0, line.length, max(2, int(line.length / 2) + 1))
    return ds, shapely.line_interpolate_point(line, ds)


def _route_pair(layout, a, b, group):
    """Find clear intervals and allow fouling only at named physical junctions."""
    first = layout.tracks[a]
    second = layout.tracks[b]
    l1, l2 = layout.line_for_track(first), layout.line_for_track(second)
    common = {m.piece for m in first.moves} & {m.piece for m in second.moves}
    common = {i for i in common if layout.pieces[i].part.kind != "buffer"}
    group_spacing = spacing(group, min(
        layout.pieces[m.piece].part.radius for m in first.moves + second.moves))
    ds, samples = _points(l1)
    close = shapely.distance(samples, l2) < group_spacing - 0.01
    for start, end in _runs(close):
        # A below-clearance interval must include real shared pointwork.
        # Beyond the pointwork, bound the permitted fan-out along the rail.
        if not common:
            return None
        projected = []
        for move in first.moves:
            if move.piece not in common:
                continue
            line = layout.line(move)
            values = [l1.project(Point(line.coords[0])), l1.project(Point(line.coords[-1]))]
            if first.kind == "loop" and abs(values[1]-values[0]) > l1.length/2:
                projected.extend(((0, min(values)), (max(values), l1.length)))
            else:
                projected.append((min(values), max(values)))
        vicinity = np.zeros(end - start + 1, dtype=bool)
        for lo, hi in projected:
            near = (ds[start:end+1] >= lo - 230) & (ds[start:end+1] <= hi + 230)
            if any(near):
                vicinity |= near
        if not all(vicinity) or not any(
            ds[start] <= hi + 2.1 and ds[end] >= lo - 2.1 for lo, hi in projected
        ):
            return None
    return close


def _runs(mask):
    return zip(np.flatnonzero(mask & ~np.r_[False, mask[:-1]]),
               np.flatnonzero(mask & ~np.r_[mask[1:], False]))


def _usable(ds, close):
    # Use the samples on the clear side of each fouling boundary.
    clear = ~close
    length = max((ds[b] - ds[a] for a, b in _runs(clear)), default=0.0)
    return float(max(0.0, length - 20))


def _clearance(layout, cfg):
    loops = [t for t in layout.tracks if t.kind == "loop"]
    for a, b in combinations(loops, 2):
        lines_a = [layout.line(move) for move in a.moves]
        lines_b = [layout.line(move) for move in b.moves]
        radii_a = [layout.pieces[move.piece].part.radius for move in a.moves]
        radii_b = [layout.pieces[move.piece].part.radius for move in b.moves]
        limit = spacing(cfg.vehicle_group, min(radii_a + radii_b))
        for i, j in STRtree(lines_b).query(lines_a, predicate="dwithin", distance=limit).T:
            m, n = a.moves[i], b.moves[j]
            if m.piece == n.piece:
                keys = {f"{layout.route(m).start}-{layout.route(m).end}",
                        f"{layout.route(n).start}-{layout.route(n).end}"}
                part = layout.pieces[m.piece].part
                if len(keys) != 2 or any(keys == set(conflict) for conflict in part.data.get("route_conflicts", [])):
                    return "continuous routes conflict at pointwork"
            radius = min(radii_a[i], radii_b[j])
            if lines_a[i].distance(lines_b[j]) < spacing(cfg.vehicle_group, radius) - 0.01:
                return "parallel curve clearance for selected rolling stock"
    for i, track in enumerate(layout.tracks):
        if track.kind == "loop":
            continue
        line = layout.line_for_track(track)
        if not line.is_simple:
            return "route crosses itself"
        ds, _ = _points(line)
        fouled = np.zeros(len(ds), dtype=bool)
        for j, other in enumerate(layout.tracks):
            if i == j:
                continue
            result = _route_pair(layout, i, j, cfg.vehicle_group)
            if result is None:
                return f"track clearance between {track.name} and {other.name}"
            fouled |= result
        usable = _usable(ds, fouled)
        layout.lengths[track.name] = usable
        if track.kind == "passing":
            # Compare the exact main route between the shared turnouts, in the
            # direction in which the siding travels.
            parent = layout.tracks[track.parent]
            first, last = track.moves[0].piece, track.moves[-1].piece
            indices = {m.piece: k for k, m in enumerate(parent.moves)}
            start, end = indices[first], indices[last]
            if parent.moves[start].entry != track.moves[0].entry:
                start, end = end, start
            leg = parent.moves[start:end+1] if start <= end else parent.moves[start:] + parent.moves[:end+1]
            main_line = layout.line_for_track(Track("", leg, "leg"))
            ms, mp = _points(main_line)
            main_usable = _usable(ms, shapely.distance(mp, line) < spacing(cfg.vehicle_group, min(
                layout.pieces[m.piece].part.radius for m in leg + track.moves)) - 0.01)
            layout.lengths["station_main"] = main_usable
            layout.lengths["passing_lap"] = (layout.length_for_track(parent)
                - sum(layout.route(move).length for move in leg) + layout.length_for_track(track))
            if min(main_usable, usable) < cfg.train_length:
                return "configured train exceeds usable passing length"
    return None


def validate(layout, cfg):
    if not layout.pieces:
        return "missing track"
    counts = layout.inventory
    for code, limit in cfg.parts.items():
        if not limit.allows(counts[code]):
            return f"inventory limit for {code}"
    running_pieces = {move.piece for track in layout.tracks if track.kind != "spur" for move in track.moves}
    if any(piece.part.radius < min_radius(cfg.vehicle_group)
           or (piece.part.radius == 267 and index in running_pieces)
           for index, piece in enumerate(layout.pieces)):
        return "minimum curve radius for selected rolling stock"
    if not layout.fit(cfg):
        return "board size or edge margin"
    failure = _check_joints(layout)
    if failure:
        return failure
    if cfg.require_connected_tracks and not _tracks_connected(layout):
        return "track network is disconnected"
    failure = _intersections(layout)
    if failure:
        return failure
    loops = [track for track in layout.tracks if track.kind == "loop"]
    if len(loops) != (2 if layout.mode == "double" else 1):
        return "continuous route count"
    for i, track in enumerate(loops):
        line = layout.line_for_track(track)
        if not line.is_simple:
            return "continuous route crosses itself"
        length = layout.length_for_track(track)
        if length < cfg.train_length:
            return "configured train exceeds continuous route length"
        layout.lengths["lap" if i == 0 else "inner_lap"] = length
    if layout.mode == "double" and layout.parallel_mm < cfg.train_length:
        return "configured train exceeds parallel running length"
    failure = _clearance(layout, cfg)
    if failure:
        return failure
    if layout.mode == "single" and max(
        (layout.lengths[t.name] for t in layout.tracks if t.kind == "spur"), default=0
    ) < min(100, cfg.train_length):
        return "no usable freight siding"
    return None
