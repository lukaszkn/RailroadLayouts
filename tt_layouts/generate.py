"""Search stock track formations. Each candidate uses the same independent validator."""

from collections import Counter
from functools import lru_cache
from itertools import product
import logging
from math import ceil
from random import Random

from .config import MODES
from .geometry import Pose, Step, min_radius, spacing
from .inventory import repartition
from .model import Layout, equivalent, validate

BUFFER = Step("83100", "a-stop")
logger = logging.getLogger(__name__)


def backwards(moves):
    return tuple(move.reversed() for move in reversed(moves))


def mm(steps, catalog):
    return sum(catalog[s.code].routes[s.route].length for s in steps)


class Stock:
    def __init__(self, catalog, cfg):
        self.catalog = catalog
        self.cfg = cfg
        self.straights = sorted(
            ((round(p.routes["a-b"].length * 10), code) for code, p in catalog.items()
             if p.kind == "straight" and cfg.available(code)), reverse=True)
        self.curves = {}
        for code, part in catalog.items():
            if part.kind == "curve" and cfg.available(code) and part.radius >= min_radius(cfg.vehicle_group):
                self.curves.setdefault(part.radius, []).append(
                    (round(abs(part.routes["a-b"].segments[0].sweep) * 2), code))
        # Reordering equal parts has no effect on placement.
        self._lengths = {0: ()}
        self._searched = 0

    def available(self, *steps):
        return all(self.cfg.available(step.code) for step in steps)

    def _cost(self, codes):
        return len(codes), tuple(sorted(codes))

    def straight(self, length):
        target = round(length * 10)
        if target < 0 or abs(target - length * 10) > 1e-5:
            return None
        if target > self._searched:
            for key in range(self._searched + 1, target + 1):
                best = None
                for size, code in self.straights:
                    prev = self._lengths.get(key - size)
                    if prev is None:
                        continue
                    current = prev + (code,)
                    if best is None or self._cost(current) < self._cost(best):
                        best = current
                if best is not None:
                    self._lengths[key] = best
            self._searched = target
        if target in self._lengths:
            return tuple(Step(code) for code in self._lengths[target])
        return None

    @lru_cache(maxsize=256)
    def curve(self, radius, degrees):
        target = round(abs(degrees) * 2)
        options = self.curves.get(radius, [])
        paths = {0: ()}
        for n in range(1, target + 1):
            for sweep, code in options:
                prev = paths.get(n - sweep)
                if prev is None:
                    continue
                path = prev + (code,)
                if n not in paths or self._cost(path) < self._cost(paths[n]):
                    paths[n] = path
        if target in paths:
            return tuple(Step(code, mirror=1 if degrees > 0 else -1) for code in paths[target])
        return None

    def runs(self, low, high, step=35):
        """Try increasing or decreasing fitted lengths with roughly step mm between plans."""
        value = high
        while value >= low:
            target = round(value * 10)
            lower = max(round(low * 10), target - round(step * 10))
            for length in range(target, lower - 1, -1):
                run = self.straight(length / 10)
                if run is not None:
                    yield length / 10, run
                    value = length / 10
                    break
            value -= step

    def side_lengths(self, available, defaults):
        if self.cfg.strategy == "longest":
            longest = next(self.runs(0, available), None)
            if longest and longest[0] not in defaults:
                return (*defaults, longest[0])
        return defaults


def running_loop(layout, start, forward, back, bend, sides=()):
    """Clockwise oval or rounded rectangle, beginning on the upper straight."""
    upper = layout.lay(start, forward)
    current = layout.port(upper[-1], True)
    cap = bend + sides + bend
    lower = layout.lay(current, cap + back + cap)
    return upper, lower


def from_port(layout, move, steps, entry="a", exit="c"):
    branch = layout.branch(move, entry, exit)
    return (branch,) + layout.lay(layout.port(branch, True), steps)


def spur(layout, move, steps, name, parent, *, entry="a", exit="c"):
    moves = from_port(layout, move, steps + (BUFFER,), entry, exit)
    return layout.add(name, moves, "spur", parent)


def ordinary_modes(stock):
    """Single track freight layouts, with one or more switching leads."""
    cfg = stock.cfg
    span, depth = sorted((cfg.width, cfg.height), reverse=True)
    margin = cfg.edge_margin_mm
    if not cfg.available(BUFFER.code):
        return
    turnouts = [p for p in stock.catalog.values() if p.kind in ("turnout", "three_way")
                and cfg.available(p.code)]
    for radius, point in product(sorted(stock.curves, reverse=True), turnouts):
        bend = stock.curve(radius, 90)
        if radius == 267 or bend is None or 2 * radius + 2 * margin > depth:
            continue
        switch = Step(point.code)
        length = point.routes["a-b"].length
        branch_sign = 1 if point.ports["c"].y > 0 else -1
        turnout_sweep = point.routes["a-c"].segments[0].sweep
        spur_recipe = stock.curve(353, -turnout_sweep)
        for lead in (83, 41.5, 0):
            approach = stock.straight(lead)
            if approach is None:
                continue
            high = span - 2 * radius - 2 * margin - length - lead
            for remaining, run in stock.runs(120, high, 65):
                forward = approach + (switch,) + run
                back = stock.straight(lead + length + remaining)
                if back is None:
                    continue
                for side_len in stock.side_lengths(depth-2*radius-2*margin, (0, 83, 166)):
                    sides = stock.straight(side_len)
                    if sides is None or 2 * radius + side_len + 2 * margin > depth:
                        continue
                    for variant in ("diagonal", "parallel", "ladder"):
                        if variant != "diagonal" and (spur_recipe is None or point.kind == "three_way"):
                            continue
                        extra = stock.straight(83)
                        if variant != "diagonal" and extra is None:
                            continue
                        # Reserve length beyond the fouling point so a locomotive
                        # and wagons have room to clear the through track.
                        for tail_mm in (290.5, 249, 207.5, 166):
                            tail = stock.straight(tail_mm)
                            if tail is None:
                                continue
                            names = {
                                "diagonal": ("Freight spur", "Work the freight spur from the main line."),
                                "parallel": ("Goods depot", "Leave wagons on the goods road while the main line stays clear."),
                                "ladder": ("Freight yard", "Sort wagons between the yard road and the loading spur."),
                            }
                            title, instructions = names[variant]
                            if point.kind == "three_way":
                                title = "Three-way goods junction"
                                instructions = "Use the two diverging roads for different freight customers."
                            layout = Layout(stock.catalog, "single", variant if point.kind != "three_way" else "three_way",
                                            title, instructions)
                            top, rest = running_loop(layout, Pose(), forward, back, bend, sides)
                            main = layout.add("Main line", top + rest, "loop")
                            turnout = top[len(approach)]
                            if variant == "diagonal":
                                spur(layout, turnout, tail, "Goods road", main)
                                if point.kind == "three_way":
                                    spur(layout, turnout, tail, "Loading spur", main, exit="d")
                            elif variant == "parallel":
                                spur(layout, turnout, extra + spur_recipe + tail, "Goods road", main)
                            else:
                                yard_point = "83331" if branch_sign > 0 else "83332"
                                if not cfg.available(yard_point):
                                    continue
                                yard = from_port(layout, turnout, extra + spur_recipe +
                                                 (Step(yard_point),) + tail + (BUFFER,))
                                parent = layout.add("Yard road", yard, "spur", main)
                                spur(layout, yard[len(extra) + len(spur_recipe) + 1], tail, "Loading spur", parent)
                            yield layout


def _branches(stock, part):
    turn = part.routes["a-c"].segments[0].sweep
    for radius, lead in product(sorted(stock.curves), (41.5, 43, 83, 119.5, 36.5, 0)):
        curve = stock.curve(radius, -turn)
        straight = stock.straight(lead)
        if curve and straight is not None:
            yield straight + curve


def _end(stock, start, steps):
    # Temporary placement uses the catalog ports in exactly the same way as
    # the final assembly. No geometry is calculated from a template drawing.
    temp = Layout(stock.catalog, "", "", "", "")
    moves = temp.lay(start, steps)
    return temp.port(moves[-1], True) if moves else start


def passing_modes(stock):
    cfg = stock.cfg
    span, depth = sorted((cfg.width, cfg.height), reverse=True)
    margin = cfg.edge_margin_mm
    running_radii = [r for r in sorted(stock.curves, reverse=True) if r != 267]
    if not running_radii:
        return
    turnouts = [p for p in stock.catalog.values() if p.kind == "turnout" and cfg.available(p.code)]
    for left, right in product(turnouts, repeat=2):
        if left.ports["c"].y * right.ports["c"].y >= 0:
            continue
        length = left.routes["a-b"].length + right.routes["a-b"].length
        first = Step(left.code)
        last = Step(right.code, "b-a")
        a = _end(stock, Pose(), (Step(left.code, "a-c"),))
        b = _end(stock, Pose(0, 0, 180), (Step(right.code, "a-c"),))
        left_legs = [(steps, _end(stock, a, steps)) for steps in _branches(stock, left)]
        right_legs = [(steps, _end(stock, b, steps)) for steps in _branches(stock, right)]
        for (lead, p), (tail, q) in product(left_legs, right_legs):
            if abs(p.y - q.y) > .15 or abs(p.y) < 38:
                continue
            gap = p.x - q.x - length
            # 20 mm of stopping room on the platform. Try additional stock
            # length so actual fouling geometry can decide the usable length.
            for mid_mm, mid in stock.runs(cfg.train_length + 10, span - 2*min(running_radii) - 2*margin - p.x + q.x, 20):
                target = gap + mid_mm
                for tenths in range(ceil((target - 0.000001) * 10), int((target + .5) * 10) + 1):
                    fill = stock.straight(tenths / 10)
                    if fill is None:
                        continue
                    full = length + tenths / 10
                    dimensions = ((radius, side_mm) for radius in running_radii
                                  for side_mm in stock.side_lengths(depth-2*radius-2*margin, (0, 83, 166, 249)))
                    for radius, side_mm in dimensions:
                        if full + 2*radius + 2*margin > span + 1e-6 or 2*radius + side_mm + 2*margin > depth + 1e-6:
                            continue
                        bend, sides = stock.curve(radius, 90), stock.straight(side_mm)
                        back = stock.straight(full)
                        if bend is None or sides is None or back is None:
                            continue
                        family = "passing_oval" if side_mm == 0 else "passing_rectangle"
                        title = "Oval passing station" if side_mm == 0 else "Passing station"
                        layout = Layout(stock.catalog, "passing", family, title,
                                        "Hold one train on the station loop while the other uses the main line.")
                        top, rest = running_loop(layout, Pose(), (first,) + fill + (last,), back, bend, sides)
                        main = layout.add("Main line", top + rest, "loop")
                        left_leg = from_port(layout, top[0], lead + mid)
                        right_leg = from_port(layout, top[-1], tail)
                        layout.add("Passing track", left_leg + backwards(right_leg), "passing", main)
                        yield layout
                        # Where room permits, a station gets a separate industry
                        # road off the opposite side of the main.
                        if not cfg.available("83100"):
                            continue
                        freight_point = next((v for v in turnouts if v.ports["c"].y > 0), None)
                        if freight_point is None:
                            continue
                        fp_length = freight_point.routes["a-b"].length
                        prefix = stock.straight(83)
                        suffix = stock.straight(full - fp_length - 83)
                        freight_tail = stock.straight(249)
                        if None in (prefix, suffix, freight_tail):
                            continue
                        freight_family = "passing_freight_in" if p.y > 0 else "passing_freight_out"
                        freight_title = ("Passing station and freight" if p.y > 0
                                         else "Outside passing track and freight")
                        freight = Layout(stock.catalog, "passing", freight_family, freight_title,
                                         "Make a meet at the station, then spot wagons on the separate goods road.")
                        freight_back = prefix + (Step(freight_point.code),) + suffix
                        top, rest = running_loop(freight, Pose(), (first,) + fill + (last,), freight_back, bend, sides)
                        main = freight.add("Main line", top + rest, "loop")
                        left_leg = from_port(freight, top[0], lead + mid)
                        right_leg = from_port(freight, top[-1], tail)
                        freight.add("Passing track", left_leg + backwards(right_leg), "passing", main)
                        spur(freight, rest[2*len(bend)+len(sides)+len(prefix)], freight_tail, "Goods road", main)
                        yield freight


def double_modes(stock):
    cfg = stock.cfg
    span, depth = sorted((cfg.width, cfg.height), reverse=True)
    margin = cfg.edge_margin_mm
    running_radii = [r for r in sorted(stock.curves, reverse=True) if r != 267]
    for outer, inner in product(running_radii, repeat=2):
        offset = outer - inner
        if offset < spacing(cfg.vehicle_group, inner) or 2*outer + 2*margin > depth:
            continue
        bend, inner_bend = stock.curve(outer, 90), stock.curve(inner, 90)
        if bend is None or inner_bend is None:
            continue
        high = span - 2*outer - 2*margin
        for side_mm in stock.side_lengths(depth-2*outer-2*margin, (0, 83, 166, 249)):
            sides = stock.straight(side_mm)
            if sides is None or 2*outer + side_mm + 2*margin > depth:
                continue
            if offset == 43 and cfg.available("83212"):
                # Find a stock prefix such that its length plus 251 mm
                # has an exact, all-straight counterpart on the return side.
                prefixes = [(v, stock.straight(v), stock.straight(v+251))
                            for v in (122.5, 164, 143.8, 159, 165.5, 205.5)]
                for prefix_mm, prefix, opposite in prefixes:
                    if prefix is None or opposite is None:
                        continue
                    extensions = list(stock.runs(max(0, cfg.train_length - prefix_mm - 251),
                                                 high - prefix_mm - 251, 55))
                    if prefix_mm + 251 >= cfg.train_length and prefix_mm + 251 <= high:
                        extensions.append((0, ()))
                    for extension, rest in extensions:
                        title = "Double track crossover"
                        layout = Layout(stock.catalog, "double", "crossover", title,
                                        "Run a train on each loop with the crossover lined straight. Stop the other train before crossing over.")
                        outer_top, outer_rest = running_loop(
                            layout, Pose(), prefix + (Step("83212"),) + rest, opposite + rest, bend, sides)
                        layout.add("Outer main", outer_top + outer_rest, "loop")
                        inner_prefix = layout.lay(Pose(0, offset, 0), prefix)
                        cross = layout.branch(outer_top[len(prefix)], "c", "d")
                        following = layout.lay(layout.port(cross, True), rest + inner_bend + sides
                                               + inner_bend + opposite + rest + inner_bend + sides + inner_bend)
                        layout.add("Inner main", inner_prefix + (cross,) + following, "loop")
                        layout.parallel_mm = prefix_mm + 251 + extension
                        yield layout
                        length = layout.parallel_mm
                        if not stock.available(BUFFER, Step("83331")):
                            continue
                        lead = stock.straight(83)
                        after = stock.straight(length - 249)
                        tail = stock.straight(249)
                        if None in (lead, after, tail):
                            continue
                        freight = Layout(stock.catalog, "double", "crossover_freight", "Crossover and goods road",
                                         "Hold the other train while the local crosses over, then switch the goods road from the inner main.")
                        top, lower = running_loop(freight, Pose(), prefix + (Step("83212"),) + rest,
                                                  opposite + rest, bend, sides)
                        freight.add("Outer main", top + lower, "loop")
                        inner_top = freight.lay(Pose(0, offset, 0), prefix)
                        cross = freight.branch(top[len(prefix)], "c", "d")
                        cap = inner_bend + sides + inner_bend
                        inner_lower = freight.lay(freight.port(cross, True),
                            rest + cap + lead + (Step("83331"),) + after + cap)
                        main = freight.add("Inner main", inner_top + (cross,) + inner_lower, "loop")
                        spur(freight, inner_lower[len(rest) + len(cap) + len(lead)], tail, "Goods road", main)
                        freight.parallel_mm = length
                        yield freight
            for length, straight in stock.runs(cfg.train_length, high, 55):
                layout = Layout(stock.catalog, "double", "parallel",
                                "Independent double track", "Run passenger and freight trains simultaneously on their own tracks.")
                top, rest = running_loop(layout, Pose(), straight, straight, bend, sides)
                layout.add("Outer main", top + rest, "loop")
                top, rest = running_loop(layout, Pose(0, offset, 0), straight, straight, inner_bend, sides)
                layout.add("Inner main", top + rest, "loop")
                layout.parallel_mm = length
                yield layout
                if not stock.available(BUFFER, Step("83331")):
                    continue
                point = stock.catalog["83331"]
                lead = stock.straight(83)
                after = stock.straight(length - 83 - point.routes["a-b"].length)
                tail = stock.straight(249)
                if None in (lead, after, tail):
                    continue
                freight = Layout(stock.catalog, "double", "double_freight", "Double track and freight",
                                 "Keep the outer main running while a local works the goods road from the inner main.")
                top, rest = running_loop(freight, Pose(), straight, straight, bend, sides)
                freight.add("Outer main", top + rest, "loop")
                top, rest = running_loop(freight, Pose(0, offset, 0),
                                         lead + (Step("83331"),) + after, straight, inner_bend, sides)
                inner_main = freight.add("Inner main", top + rest, "loop")
                spur(freight, top[len(lead)], tail, "Goods road", inner_main)
                freight.parallel_mm = length
                yield freight


def specialty_modes(stock):
    cfg = stock.cfg
    if not cfg.available(BUFFER.code):
        return
    for code in ("83161", "83170", "83300"):
        # The offset curves bring the branch back toward the crossing. At the
        # crossing it traverses the other through route and continues to a buffer.
        if not cfg.available(code):
            continue
        is_k2 = code == "83170"
        formations = (("83329", 364.4), ("83332", 364.5), ("83332", 364.6)) if is_k2 else (
            ("83329", 177.3), ("83332", 177.3))
        cross_route = "c-d"
        for point, offset in formations:
            if not cfg.available(point):
                continue
            lead = stock.straight(offset)
            flat = stock.straight(119.5 if is_k2 else 0)
            r22 = stock.curve(353, 30)
            r12 = stock.curve(310, 15) if is_k2 else ()
            if lead is None or flat is None or r22 is None or r12 is None:
                continue
            branch = flat + r22 + r12
            crossing = Step(code, cross_route)
            for radius in sorted(stock.curves, reverse=True):
                if radius == 267:
                    continue
                bend = stock.curve(radius, 90)
                if not bend:
                    continue
                for suffix_mm, tail_mm in product((166, 249, 332), (207.5, 249, 332)):
                    suffix = stock.straight(suffix_mm)
                    spur_tail = stock.straight(tail_mm)
                    if suffix is None or spur_tail is None:
                        continue
                    top = (Step(point),) + lead + (crossing,) + suffix
                    back = stock.straight(mm(top, stock.catalog))
                    if back is None:
                        continue
                    family = stock.catalog[code].kind
                    title = "Double slip freight junction" if family == "slip" else "Diamond freight junction"
                    layout = Layout(stock.catalog, "single", family, title,
                                    "Hold main-line traffic clear while the local crosses to the goods road.")
                    upper, rest = running_loop(layout, Pose(), top, back, bend)
                    main = layout.add("Main line", upper + rest, "loop")
                    approach = from_port(layout, upper[0], branch)
                    crossing_move = layout.branch(upper[-len(suffix)-1], "a", "b")
                    exit_path = layout.lay(layout.port(crossing_move, True), spur_tail + (BUFFER,))
                    layout.add("Goods road", approach + (crossing_move,) + exit_path, "spur", main)
                    yield layout

    # An IBW can replace the first R310/30° piece exactly on its tight route.
    for code, route, exit_port in (("83363", "a-b", "c"), ("83364", "b-a", "c")):
        if not cfg.available(code):
            continue
        bend = stock.curve(310, 90)
        rest_bend = stock.curve(310, 60)
        if not bend or not rest_bend:
            continue
        for length in (332, 415, 498, 581):
            straight = stock.straight(length)
            if not straight:
                continue
            for tail_len in (249, 332, 415):
                tail = stock.straight(tail_len)
                if tail is None:
                    continue
                layout = Layout(stock.catalog, "single", "curved_turnout", "Curved freight junction",
                                "Set out wagons on the freight spur from the curve.")
                # Keep both possible pieces in the same clockwise coordinate
                # system. The reversed left-hand point sits at the end of the cap.
                cap = (Step(code, route),) + rest_bend + bend if code == "83363" else bend + rest_bend + (Step(code, route),)
                top = layout.lay(Pose(), straight)
                others = layout.lay(layout.port(top[-1], True), cap + straight + bend + bend)
                main = layout.add("Main line", top + others, "loop")
                turnout = others[0] if code == "83363" else others[len(cap)-1]
                spur(layout, turnout, tail, "Goods road", main, exit=exit_port)
                yield layout

    for code in ("83380", "83382"):
        if not cfg.available(code):
            continue
        angle = stock.catalog[code].routes["a-c"].segments[0].sweep
        for radius in sorted(stock.curves):
            if radius == 267:
                continue
            bend = stock.curve(radius, 180 - angle) if code == "83380" else stock.curve(radius, 180)
            if bend is None:
                continue
            for length in (249, 290.5, 332, 415, 498):
                straight = stock.straight(length if code == "83380" else 83)
                tail = stock.straight(249 if code == "83380" else 290.5)
                gap = stock.straight(83)
                runway = stock.straight(length)
                if not straight or None in (tail, gap, runway):
                    continue
                points = (Step(code, "a-c"),)
                if code == "83382":
                    points += gap + (Step(code, "a-b"),) + runway
                half = straight + points + bend
                layout = Layout(stock.catalog, "single", f"y_turnout_{code}", "Y turnout freight junction",
                                "Use the paired freight spurs for opposite-direction local trains.")
                main_path = layout.lay(Pose(), half + half)
                main = layout.add("Main line", main_path, "loop")
                for index, offset in enumerate((0, len(half))):
                    spur(layout, main_path[offset+len(straight)], tail, f"Freight spur {index+1}", main, exit="b")
                    if code == "83382":
                        spur(layout, main_path[offset+len(straight)+len(gap)+1], tail,
                             f"Loading spur {index+1}", main, exit="c")
                yield layout


def score(layout, strategy):
    usable = sum(layout.lengths[t.name] for t in layout.tracks if t.kind == "spur")
    siding = sum(t.kind == "spur" for t in layout.tracks)
    pieces = len(layout.pieces)
    if strategy == "longest":
        return (layout.lap, usable, -pieces)
    if strategy == "shunting":
        return (usable, siding, layout.lap, -pieces)
    if strategy == "fewest_parts":
        return (-pieces, usable, layout.lap)
    return (siding, round(usable / 25), round(layout.lap / 200), -pieces, layout.lap)


def _estimate(layout, cfg):
    lengths = [layout.length_for_track(t) for t in layout.tracks if t.kind == "loop"]
    layout.lengths["lap"] = max(lengths)
    for track in layout.tracks:
        if track.kind == "spur":
            layout.lengths[track.name] = max(0, layout.length_for_track(track) - 150)
    return score(layout, cfg.strategy)


def _family_preference(layout):
    # In a two-plan freight set, show a yard ladder alongside the best
    # junction. For station plans, give each footprint a representative.
    return {"ladder": 3, "passing_freight_in": 3, "passing_rectangle": 2,
            "passing_oval": 1}.get(layout.family, 0)


def generate(catalog, cfg):
    """Return ranked validated layouts and rejection counts."""
    stock = Stock(catalog, cfg)
    rng = Random(cfg.seed)
    rejected = Counter()
    running_radii = [r for r in stock.curves if r >= 310 and stock.curve(r, 90)]
    if not running_radii:
        return [], Counter({"no enabled main-line curve can form a 90° bend for this vehicle group": 1})
    if min(cfg.width, cfg.height) + 1e-6 < 2 * (min(running_radii) + cfg.edge_margin_mm):
        return [], Counter({"board and edge margin cannot fit the minimum main-line diameter": 1})
    if cfg.mode == "single" and not cfg.available("83100"):
        return [], Counter({"a freight spur needs a buffer stop (83100)": 1})
    if cfg.mode in ("passing", "double") and cfg.train_length + 2 * (
        min(running_radii) + cfg.edge_margin_mm
    ) > max(cfg.width, cfg.height):
        return [], Counter({"configured train is too long for the available parallel straight": 1})
    mode_sources = {
        "single": (specialty_modes, ordinary_modes),
        "passing": (passing_modes,),
        "double": (double_modes,),
    }
    modes = MODES[:3] if cfg.mode == "mixed" else (cfg.mode,)
    candidates = []
    sectional = {code: limit for code, limit in cfg.parts.items() if catalog[code].kind in ("straight", "curve")}
    for mode in modes:
        logger.info("Searching for %s track layouts", mode)
        shelves = {}
        for source in mode_sources[mode]:
            for candidate in source(stock):
                inventory = candidate.inventory
                failure = next((f"inventory limit for {code}" for code, limit in cfg.parts.items()
                                if code not in sectional
                                if not limit.allows(inventory[code])), None)
                if failure:
                    rejected[failure] += 1
                    continue
                # Sort lightweight parametric candidates before invoking GEOS.
                # Preserve representatives for each distinct topology.
                estimate = _estimate(candidate, cfg)
                shelves.setdefault(candidate.family, []).append((estimate, rng.random(), candidate))
        logger.info("Checking %s layouts from a pool of %d candidates",
                    mode, sum(len(shelf) for shelf in shelves.values()))
        valid = []
        unique = {}
        for shelf in shelves.values():
            accepted = 0
            for _, tie, layout in sorted(shelf, key=lambda row: row[:2], reverse=True):
                if sectional:
                    layout = repartition(layout, stock, layout.inventory)
                layout.lengths.clear()
                failure = validate(layout, cfg)
                if failure:
                    rejected[failure] += 1
                    continue
                tracks = layout.centered_tracks()
                # Quantized length acts only as an index. Search neighbouring
                # bins before doing the geometric comparison.
                key = int(layout.lap)
                nearby = [row for length in range(key-1, key+2) for row in unique.get(length, ())]
                rank = score(layout, cfg.strategy)
                duplicate = next((row for row in nearby if equivalent(tracks, row[3])), None)
                if duplicate and duplicate[0] >= rank:
                    continue
                if duplicate:
                    unique[int(duplicate[2].lap)].remove(duplicate)
                else:
                    accepted += 1
                unique.setdefault(key, []).append((rank, tie, layout, tracks))
                if cfg.strategy == "balanced" and accepted >= max(cfg.count, 4):
                    break
        valid.extend(sorted((row[:3] for rows in unique.values() for row in rows),
                            key=lambda row: row[:2], reverse=True))
        if cfg.strategy == "balanced":
            # Take one of each operation family before repeating one.
            families = {}
            for _, _, layout in sorted(valid, key=lambda row: _family_preference(row[2]), reverse=True):
                families.setdefault(layout.family, []).append(layout)
            valid = []
            while any(families.values()):
                valid.extend(items.pop(0) for items in families.values() if items)
        else:
            valid = [layout for _, _, layout in valid]
        logger.info("Found %d distinct valid %s layouts", len(valid), mode)
        candidates.append(valid)
    if cfg.mode != "mixed":
        return candidates[0][:cfg.count], rejected
    if cfg.strategy != "balanced":
        all_layouts = [layout for mode in candidates for layout in mode]
        return sorted(all_layouts, key=lambda v: score(v, cfg.strategy), reverse=True)[:cfg.count], rejected
    # The default ten gives two switching designs and four each for two-train
    # operation. For other counts, cycle through the same proportions.
    schedule = (0, 1, 2, 1, 2)
    result = []
    while len(result) < cfg.count and any(candidates):
        for index in schedule:
            if candidates[index] and len(result) < cfg.count:
                result.append(candidates[index].pop(0))
    return result, rejected
