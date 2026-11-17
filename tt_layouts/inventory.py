"""Repartition straight and circular runs to lower cost or meet SKU limits."""

from collections import Counter, defaultdict
from dataclasses import replace
from functools import cache
from math import hypot

from .geometry import Step, angle_delta
from .model import Layout


def priced_families(stock):
    """Identify stock families with enough geometry and price data to compare."""
    prices = {}
    for family in [("straight",), *(("curve", radius) for radius in stock.curves)]:
        options = _sizes(stock, family)
        family_prices = {}
        for size, code in options:
            part = stock.catalog[code]
            price = part.data.get("unit_price_eur_cents")
            if price is None or part.ports.keys() != {"a", "b"} or part.routes.keys() != {"a-b"}:
                break
            route = part.routes["a-b"]
            if len(route.segments) != 1 or part.stop is not None:
                break
            segment = route.segments[0]
            if family == ("straight",):
                exact = not segment.radius and segment.length * 10 == size
            else:
                exact = segment.radius == family[1] and abs(segment.sweep) * 2 == size
            if not exact:
                break
            family_prices[code] = price
        else:
            if family_prices:
                prices[family] = family_prices
    return prices


def _signature(layout, move):
    piece = layout.pieces[move.piece]
    if piece.part.kind == "straight":
        return ("straight",)
    if piece.part.kind == "curve":
        route = layout.route(move)
        sweep = route.segments[0].sweep
        sign = piece.mirror if move.entry == route.start else -piece.mirror
        return "curve", piece.part.radius, 1 if sweep * sign > 0 else -1
    return ("pointwork", move.piece)


def _same_pose(first, second):
    # Use the catalog's connector-to-centerline tolerance.
    return (hypot(first.x - second.x, first.y - second.y) <= 1e-5
            and abs(angle_delta(first.heading, second.heading)) <= 1e-5)


def _continuous(layout, first, second):
    return _same_pose(layout.port(first, True), layout.port(second).turned(180))


def _runs(layout, priced):
    owners = Counter(move.piece for track in layout.tracks for move in track.moves) if priced else {}
    tracks = list(layout.tracks)
    families = defaultdict(list)
    for index, track in enumerate(tracks):
        runs = []
        for move in track.moves:
            kind = _signature(layout, move)
            prices = priced.get(kind[:2])
            if prices is not None and (owners[move.piece] > 1
                                       or layout.pieces[move.piece].part.code not in prices):
                kind = ("pointwork", move.piece)
            if (runs and runs[-1][0] == kind
                    and (kind[:2] not in priced or _continuous(layout, runs[-1][1][-1], move))):
                runs[-1][1].append(move)
            else:
                runs.append((kind, [move]))
        if track.kind == "loop" and len(runs) > 1:
            first_kind, first = runs[0]
            last_kind, last = runs[-1]
            if (first_kind == last_kind and first_kind[:2] in priced
                    and _continuous(layout, last[-1], first[0])):
                runs = [(last_kind, last + first), *runs[1:-1]]
                tracks[index] = replace(track, moves=tuple(move for _, run in runs for move in run))
        for kind, moves in runs:
            if kind[0] != "pointwork":
                families[kind[:2]].append((kind, _amount(layout, moves, kind), tuple(moves)))
    return families, tracks


def repartition(layout, stock, initial_counts, *, by_price=False):
    """Choose exact recipes across each family subject to plan-wide SKU totals."""
    relevant = {code: limit for code, limit in stock.cfg.parts.items()
                if stock.catalog[code].kind in ("straight", "curve")
                and (limit.min_count or limit.max_count is not None or not limit.enabled)}
    priced = stock.sectional_prices if by_price else {}
    repair = any(not limit.allows(initial_counts[code]) for code, limit in relevant.items())
    if not priced and not repair:
        return layout

    families, tracks = _runs(layout, priced)
    replacements = {}
    exact_runs = set()
    for family, entries in sorted(families.items()):
        prices = priced.get(family, {})
        options = _sizes(stock, family)
        bounded = tuple((size, code) for size, code in options if code in relevant)
        if not prices and not (repair and bounded):
            continue
        original_codes = [tuple(layout.pieces[move.piece].part.code for move in moves)
                          for _, _, moves in entries]
        # Shared pieces and other fixed parts count toward the same inventory limits.
        fixed = initial_counts - Counter(code for codes in original_codes for code in codes) if prices else {}
        minima = tuple(max(0, relevant[code].min_count - fixed.get(code, 0)) for _, code in bounded)
        maxima = tuple(None if relevant[code].max_count is None
                       else relevant[code].max_count - fixed.get(code, 0) for _, code in bounded)
        if any(cap is not None and cap < 0 for cap in maxima):
            return layout
        free = tuple((size, code) for size, code in options if code not in relevant)
        selection = _assign_runs([amount for _, amount, _ in entries], free, bounded,
                                 minima, maxima, prices, stock.sectional_recipes)
        if selection is None:
            return layout
        cost, count, recipes = selection
        if prices and all(relevant[code].allows(initial_counts[code]) for _, code in bounded):
            original_cost = sum(prices[code] for codes in original_codes for code in codes)
            if (cost, count) >= (original_cost, sum(map(len, original_codes))):
                continue
        for (kind, _, moves), old, codes in zip(entries, original_codes, recipes):
            if prices and Counter(old) == Counter(codes):
                continue
            mirror = kind[2] if kind[0] == "curve" else 1
            if prices:
                exact_runs.add(moves[0])
            if prices and kind[0] == "curve":
                steps = tuple(Step(code, mirror=mirror if stock.catalog[code].routes["a-b"].segments[0].sweep > 0
                                   else -mirror) for code in codes)
            else:
                steps = tuple(Step(code, mirror=mirror) for code in codes)
            replacements[moves[0]] = moves, steps
    if not replacements:
        return layout
    return rebuild(layout, replacements, tracks=tracks, exact_runs=exact_runs)


def _free_recipe(amount, options, prices, table):
    # Reuse the bottom-up table across runs and candidates with the same stock.
    for n in range(len(table), amount + 1):
        best = None
        for size, code in options:
            if n >= size and (rest := table[n-size]) is not None:
                cost, count, codes = rest
                recipe = cost + prices.get(code, 0), count + 1, codes + (code,)
                if best is None or recipe < best:
                    best = recipe
        table.append(best)
    return table[amount]


def _assign_runs(amounts, free, bounded, minima, maxima, prices, tables):
    if prices or not bounded:
        # The cheapest independent runs are also optimal if their totals fit.
        options = tuple(sorted(free + bounded))
        table = tables.setdefault((bool(prices), options), [(0, 0, ())])
        recipes = [_free_recipe(amount, options, prices, table) for amount in amounts]
        if any(recipe is None for recipe in recipes):
            return None
        codes = tuple(recipe[2] for recipe in recipes)
        counts = Counter(code for run in codes for code in run)
        if all(counts[code] >= low and (cap is None or counts[code] <= cap)
               for (_, code), low, cap in zip(bounded, minima, maxima)):
            return sum(r[0] for r in recipes), sum(r[1] for r in recipes), codes

    free_runs = tables.setdefault((bool(prices), tuple(sorted(free))), [(0, 0, ())])

    def allocations(remaining, index):
        if index == len(bounded):
            yield (), remaining
            return
        size, _ = bounded[index]
        cap = maxima[index]
        limit = remaining // size
        for qty in range((limit if cap is None else min(limit, cap)) + 1):
            for counts, left in allocations(remaining - qty * size, index + 1):
                yield (qty,) + counts, left

    @cache
    def choices(amount):
        result = []
        for counts, left in allocations(amount, 0):
            rest = _free_recipe(left, free, prices, free_runs)
            if rest is None:
                continue
            cost, count, codes = rest
            cost += sum(qty * prices.get(code, 0) for qty, (_, code) in zip(counts, bounded))
            selected = tuple(code for qty, (_, code) in zip(counts, bounded) for _ in range(qty))
            result.append((counts, cost, count + sum(counts), selected + codes))
        return result

    @cache
    def assign(index, totals):
        if index == len(amounts):
            return (0, 0, ()) if all(v >= low for v, low in zip(totals, minima)) else None
        best = None
        for counts, cost, count, codes in choices(amounts[index]):
            used = tuple(a+b for a, b in zip(totals, counts))
            if any(cap is not None and value > cap for cap, value in zip(maxima, used)):
                continue
            # Uncapped inventory needs only to remember progress up to its minimum.
            state = tuple(value if cap is not None else min(value, low)
                          for value, cap, low in zip(used, maxima, minima))
            suffix = assign(index + 1, state)
            if suffix is not None:
                candidate = cost + suffix[0], count + suffix[1], (codes,) + suffix[2]
                if best is None or candidate < best:
                    best = candidate
        return best

    return assign(0, (0,) * len(bounded))


def _amount(layout, moves, kind):
    if kind[0] == "straight":
        return round(sum(layout.route(move).length for move in moves)*10)
    return round(sum(abs(layout.route(move).segments[0].sweep) for move in moves)*2)


def _sizes(stock, family):
    if family[0] == "straight":
        return stock.straights
    return stock.curves[family[1]]


def rebuild(layout, replacements, *, tracks=None, exact_runs=()):
    """Replace runs in a new layout, preserving track indices and shared pieces."""
    updated = Layout(layout.catalog, layout.mode, layout.family, layout.title, layout.instructions)
    updated.parallel_mm = layout.parallel_mm
    updated.station_tracks = layout.station_tracks
    updated.interior_tracks = layout.interior_tracks
    placed = {}
    for track in layout.tracks if tracks is None else tracks:
        moves = []
        index = 0
        while index < len(track.moves):
            move = track.moves[index]
            if move in replacements:
                original, steps = replacements[move]
                laid = updated.lay(layout.port(move).turned(180), steps)
                if move in exact_runs and (
                    not _same_pose(layout.port(move), updated.port(laid[0]))
                    or not _same_pose(layout.port(original[-1], True), updated.port(laid[-1], True))
                ):
                    return layout
                moves.extend(laid)
                index += len(original)
                continue
            if move.piece not in placed:
                placed[move.piece] = len(updated.pieces)
                updated.pieces.append(layout.pieces[move.piece])
            moves.append(type(move)(placed[move.piece], move.entry, move.exit))
            index += 1
        updated.add(track.name, moves, track.kind, track.parent)
    return updated
