"""Repartition existing straight and circular runs when a SKU limit requires it."""

from functools import cache
from itertools import groupby, product

from .geometry import Step
from .model import Layout


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


def repartition(layout, stock, initial_counts):
    """Preserve route endpoints and optimize piece count subject to SKU totals."""
    relevant = {code: limit for code, limit in stock.cfg.parts.items()
                if stock.catalog[code].kind in ("straight", "curve")
                and (limit.min_count or limit.max_count is not None or not limit.enabled)}
    if not relevant or all(limit.allows(initial_counts[code]) for code, limit in relevant.items()):
        return layout
    groups = []
    for track in layout.tracks:
        for kind, moves in groupby(track.moves, key=lambda move: _signature(layout, move)):
            moves = tuple(moves)
            if kind[0] == "pointwork":
                continue
            amount = _amount(layout, moves, kind)
            groups.append((kind, amount, moves))
    replacements = {}
    for family, entries in groupby(sorted(groups, key=lambda v: v[0][:2]),
                                    key=lambda v: v[0][:2]):
        entries = list(entries)
        # A curve's handedness does not change its SKU counts.
        options = _sizes(stock, family)
        constrained = tuple(code for _, code in options if code in relevant)
        if not constrained:
            continue
        free = tuple((size, code) for size, code in options if code not in relevant)
        bounded = tuple((size, code) for size, code in options if code in relevant)
        minima = tuple(relevant[code].min_count for code in constrained)
        maxima = tuple(relevant[code].max_count for code in constrained)

        def free_run(amount):
            # The unconstrained lengths are computed bottom up to keep long
            # straight runs independent of the interpreter's recursion limit.
            for n in range(len(free_runs), amount+1):
                free_runs.append(min((free_runs[n-size] + (code,) for size, code in free
                                      if n >= size and free_runs[n-size] is not None),
                                     key=lambda codes: (len(codes), codes), default=None))
            return free_runs[amount]

        @cache
        def choices(amount):
            result = []
            for counts in product(*(range(min(amount//size, maximum if maximum is not None else amount//size)+1)
                                    for (size, _), maximum in zip(bounded, maxima))):
                left = amount - sum(qty*size for qty, (size, _) in zip(counts, bounded))
                if left < 0 or (rest := free_run(left)) is None:
                    continue
                selected = tuple(code for qty, (_, code) in zip(counts, bounded) for _ in range(qty))
                result.append((counts, selected + rest))
            return sorted(result, key=lambda v: (len(v[1]), v[1]))

        @cache
        def assign(index, totals):
            if index == len(entries):
                return (0, ()) if all(v >= low for v, low in zip(totals, minima)) else None
            _, amount, _ = entries[index]
            best = None
            for counts, codes in choices(amount):
                used = tuple(a+b for a, b in zip(totals, counts))
                if any(maximum is not None and value > maximum for maximum, value in zip(maxima, used)):
                    continue
                # Uncapped inventory needs only to remember progress up to its minimum.
                state = tuple(value if cap is not None else min(value, low)
                              for value, cap, low in zip(used, maxima, minima))
                suffix = assign(index+1, state)
                if suffix is not None:
                    answer = len(codes) + suffix[0], (codes,) + suffix[1]
                    if best is None or answer < best:
                        best = answer
            return best

        free_runs = [()]
        selection = assign(0, (0,) * len(constrained))
        if selection is None:
            return layout
        for (kind, _, moves), codes in zip(entries, selection[1]):
            mirror = kind[2] if kind[0] == "curve" else 1
            replacements[moves[0]] = moves, tuple(Step(code, mirror=mirror) for code in codes)
    if not replacements:
        return layout
    return _rebuild(layout, replacements)


def _amount(layout, moves, kind):
    if kind[0] == "straight":
        return round(sum(layout.route(move).length for move in moves)*10)
    return round(sum(abs(layout.route(move).segments[0].sweep) for move in moves)*2)


def _sizes(stock, family):
    if family[0] == "straight":
        return stock.straights
    return stock.curves[family[1]]


def _rebuild(layout, replacements):
    updated = Layout(layout.catalog, layout.mode, layout.family, layout.title, layout.instructions)
    updated.parallel_mm = layout.parallel_mm
    placed = {}
    for track in layout.tracks:
        moves = []
        index = 0
        while index < len(track.moves):
            move = track.moves[index]
            if move in replacements:
                original, steps = replacements[move]
                moves.extend(updated.lay(layout.port(move).turned(180), steps))
                index += len(original)
                continue
            if move.piece not in placed:
                placed[move.piece] = len(updated.pieces)
                updated.pieces.append(layout.pieces[move.piece])
            moves.append(type(move)(placed[move.piece], move.entry, move.exit))
            index += 1
        updated.add(track.name, moves, track.kind, track.parent)
    return updated
