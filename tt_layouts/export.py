"""Render assembly drawings and write shopping and placement data."""

from collections import Counter
from dataclasses import asdict
import csv
import json
import logging
from math import ceil, cos, floor, radians, sin
from pathlib import Path
from shutil import copyfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

COLORS = {
    "main": "#193c4b",
    "inner": "#34798a",
    "passing": "#2d7160",
    "spur": "#935328",
    "point": "#86551c",
}
INK = "#182b33"
SUBTLE = "#4e646e"
PAPER = "#ffffff"
BOARD = "#f7faf9"
GRID = "#e2e9e7"


def _font(size):
    return ImageFont.load_default(size)


def _units(mm):
    return f"{mm/10:.1f} cm"


def _route_colors(layout):
    result = {}
    for track in layout.tracks:
        for move in track.moves:
            route = layout.route(move)
            color = track.kind
            if track.kind == "loop":
                color = "inner" if track.name == "Inner main" else "main"
            elif track.kind == "station":
                color = "passing"
            elif track.kind == "crossover":
                color = "point"
            result[(move.piece, f"{route.start}-{route.end}")] = COLORS[color]
    return result


def _label_positions(x, y, normal, bounds):
    left, top, right, bottom = bounds
    nx, ny = normal
    for distance in (20, 27, 37, 48, 62, 82, 110):
        for sign in (-1, 1):
            for shift in (0, -22, 22, -45, 45, -68, 68):
                tx = x + sign*nx*distance + ny*shift
                ty = y + sign*ny*distance - nx*shift
                if left <= tx <= right and top <= ty <= bottom:
                    yield tx, ty

    # If the nearby positions are blocked, sample the full drawing area.
    positions = ((tx, ty) for tx in range(ceil(left), floor(right) + 1, 12)
                 for ty in range(ceil(top), floor(bottom) + 1, 12))
    yield from sorted(positions, key=lambda point: (point[0] - x)**2 + (point[1] - y)**2)


def _drawing(layout, cfg, number, target):
    width, height = 3000, 2360
    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)
    draw.text((100, 68), layout.title, fill=INK, font=_font(72), stroke_width=1)
    draw.text((103, 160),
              f"PLAN {number:02d}   |   {layout.mode.upper()}   |   TILLIG TT MODELLGLEIS   |   {cfg.width_cm:g} × {cfg.height_cm:g} CM",
              fill=SUBTLE, font=_font(28))
    draw.line((100, 225, width-100, 225), fill=GRID, width=3)

    scale = min(2600 / cfg.width, 1580 / cfg.height)
    x0, y0 = (width - cfg.width*scale)/2, 290 + (1580 - cfg.height*scale)/2
    x1, y1 = x0 + cfg.width*scale, y0 + cfg.height*scale

    def px(x, y):
        return x0 + x*scale, y0 + y*scale

    draw.rectangle((x0, y0, x1, y1), fill=BOARD)
    grid = 100 if max(cfg.width, cfg.height) < 2400 else 200
    font_grid = _font(19)
    for x in range(grid, round(cfg.width), grid):
        a, _ = px(x, 0)
        draw.line((a, y0, a, y1), fill=GRID, width=2)
        draw.text((a, y0-16), str(x//10), font=font_grid, fill=SUBTLE, anchor="ms")
    for y in range(grid, round(cfg.height), grid):
        _, b = px(0, y)
        draw.line((x0, b, x1, b), fill=GRID, width=2)
        draw.text((x0-10, b), str(y//10), font=font_grid, fill=SUBTLE, anchor="rm")
    draw.rectangle((x0, y0, x1, y1), outline="#829c9d", width=3)
    draw.text((x0-10, y0-9), "0", font=font_grid, fill=SUBTLE, anchor="rs")

    colors = _route_colors(layout)
    tracks = []
    for i, piece in enumerate(layout.pieces):
        for route, line in piece.lines.items():
            coords = [px(x, y) for x, y in line.coords]
            color = colors.get((i, route), COLORS["point"])
            tracks.append((coords, color))
    for coords, color in tracks:
        draw.line(coords, fill="#d4dfdd", width=max(7, round(14*scale)), joint="curve")
    for coords, color in tracks:
        draw.line(coords, fill=color, width=max(4, round(5.5*scale)), joint="curve")

    for (a, port), _, _ in layout.joints:
        joint = layout.pieces[a].ports[port]
        x, y = px(*joint.xy)
        t = radians(joint.heading)
        dx, dy = -sin(t)*7, cos(t)*7
        draw.line((x-dx, y-dy, x+dx, y+dy), fill=PAPER, width=4)
    for piece in layout.pieces:
        if piece.stop is None:
            continue
        x, y = px(*piece.stop.xy)
        t = radians(piece.stop.heading)
        dx, dy = -sin(t)*11, cos(t)*11
        draw.line((x-dx, y-dy, x+dx, y+dy), fill=COLORS["spur"], width=5)

    # Prefer compact labels to either side of each piece. A numbered label with
    # the actual SKU stays available even when several short pieces meet.
    occupied = []
    rails = Image.new("1", image.size)
    rail_mask = ImageDraw.Draw(rails)
    for coords, _ in tracks:
        rail_mask.line(coords, fill=1, width=13, joint="curve")
    labels = []
    for i, piece in enumerate(layout.pieces):
        line = next(iter(piece.lines.values()))
        center = line.interpolate(0.5, normalized=True)
        left, right = line.interpolate(0.4, normalized=True), line.interpolate(0.6, normalized=True)
        tangent = (right.x-left.x, right.y-left.y)
        length = (tangent[0]**2 + tangent[1]**2)**.5
        normal = -tangent[1]/length, tangent[0]/length
        labels.append((i, piece, px(center.x, center.y), normal, line.length))
    font = _font(20 if len(labels) > 60 else 23)
    pending = []
    for i, piece, (x,y), normal, _ in sorted(labels, key=lambda item: item[-1]):
        label = f"{i+1:02d}  {piece.part.code}"
        w = draw.textlength(label, font=font)/2 + 5
        h = 13
        bounds = (90 + w, 255 + h, width - 90 - w, 1910 - h)
        for tx, ty in _label_positions(x, y, normal, bounds):
            box = (tx-w, ty-h, tx+w, ty+h)
            if any(box[0] < b[2]+3 and box[2] > b[0]-3 and box[1] < b[3]+3 and box[3] > b[1]-3 for b in occupied):
                continue
            if rails.crop(tuple(map(int, box))).getbbox():
                continue
            occupied.append(box)
            pending.append((label, (x,y), (tx,ty), box))
            break
        else:
            raise ValueError(f"Unable to place drawing label {label}")
    for _, p, q, _ in pending:
        draw.line((p, q), fill="#9caeae", width=2)
    for label, _, (x, y), box in pending:
        draw.rounded_rectangle(box, 5, fill=PAPER, outline="#d9e2e0", width=1)
        draw.text((x, y), label, fill=INK, font=font, anchor="mm")

    footer = 1955
    draw.line((100, footer, width-100, footer), fill=GRID, width=3)
    kinds = [("Main line", "main")]
    if layout.mode == "double":
        kinds.append(("Inner main", "inner"))
    if layout.mode == "passing":
        kinds.append(("Passing track", "passing"))
    if any(t.kind == "station" for t in layout.tracks):
        kinds.append(("Station track", "passing"))
    if any(t.kind == "crossover" for t in layout.tracks):
        kinds.append(("Crossover", "point"))
    if any(t.kind == "spur" for t in layout.tracks):
        kinds.append(("Freight / yard", "spur"))
    for index, (label, key) in enumerate(kinds):
        x = 103 + index * 325
        draw.line((x, footer+49, x+48, footer+49), fill=COLORS[key], width=9)
        draw.text((x+62, footer+49), label, anchor="lm", fill=INK, font=_font(27))
    draw.text((width-103, footer+49), "Labels show piece ID and Tillig SKU", anchor="rm", fill=SUBTLE, font=_font(25))
    bar = min(grid*2, cfg.width / 4)
    left = width - 105 - bar*scale
    draw.line((left, footer+114, width-105, footer+114), width=4, fill=INK)
    for x in (left, width-105):
        draw.line((x, footer+102, x, footer+126), width=4, fill=INK)
    draw.text((width-105, footer+136), f"{bar/10:g} cm", anchor="ra", fill=INK, font=_font(24))

    details = [f"Longest lap {_units(layout.lap)}"]
    if cfg.strategy == "longest_tracks":
        details.append(f"Total track {_units(round(layout.total_track_length_mm, 2))}")
    if layout.mode == "double":
        details.append(f"Parallel straight {_units(layout.parallel_mm)}")
    if "station_main" in layout.lengths:
        details.append(f"Main at station {_units(layout.lengths['station_main'])}")
    if layout.station_tracks:
        details.append(f"Shared station straight {_units(layout.lengths['station_usable'])}")
    details.extend(f"{t.name} {_units(layout.lengths[t.name])}" for t in layout.tracks if t.kind != "loop")
    detail_font = _font(27)
    detail_rows = [""]
    for detail in details:
        candidate = f"{detail_rows[-1]}   |   {detail}" if detail_rows[-1] else detail
        if draw.textlength(candidate, font=detail_font) > left - 150 and detail_rows[-1]:
            detail_rows.append(detail)
        else:
            detail_rows[-1] = candidate
    for index, row in enumerate(detail_rows):
        draw.text((103, footer+102+index*35), row, fill=INK, font=detail_font)
    instructions_y = footer + 167 + 25 * (len(detail_rows) - 1)
    for i, row in enumerate(textwrap.wrap(layout.instructions, 122)):
        draw.text((103, instructions_y+i*35), row, fill=SUBTLE, font=_font(29))
    draw.text((103, height-68), "Grid in centimetres   •   Board origin at upper left   •   Placement coordinates in the companion JSON",
              fill=SUBTLE, font=_font(24))
    image.save(target, format="JPEG", quality=94, subsampling=0, dpi=(300, 300))


def _point(pose):
    return dict(x_mm=round(pose.x, 4), y_mm=round(pose.y, 4),
                heading_deg=round(pose.heading % 360, 4) % 360)


def _write_json(target, data):
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _shopping(layout, target):
    extras = Counter()
    drives = {}
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("category", "tillig_code", "quantity_pieces", "name_en", "name_pl", "product_url"))
        for code, quantity in sorted(layout.inventory.items()):
            data = layout.catalog[code].data
            writer.writerow(("track", code, quantity, data["name_en"], data["name_pl"], data["product_url"]))
            if drive := data.get("point_drive"):
                extras[drive["code"]] += quantity * drive["quantity"]
                drives[drive["code"]] = drive["product_url"]
        for code, quantity in sorted(extras.items()):
            writer.writerow(("point_drive", code, quantity, "Under-baseboard turnout motor",
                             "Napęd podmakietowy rozjazdu", drives[code]))
    return extras


def write_run(layouts, rejected, cfg, directory: Path):
    # Build in a sibling directory and publish only when all exports succeed.
    from tempfile import TemporaryDirectory
    directory = directory.resolve()
    if directory.exists():
        raise ValueError(f"Output already exists at {directory}. Choose a new --output directory.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".tt-layouts-") as tmp:
        work = Path(tmp) / "plans"
        work.mkdir()
        _write_outputs(layouts, rejected, cfg, work)
        # Rename within the same filesystem. The enclosing temporary directory
        # also clears intermediate files on export errors.
        work.rename(directory)


def _write_outputs(layouts, rejected, cfg, directory):
    all_drawings = directory / "all"
    all_drawings.mkdir()
    rows = []
    for number, layout in enumerate(layouts, 1):
        logger.info("Exporting plan %d (%s)", number, layout.title)
        stem = f"{number:02d}-{layout.mode}"
        folder = directory / stem
        folder.mkdir()
        drawing = folder / "layout.jpg"
        _drawing(layout, cfg, number, drawing)
        copyfile(drawing, all_drawings / f"{number:03d}-{layout.mode}-layout.jpg")
        drives = _shopping(layout, folder/"shopping.csv")
        joints = [{"from": [a+1, pa], "to": [b+1, pb],
                   "longitudinal_mm": round(g[0], 6), "lateral_mm": round(g[1], 6),
                   "angular_deg": round(g[2], 6)} for (a, pa), (b, pb), g in layout.joints]
        largest = {key: max((abs(j[key]) for j in joints), default=0)
                   for key in ("longitudinal_mm", "lateral_mm", "angular_deg")}
        row = dict(plan=stem, title=layout.title, mode=layout.mode, operations=layout.instructions,
                   pieces=len(layout.pieces), longest_lap_mm=round(layout.lap, 2),
                   total_track_length_mm=round(layout.total_track_length_mm, 2),
                   track_lengths_mm={name:round(length, 2) for name, length in layout.lengths.items()},
                   parallel_straight_mm=round(layout.parallel_mm, 2), point_drives=dict(drives),
                   maximum_joint_residuals=largest)
        if layout.station_tracks:
            row["station_track_count"] = len(layout.station_tracks)
            row["station_usable_length_mm"] = round(layout.lengths["station_usable"], 2)
        if layout.interior_tracks:
            row["interior_siding_count"] = sum(
                layout.lengths[layout.tracks[i].name] > 0 for i in layout.interior_tracks)
        rows.append(row)
        _write_json(folder/"placement.json", dict(row,
                    board_mm=[cfg.width, cfg.height],
                    axes="Origin at upper left, X right, Y down, headings clockwise from X",
                    parts=[{"id":i+1, "code":p.part.code, "origin":_point(p.origin), "reflected_curve":p.mirror==-1,
                            "connectors":{name:_point(q) for name,q in p.ports.items()},
                            **({"buffer_stop":_point(p.stop)} if p.stop else {})}
                           for i,p in enumerate(layout.pieces)],
                    routes=[{"name":t.name, "kind":t.kind,
                             "moves":[[m.piece+1,m.entry,m.exit] for m in t.moves]}
                            for t in layout.tracks], joints=joints))
    _write_json(directory/"run.json", dict(config=asdict(cfg), generated=len(rows),
                shortfall=cfg.count-len(rows), rejections=dict(rejected), layouts=rows))
    lines = ["# Tillig TT track plans", "",
             f"{len(rows)} of {cfg.count} plans for a {cfg.width_cm:g} × {cfg.height_cm:g} cm board. "
             f"DCC, vehicle group {cfg.vehicle_group}, target train length {cfg.train_length_cm:g} cm.", ""]
    if len(rows) < cfg.count:
        reason = ", ".join(f"{k} ({v})" for k,v in rejected.most_common(8))
        if not reason:
            reason = "No formation matched the board, available stock, vehicle group, or train length."
        lines.extend(["## Shortfall", "", reason, ""])
    for row in rows:
        stem = row["plan"]
        drives = sum(row["point_drives"].values())
        lengths = row["track_lengths_mm"]
        usable = [f"{k.replace('_', ' ').capitalize()} {_units(v)}" for k,v in lengths.items()
                  if k not in ("lap", "inner_lap", "passing_lap")]
        if row["parallel_straight_mm"]:
            usable.append(f"Parallel straight {_units(row['parallel_straight_mm'])}")
        lines.extend([f"## {stem} · {row['title']}", "", row["operations"], "",
                      "   ·   ".join([f"Longest lap {_units(row['longest_lap_mm'])}",
                                   f"Total track {_units(row['total_track_length_mm'])}",
                                   f"{row['pieces']} track pieces", f"{drives} turnout drive{'s' if drives != 1 else ''}", *usable]), "",
                      f"[JPG]({stem}/layout.jpg) · [Shopping list]({stem}/shopping.csv) · [Placement data]({stem}/placement.json)", ""])
    (directory/"index.md").write_text("\n".join(lines), encoding="utf-8")
