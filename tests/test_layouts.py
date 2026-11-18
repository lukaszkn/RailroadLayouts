import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tt_layouts.config import Config, Limit, read_config
from tt_layouts.export import write_run
from tt_layouts.generate import Stock, _filter_similar_lengths, generate, running_loop, score, specialty_modes, spur
from tt_layouts.geometry import Part, Pose, Route, Segment, Step, joint_fits, load_catalog
from tt_layouts.model import Layout, equivalent, turnout_count, validate

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT / "tillig_tt.json")
        cls.config = Config(parts={"83100": Limit(max_count=5), "83332": Limit(enabled=False)})
        cls.examples, cls.rejected = generate(cls.catalog, cls.config)

    def config_for(self, **changes):
        if "parts" in changes:
            changes["parts"] = {**self.config.parts, **changes["parts"]}
        return replace(self.config, **changes)

    def test_catalog_dimensions_and_movements(self):
        c = self.catalog
        self.assertAlmostEqual(c["83112"].routes["a-b"].length, 103.672557568)
        self.assertEqual(c["83100"].stop.x, 41.5)
        self.assertEqual(c["83331"].ports["c"].heading, 15)
        self.assertEqual(c["83331"].data["dimensions"]["frog_angle_deg"], 12)
        self.assertEqual(c["83332"].ports["c"].heading, 345)
        self.assertEqual(c["83341"].ports["c"].heading, 12)
        self.assertEqual(c["83212"].ports["d"].xy, (251, 43))
        self.assertEqual(set(c["83161"].routes), {"a-b", "c-d"})
        self.assertEqual(set(c["83300"].routes), {"a-b", "c-d", "a-d", "c-b"})
        self.assertFalse(c["83391"].routes)
        self.assertFalse(joint_fits(Pose(), Pose(-0.01, 0, 180)))
        self.assertFalse(joint_fits(Pose(), Pose(0.3, 0.151, 180)))
        self.assertFalse(joint_fits(Pose(), Pose(0.3, 0.1, 180.11)))
        self.assertTrue(joint_fits(Pose(), Pose(0.5, 0.15, 180.1)))

    def test_total_track_length_counts_physical_pointwork(self):
        crossover = self.catalog["83212"]
        self.assertAlmostEqual(crossover.track_length_mm, sum(route.length for route in crossover.routes.values()))
        self.assertGreater(crossover.track_length_mm, 2 * 251)
        slip = self.catalog["83300"]
        shared_length = 4 * slip.routes["a-d"].segments[0].distance
        self.assertAlmostEqual(slip.track_length_mm,
                               sum(route.length for route in slip.routes.values()) - shared_length)

        # The shorter route traverses the first half of the arc in reverse.
        arc = Segment(radius=310, sweep=60)
        start = Pose(0, 0, 65)
        halfway = arc.advance(start, .5)
        overlapping = Part("arc", {}, {"a": start.turned(180), "c": halfway}, {
            "a-b": Route("a", "b", (arc,)),
            "c-a": Route("c", "a", (Segment(radius=310, sweep=-30),)),
        }, None)
        self.assertAlmostEqual(overlapping.track_length_mm, arc.distance)

        layout = Layout(self.catalog, "double", "", "", "")
        crossing, = layout.lay(Pose(), (Step("83212"),))
        layout.add("Outer main", (crossing,), "loop")
        layout.add("Inner main", (layout.branch(crossing, "c", "d"),), "loop")
        layout.lay(Pose(), (Step("83100", "a-stop"),))
        self.assertAlmostEqual(layout.total_track_length_mm, crossover.track_length_mm + 41.5)

    def test_longest_tracks_prefers_more_track_with_a_shorter_lap(self):
        stock = Stock(self.catalog, self.config)

        def plan(span, tail):
            layout = Layout(self.catalog, "single", "", "", "")
            bend = stock.curve(353, 90)
            top, rest = running_loop(layout, Pose(), (Step("83331"),) + stock.straight(span - 166),
                                     stock.straight(span), bend)
            main = layout.add("Main line", top + rest, "loop")
            spur(layout, top[0], stock.straight(tail), "Goods road", main)
            self.assertIsNone(validate(layout, self.config))
            return layout

        longer_lap = plan(498, 166)
        longer_track = plan(415, 415)
        self.assertGreater(score(longer_lap, "longest"), score(longer_track, "longest"))
        self.assertGreater(score(longer_track, "longest_tracks"), score(longer_lap, "longest_tracks"))

    def test_longest_tracks_searches_multiple_interior_roads(self):
        limits = {code: Limit(enabled=False) for code, part in self.catalog.items()
                  if (part.kind == "curve" and part.radius not in (310, 353))
                  or (part.kind not in ("straight", "curve", "buffer", "reference")
                      and code not in ("83329", "83331", "83212"))}
        limits["83100"] = Limit(max_count=3)
        for mode, turnouts in (("single", 3), ("passing", 4), ("double", 6)):
            with self.subTest(mode=mode):
                cfg = self.config_for(width_cm=150, height_cm=90, count=3, mode=mode,
                                      strategy="longest_tracks", min_turnouts=turnouts,
                                      max_turnouts=turnouts, parts=limits)
                plans, _ = generate(self.catalog, cfg)
                self.assertEqual(len(plans), 3)
                lengths = [plan.total_track_length_mm for plan in plans]
                self.assertEqual(lengths, sorted(lengths, reverse=True))
                for plan in plans:
                    self.assertIn("Interior road 2", plan.lengths)
                    self.assertLessEqual(plan.inventory["83100"], 3)
                    drive_counts = (quantity * self.catalog[code].data.get("point_drive", {}).get("quantity", 0)
                                    for code, quantity in plan.inventory.items())
                    self.assertEqual(sum(drive_counts), turnouts)

    def test_longest_tracks_nests_curved_roads_inside_connected_double_loops(self):
        parts = {code: Limit(enabled=False) for code, part in self.catalog.items()
                 if part.kind == "turnout" and code not in ("83328", "83341")}
        parts.update({"83212": Limit(enabled=False), "83100": Limit(max_count=5)})
        cfg = self.config_for(mode="double", count=1, strategy="longest_tracks",
                              min_turnouts=3, max_turnouts=7, parts=parts)
        plans, _ = generate(self.catalog, cfg)
        layout, = plans
        self.assertEqual(layout.inventory["83212"], 0)
        self.assertEqual(turnout_count(self.catalog, layout.inventory), 7)
        self.assertEqual(layout.inventory["83100"], 5)
        self.assertEqual(sum(track.kind == "loop" for track in layout.tracks), 2)
        self.assertGreater(layout.lengths["Crossover"], 0)
        self.assertGreater(layout.total_track_length_mm, 16_000)
        curved_roads = 0
        for track in layout.tracks:
            if track.kind != "spur":
                continue
            sweep = sum(abs(segment.sweep) for move in track.moves
                        if layout.pieces[move.piece].part.kind == "curve"
                        for segment in layout.route(move).segments)
            curved_roads += sweep >= 180
        self.assertGreaterEqual(curved_roads, 2)

    def test_require_connected_tracks_config(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            for settings, expected in (({}, True), ({"require_connected_tracks": True}, True),
                                       ({"require_connected_tracks": False}, False)):
                with self.subTest(settings=settings):
                    path.write_text(json.dumps(settings), encoding="utf-8")
                    self.assertIs(read_config(path, self.catalog).require_connected_tracks, expected)
            for value in (None, 0, 1, "true", "false", [], {}):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"require_connected_tracks": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "require_connected_tracks must be a boolean"):
                        read_config(path, self.catalog)

    def test_require_forward_loop_return_config(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            path.write_text("{}", encoding="utf-8")
            self.assertIs(read_config(path, self.catalog).require_forward_loop_return, False)
            for value in (True, False, None, 0, 1, "true"):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"require_forward_loop_return": value}), encoding="utf-8")
                    if type(value) is bool:
                        self.assertIs(read_config(path, self.catalog).require_forward_loop_return, value)
                    else:
                        with self.assertRaisesRegex(ValueError, "require_forward_loop_return must be a boolean"):
                            read_config(path, self.catalog)

    def test_ignore_similar_percentage_config(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(read_config(path, self.catalog).ignore_similar_percentage, 0)
            for value in (0, 0.05, 1):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"ignore_similar_percentage": value}), encoding="utf-8")
                    self.assertEqual(read_config(path, self.catalog).ignore_similar_percentage, value)
            for value in (None, True, "0.05", -0.01, 1.01, float("nan"), float("inf")):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"ignore_similar_percentage": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "ignore_similar_percentage must be a number between 0 and 1"):
                        read_config(path, self.catalog)

    def test_similarity_uses_every_named_track_of_selected_layouts(self):
        def plan(lengths):
            layout = Layout(self.catalog, "double", "", "", "")
            layout.lengths.update(lengths)
            return layout

        first = plan({"lap": 100, "inner_lap": 80, "Goods road": 20})
        without_goods_road = plan({"lap": 100, "inner_lap": 80})
        within_tolerance = plan({"lap": 105, "inner_lap": 76, "Goods road": 21})
        longer_goods_road = plan({"lap": 105, "inner_lap": 76, "Goods road": 21.1})
        cfg = self.config_for(count=3, ignore_similar_percentage=0.05)
        rejected = Counter()
        selected = _filter_similar_lengths(
            [first, without_goods_road, within_tolerance, longer_goods_road], cfg, rejected)

        self.assertEqual(selected, [first, without_goods_road, longer_goods_road])
        self.assertEqual(rejected["similar track lengths"], 1)

    def test_min_turnouts_config(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            for settings, expected in (({}, 1), ({"min_turnouts": 0}, 0), ({"min_turnouts": 5}, 5)):
                with self.subTest(settings=settings):
                    path.write_text(json.dumps(settings), encoding="utf-8")
                    self.assertEqual(read_config(path, self.catalog).min_turnouts, expected)
            for value in (None, True, -1, 1.5, "5"):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"min_turnouts": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "min_turnouts must be a non-negative integer"):
                        read_config(path, self.catalog)

    def test_max_turnouts_config(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(read_config(path, self.catalog).max_turnouts, 10)
            for value in (None, True, -1, 1.5, "5"):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"max_turnouts": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "max_turnouts must be a non-negative integer"):
                        read_config(path, self.catalog)
            path.write_text('{"min_turnouts": 5, "max_turnouts": 4}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "max_turnouts must be at least min_turnouts"):
                read_config(path, self.catalog)
            path.write_text('{"min_turnouts": 0, "max_turnouts": 0}', encoding="utf-8")
            self.assertEqual(read_config(path, self.catalog).max_turnouts, 0)

    def test_min_turnouts_requires_goods_road_with_double_crossover(self):
        cfg = self.config_for(mode="double", count=2, min_turnouts=5)
        plans, rejected = generate(self.catalog, cfg)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.inventory["83212"], 1)
            self.assertEqual(plan.inventory["83331"], 1)
        self.assertGreater(rejected["minimum turnout count of 5"], 0)

    def test_max_turnouts_allows_double_crossover_without_goods_road(self):
        cfg = self.config_for(mode="double", count=2, min_turnouts=4, max_turnouts=4)
        plans, rejected = generate(self.catalog, cfg)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.inventory["83212"], 1)
            self.assertEqual(plan.inventory["83331"], 0)
            self.assertIsNone(validate(plan, replace(cfg, require_forward_loop_return=True)))
        self.assertGreater(rejected["maximum turnout count of 4"], 0)

    def test_forward_loop_return_requires_crossovers_in_both_orientations(self):
        stock = Stock(self.catalog, self.config)
        outer_run = (Step("83341"),) + stock.straight(617)
        inner_prefix = stock.straight(202.5)
        inner_run = inner_prefix + (Step("83341", "b-a"),) + stock.straight(414.5)
        plain = stock.straight(824)
        for count in (1, 2):
            with self.subTest(crossovers=count):
                layout = Layout(self.catalog, "double", "", "", "")
                layout.parallel_mm = 824
                crossover_points = []
                for start, run, radius in ((Pose(), outer_run, 353), (Pose(0, 43), inner_run, 310)):
                    bend = stock.curve(radius, 90)
                    top, rest = running_loop(layout, start, run, plain if count == 1 else run, bend)
                    layout.add(str(radius), top + rest, "loop")
                    crossover_points.append((top, rest[2 * len(bend):]))
                outer, inner = crossover_points
                for i in range(count):
                    facing = layout.branch(outer[i][0])
                    trailing = layout.branch(inner[i][len(inner_prefix)]).reversed()
                    layout.add(f"Crossover {i + 1}", (facing, trailing), "crossover")
                self.assertIsNone(validate(layout, self.config))
                self.assertEqual(validate(layout, replace(self.config, require_forward_loop_return=True)),
                                 "no forward-only return between running loops")

    def test_generate_opposite_crossovers_when_short_side_cannot_fit(self):
        limits = {code: Limit(enabled=False) for code, part in self.catalog.items()
                  if part.kind == "turnout" and code not in ("83341", "83342")}
        limits["83212"] = Limit(enabled=False)
        cfg = self.config_for(mode="double", height_cm=90, count=1, max_turnouts=4,
                              require_forward_loop_return=True, parts=limits)
        plans, _ = generate(self.catalog, cfg)
        plan, = plans
        self.assertEqual(plan.inventory["83341"], 2)
        self.assertEqual(plan.inventory["83342"], 2)
        diagonals = [track for track in plan.tracks if track.kind == "crossover"]
        self.assertEqual(len(diagonals), 2)
        headings = [plan.port(track.moves[0]).heading for track in diagonals]
        self.assertAlmostEqual((headings[0] - headings[1]) % 360, 180)
        self.assertIsNone(validate(plan, cfg))

    def test_longest_tracks_finds_forward_return_with_folded_yard(self):
        limits = {code: Limit(enabled=False) for code, part in self.catalog.items()
                  if part.kind == "turnout" and code not in ("83328", "83341", "83342")}
        limits["83212"] = Limit(enabled=False)
        cfg = self.config_for(mode="double", strategy="longest_tracks", count=1,
                              min_turnouts=9, max_turnouts=9, require_forward_loop_return=True, parts=limits)
        plans, _ = generate(self.catalog, cfg)
        plan, = plans
        self.assertEqual(plan.inventory["83100"], 5)
        self.assertGreater(plan.total_track_length_mm, 17_000)
        diagonals = [track for track in plan.tracks if track.kind == "crossover"]
        self.assertEqual(len(diagonals), 2)
        headings = [plan.port(track.moves[0]).heading for track in diagonals]
        self.assertAlmostEqual((headings[0] - headings[1]) % 180, 90)
        self.assertIsNone(validate(plan, cfg))

    def test_station_config_limits(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            path.write_text("{}", encoding="utf-8")
            cfg = read_config(path, self.catalog)
            self.assertFalse(cfg.require_station)
            self.assertFalse(cfg.require_interior)
            self.assertEqual(cfg.station_track_length, cfg.train_length)
            invalid = (
                ({"min_station_tracks": True}, "min_station_tracks"),
                ({"min_station_tracks": 4, "max_station_tracks": 3}, "max_station_tracks"),
                ({"min_station_track_length_cm": 0}, "min_station_track_length_cm"),
                ({"mode": "passing", "min_station_tracks": 3}, "Station track settings"),
                ({"min_interior_sidings": -1}, "min_interior_sidings"),
                ({"interior_style": "random"}, "interior_style"),
            )
            for settings, field in invalid:
                with self.subTest(settings=settings):
                    path.write_text(json.dumps(settings), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        read_config(path, self.catalog)

    def test_four_track_station_with_curved_yard_and_rotated_board(self):
        cfg = self.station_config(width_cm=120, height_cm=160, min_station_tracks=4)
        plans, rejected = generate(self.catalog, cfg)
        plan, = plans
        station = [plan.tracks[i] for i in plan.station_tracks]
        self.assertEqual([track.kind for track in station], ["loop", "loop", "station", "station"])
        for track in station[2:]:
            parent = {move.piece for move in plan.tracks[track.parent].moves}
            self.assertTrue({track.moves[0].piece, track.moves[-1].piece} <= parent)
        self.assertGreaterEqual(plan.lengths["station_usable"], 250)
        self.assertEqual(len(plan.interior_tracks), 2)
        self.assertEqual(turnout_count(self.catalog, plan.inventory), 10)
        self.assertEqual(plan.inventory["83212"], 0)
        self.assertIsNone(validate(plan, cfg))

        with TemporaryDirectory() as root:
            path = Path(root) / "station"
            write_run(plans, rejected, cfg, path)
            report = json.loads((path / "run.json").read_text())
            placement = json.loads((path / "01-double" / "placement.json").read_text())
            for row in (report["layouts"][0], placement):
                self.assertEqual(row["station_track_count"], 4)
                self.assertEqual(row["station_usable_length_mm"], round(plan.lengths["station_usable"], 2))
                self.assertEqual(row["interior_siding_count"], 2)
        tighter = replace(cfg, min_station_track_length_cm=plan.lengths["station_usable"] / 10 + 1)
        self.assertEqual(validate(plan, tighter),
                         "configured train or station length exceeds the shared clear station straight")

    def test_three_track_station_respects_yard_budget_with_balanced_ranking(self):
        cfg = self.station_config(mode="mixed", strategy="balanced", min_station_tracks=3,
                                  max_station_tracks=3, max_turnouts=8)
        plans, _ = generate(self.catalog, cfg)
        plan, = plans
        self.assertEqual(plan.mode, "double")
        self.assertEqual(len(plan.station_tracks), 3)
        self.assertEqual(len(plan.interior_tracks), 2)
        self.assertEqual(turnout_count(self.catalog, plan.inventory), 8)
        plans, rejected = generate(self.catalog, replace(cfg, max_turnouts=7))
        self.assertEqual(plans, [])
        self.assertTrue(rejected)

    def station_config(self, **changes):
        return replace(self.config_for(
            mode="double", strategy="longest_tracks", count=1, edge_margin_mm=55,
            require_forward_loop_return=True, min_turnouts=3, max_turnouts=10,
            min_station_tracks=3, max_station_tracks=4, min_interior_sidings=2,
            interior_style="curved_yard",
            parts={"83212": Limit(enabled=False), "83300": Limit(enabled=False),
                   "83332": Limit(enabled=True)},
        ), **changes)

    def test_mode_mix_and_inventory(self):
        self.assertTrue(self.config.require_connected_tracks)
        self.assertEqual(Counter(p.mode for p in self.examples),
                         {"single": 2, "passing": 4, "double": 4})
        for plan in self.examples:
            if plan.mode == "double":
                self.assertGreater(plan.inventory["83212"], 0)
            self.assertEqual(plan.inventory["83332"], 0)
            self.assertLessEqual(plan.inventory["83100"], 5)
            self.assertTrue(all(g[0] <= .500001 and abs(g[1]) <= .150001 and abs(g[2]) <= .100001
                                for _, _, g in plan.joints))
            left, top, right, bottom = plan.bounds()
            self.assertGreaterEqual(min(left, top, self.config.width-right, self.config.height-bottom),
                                    self.config.edge_margin_mm - .001)
        for i, plan in enumerate(self.examples):
            for other in self.examples[i+1:]:
                self.assertFalse(equivalent(plan.centered_tracks(), other.centered_tracks()))

    def test_each_specialty_is_usable(self):
        stock = Stock(self.catalog, self.config)
        required = {"83161", "83170", "83300", "83363", "83364", "83380", "83382"}
        tested = set()
        for candidate in specialty_modes(stock):
            wanted = required.intersection(candidate.inventory) - tested
            if wanted and validate(candidate, self.config) is None:
                tested.update(wanted)
        self.assertEqual(tested, required)
        for code in ("83230", "83341", "83342"):
            with self.subTest(code=code):
                cfg = self.config_for(mode="single", count=1, parts={code: Limit(min_count=1)})
                plans, _ = generate(self.catalog, cfg)
                self.assertEqual(len(plans), 1)
                self.assertGreater(plans[0].inventory[code], 0)

    def test_stock_passing_with_ew2_right_and_ew1_left(self):
        cfg = self.config_for(mode="passing", count=2, parts={
            "83331": Limit(min_count=1), "83329": Limit(min_count=1), "83100": Limit(max_count=0),
        })
        plans, _ = generate(self.catalog, cfg)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertGreaterEqual(plan.lengths["station_main"], 250)
            self.assertGreaterEqual(plan.lengths["Passing track"], 250)
            self.assertEqual(plan.inventory["83100"], 0)
            self.assertEqual(plan.inventory["83332"], 0)
            self.assertGreater(plan.inventory["83331"], 0)

    def test_double_crossover_with_sectional_inventory_caps(self):
        cfg = self.config_for(mode="double", count=2, parts={
            "83212": Limit(min_count=1), "83101": Limit(max_count=1),
            "83106": Limit(max_count=4), "83112": Limit(min_count=3),
        })
        plans, _ = generate(self.catalog, cfg)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.inventory["83212"], 1)
            self.assertLessEqual(plan.inventory["83101"], 1)
            self.assertLessEqual(plan.inventory["83106"], 4)
            self.assertGreaterEqual(plan.inventory["83112"], 3)
            self.assertGreaterEqual(plan.parallel_mm, 250)

    def test_double_without_crossover_requires_opt_out(self):
        cfg = self.config_for(mode="double", count=1, min_turnouts=0, parts={
            "83212": Limit(enabled=False), "83331": Limit(enabled=False),
        })
        reason = "track network is disconnected"
        plans, rejected = generate(self.catalog, cfg)
        self.assertEqual(plans, [])
        self.assertGreater(rejected[reason], 0)
        with TemporaryDirectory() as root:
            path = Path(root) / "plans"
            write_run(plans, rejected, cfg, path)
            report = json.loads((path / "run.json").read_text())
            self.assertIs(report["config"]["require_connected_tracks"], True)
            self.assertEqual(report["shortfall"], 1)
            self.assertIn(reason, report["rejections"])

        plans, _ = generate(self.catalog, replace(cfg, require_connected_tracks=False))
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].inventory["83212"], 0)
        self.assertEqual(validate(plans[0], cfg), reason)
        self.assertEqual(validate(plans[0], replace(cfg, require_connected_tracks=False, min_turnouts=1)),
                         "minimum turnout count of 1")
        forward = replace(cfg, require_connected_tracks=False, require_forward_loop_return=True,
                          max_turnouts=3)
        self.assertEqual(validate(plans[0], forward), "no forward-only return between running loops")
        plans, rejected = generate(self.catalog, forward)
        self.assertEqual(plans, [])
        self.assertGreater(rejected["no forward-only return between running loops"], 0)

    def test_fixed_diamond_does_not_connect_its_routes(self):
        layout = Layout(self.catalog, "single", "diamond", "Fixed crossing", "")
        crossing, = layout.lay(Pose(), (Step("83161"),))
        buffer = (Step("83100", "a-stop"),)
        for move in (crossing, layout.branch(crossing, "c", "d")):
            before, = layout.lay(layout.port(move), buffer)
            after, = layout.lay(layout.port(move, True), buffer)
            layout.add(move.entry, (before.reversed(), move, after), "spur")
        self.assertEqual(validate(layout, self.config), "minimum turnout count of 1")
        self.assertEqual(validate(layout, self.config_for(min_turnouts=0)), "track network is disconnected")

    def test_smaller_board_long_train_and_no_stock(self):
        for changes in (
            {"width_cm": 45, "height_cm": 45},
            {"mode": "passing", "train_length_cm": 90},
            {"mode": "single", "parts": {"83100": Limit(enabled=False)}},
            {"mode": "passing", "parts": {code: Limit(enabled=False)
              for code,p in self.catalog.items() if p.kind == "curve"}},
        ):
            with self.subTest(changes=changes):
                plans, _ = generate(self.catalog, self.config_for(count=1, **changes))
                self.assertEqual(plans, [])

    def test_rotated_resegmented_routes_compare_equal(self):
        def plan(start, steps):
            layout = Layout(self.catalog, "", "", "", "")
            layout.add("road", layout.lay(start, steps), "spur")
            return layout.centered_tracks()

        self.assertTrue(equivalent(plan(Pose(), (Step("83101"),)),
                                   plan(Pose(25, 74, 90), (Step("83102"), Step("83102")))))
        self.assertTrue(equivalent(plan(Pose(), (Step("83111"),)),
                                   plan(Pose(35, 72, 180), (Step("83112"), Step("83112")))))

    def test_ranking_repeatability_and_order(self):
        for strategy in ("longest", "longest_tracks", "fewest_parts", "shunting"):
            with self.subTest(strategy=strategy):
                cfg = self.config_for(mode="double", width_cm=125, height_cm=90, count=3, strategy=strategy)
                first, _ = generate(self.catalog, cfg)
                second, _ = generate(self.catalog, cfg)
                self.assertEqual(len(first), 3)
                self.assertTrue(all(equivalent(a.centered_tracks(), b.centered_tracks())
                                    for a,b in zip(first,second)))
                self.assertEqual([p.inventory for p in first], [p.inventory for p in second])
                if strategy == "longest":
                    self.assertEqual([p.lap for p in first], sorted((p.lap for p in first), reverse=True))
                if strategy == "longest_tracks":
                    lengths = [p.total_track_length_mm for p in first]
                    self.assertEqual(lengths, sorted(lengths, reverse=True))
                if strategy == "fewest_parts":
                    self.assertEqual([len(p.pieces) for p in first], sorted(len(p.pieces) for p in first))

    def test_exported_examples_match_bom_and_jpeg(self):
        root = ROOT / "examples"
        report = json.loads((root / "run.json").read_text())
        self.assertEqual(report["generated"], 10)
        for row in report["layouts"]:
            directory = root / row["plan"]
            placement = json.loads((directory / "placement.json").read_text())
            counts = Counter(item["code"] for item in placement["parts"])
            with (directory / "shopping.csv").open(encoding="utf-8-sig") as f:
                bom = {r["tillig_code"]: int(r["quantity_pieces"]) for r in csv.DictReader(f)
                       if r["category"] == "track"}
            self.assertEqual(dict(counts), bom)
            with Image.open(directory / "layout.jpg") as image:
                self.assertEqual((image.format, image.size), ("JPEG", (3000, 2360)))
                image.verify()

    def test_export_collects_numbered_jpgs(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "plans"
            write_run(self.examples[:2], Counter(), self.config, path)
            drawings = path / "all"
            self.assertEqual(sorted(drawing.name for drawing in drawings.iterdir()),
                             ["001-single-layout.jpg", "002-passing-layout.jpg"])
            report = json.loads((path / "run.json").read_text())
            index = (path / "index.md").read_text()
            for layout, row in zip(self.examples[:2], report["layouts"], strict=True):
                plan = row["plan"]
                original = path / plan / "layout.jpg"
                collected = drawings / f"0{plan}-layout.jpg"
                self.assertEqual(collected.read_bytes(), original.read_bytes())
                placement = json.loads((path / plan / "placement.json").read_text())
                total = round(layout.total_track_length_mm, 2)
                self.assertEqual(row["total_track_length_mm"], total)
                self.assertEqual(placement["total_track_length_mm"], total)
                self.assertIn(f"Total track {total / 10:.1f} cm", index)

    def test_export_drawing_with_crowded_piece_labels(self):
        cfg = self.config_for(mode="double", count=1)
        layout = Layout(self.catalog, "double", "parallel", "Parallel straights", "")
        short_straights = (Step("83120"),) * 25
        for name, y in (("Outer main", 540), ("Inner main", 583)):
            layout.add(name, layout.lay(Pose(550, y), short_straights), "loop")

        with TemporaryDirectory() as root:
            path = Path(root) / "plans"
            write_run([layout], Counter(), cfg, path)
            with Image.open(path / "01-double" / "layout.jpg") as image:
                self.assertEqual((image.format, image.size), ("JPEG", (3000, 2360)))
                image.verify()

    def test_shortfall_report(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "partial"
            write_run([], Counter({"board size or edge margin": 1}), self.config, path)
            result = json.loads((path / "run.json").read_text())
            self.assertEqual(result["shortfall"], 10)
            self.assertIn("board size or edge margin", (path / "index.md").read_text())


if __name__ == "__main__":
    unittest.main()
