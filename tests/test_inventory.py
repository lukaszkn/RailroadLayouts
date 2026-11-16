"""Price and geometry regressions for sectional track substitutions."""

from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tt_layouts.config import Config, Limit, read_config
from tt_layouts.export import write_run
from tt_layouts.generate import Stock, backwards, generate
from tt_layouts.geometry import Pose, Step, angle_delta, load_catalog
from tt_layouts.inventory import repartition
from tt_layouts.model import Layout, equivalent, validate

ROOT = Path(__file__).resolve().parents[1]


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT / "tillig_tt.json")

    def run_layout(self, codes, *, catalog=None, mirror=1, reverse=False):
        layout = Layout(catalog or self.catalog, "single", "test", "Test", "")
        moves = layout.lay(Pose(30, 40, 85), [Step(code, mirror=mirror) for code in codes])
        layout.add("Track", backwards(moves) if reverse else moves, "spur")
        return layout

    def optimize(self, layout, cfg=Config()):
        return repartition(layout, Stock(layout.catalog, cfg), layout.inventory, by_price=True)

    def test_config_default_and_opt_out(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "layout.json"
            for data, expected in (({}, True), ({"optimize_track_parts": True}, True),
                                   ({"optimize_track_parts": False}, False)):
                path.write_text(json.dumps(data), encoding="utf-8")
                self.assertIs(read_config(path, self.catalog).optimize_track_parts, expected)
            for value in (None, 0, 1, "true", [], {}):
                with self.subTest(value=value):
                    path.write_text(json.dumps({"optimize_track_parts": value}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "optimize_track_parts must be a boolean"):
                        read_config(path, self.catalog)

    def test_exact_substitutions_preserve_length_and_end_poses(self):
        for original, expected in (
            (["83103"] * 2, ["83102"]),
            (["83102"] * 2, ["83101"]),
            (["83104"] * 8, ["83101", "83102", "83105"]),
            (["83105"] * 2 + ["83120"] * 5, ["83102"] + ["83104"] * 3),
            (["83115"] * 4, ["83116"]),
            (["83113"] * 2, ["83110"]),
            (["83110"] * 2, ["83109"]),
            (["83114"] * 2, ["83107"]),
            (["83107"] * 2, ["83106"]),
            (["83112"] * 2, ["83111"]),
        ):
            with self.subTest(original=original):
                directions = [(1, False)]
                if self.catalog[original[0]].kind == "curve":
                    directions.extend(((1, True), (-1, False), (-1, True)))
                for mirror, reverse in directions:
                    layout = self.run_layout(original, mirror=mirror, reverse=reverse)
                    updated = self.optimize(layout, Config(vehicle_group="A"))
                    self.assertEqual(updated.inventory, Counter(expected))
                    self.assertAlmostEqual(updated.total_track_length_mm, layout.total_track_length_mm)
                    old_moves, new_moves = layout.tracks[0].moves, updated.tracks[0].moves
                    for old, new, end in ((old_moves[0], new_moves[0], False),
                                          (old_moves[-1], new_moves[-1], True)):
                        first, second = layout.port(old, end), updated.port(new, end)
                        self.assertAlmostEqual(first.x, second.x)
                        self.assertAlmostEqual(first.y, second.y)
                        self.assertAlmostEqual(angle_delta(first.heading, second.heading), 0)

    def test_cost_breaks_piece_count_tie(self):
        layout = self.run_layout(["83105"] * 9)
        updated = self.optimize(layout)
        self.assertEqual(len(updated.pieces), 9)
        self.assertEqual(updated.inventory["83101"], 1)
        self.assertEqual(updated.total_track_length_mm, layout.total_track_length_mm)

    def test_keeps_exact_runs_with_equal_cost_and_count(self):
        layout = self.run_layout(["83102", "83101"])
        self.assertIs(self.optimize(layout), layout)

    def test_continuous_loop_can_merge_across_its_start(self):
        layout = Layout(self.catalog, "single", "test", "Test", "")
        bend = [Step("83109")] * 6
        steps = [Step("83102"), *bend, Step("83101"), *bend, Step("83102")]
        layout.add("Main line", layout.lay(Pose(), steps), "loop")
        updated = self.optimize(layout)
        self.assertEqual(updated.inventory, Counter({"83101": 2, "83109": 12}))
        self.assertTrue(equivalent(layout.centered_tracks(), updated.centered_tracks()))

    def test_near_match_gap_and_curve_direction_are_run_boundaries(self):
        short = self.run_layout(["83120", "83120"])
        self.assertIs(self.optimize(short), short)

        layout = Layout(self.catalog, "single", "test", "Test", "")
        first = layout.lay(Pose(), [Step("83102")])
        second = layout.lay(Pose(83.2, 0), [Step("83102")])
        layout.add("Gapped track", first + second, "spur")
        self.assertIs(self.optimize(layout), layout)

        s_curve = Layout(self.catalog, "single", "test", "Test", "")
        steps = [Step("83107"), Step("83107", mirror=-1), Step("83110")]
        s_curve.add("S curve", s_curve.lay(Pose(), steps), "spur")
        self.assertIs(self.optimize(s_curve), s_curve)

    def test_inventory_limits_include_shared_fixed_pieces(self):
        layout = self.run_layout(["83102"] * 2)
        layout.add("Second run", layout.lay(Pose(0, 200), [Step("83102")] * 2), "spur")
        shared = layout.lay(Pose(0, 300), [Step("83101")])
        layout.add("Shared", shared, "spur")
        layout.add("Shared in reverse", backwards(shared), "spur")
        for limits in ({"83101": Limit(max_count=2)}, {"83102": Limit(min_count=2)}):
            with self.subTest(limits=limits):
                updated = self.optimize(layout, Config(parts=limits))
                self.assertEqual(updated.inventory, Counter({"83101": 2, "83102": 2}))
                self.assertEqual(updated.tracks[2].moves[0].piece, updated.tracks[3].moves[0].piece)

        cfg = Config(parts={"83101": Limit(enabled=False)})
        unshared = self.run_layout(["83102"] * 2)
        self.assertIs(self.optimize(unshared, cfg), unshared)

    def test_opt_out_and_unpriced_families_use_inventory_repair(self):
        catalog = self.catalog.copy()
        g3 = catalog["83105"]
        catalog["83105"] = replace(g3, data={key: value for key, value in g3.data.items()
                                             if key != "unit_price_eur_cents"})
        layout = self.run_layout(["83102"] * 2, catalog=catalog)
        self.assertIs(self.optimize(layout), layout)
        for opt_in in (True, False):
            with self.subTest(opt_in=opt_in):
                cfg = Config(optimize_track_parts=opt_in, parts={"83102": Limit(max_count=1)})
                self.assertEqual(self.optimize(layout, cfg).inventory, Counter({"83101": 1}))

    def test_generated_crossover_exports_the_selected_inventory(self):
        cfg = Config(width_cm=155, height_cm=90, mode="double", count=1, strategy="longest",
                     max_turnouts=4, parts={"83212": Limit(min_count=1)})
        results = []
        with TemporaryDirectory() as root:
            for opt_in in (False, True):
                config = replace(cfg, optimize_track_parts=opt_in)
                plans, rejected = generate(self.catalog, config)
                self.assertEqual(len(plans), 1)
                layout = plans[0]
                self.assertIsNone(validate(layout, config))
                self.assertEqual(layout.inventory["83212"], 1)
                results.append(layout)

                output = Path(root) / str(opt_in)
                write_run(plans, rejected, config, output)
                report = json.loads((output / "run.json").read_text())
                self.assertIs(report["config"]["optimize_track_parts"], opt_in)
                plan = output / report["layouts"][0]["plan"]
                placement = json.loads((plan / "placement.json").read_text())
                self.assertEqual(Counter(part["code"] for part in placement["parts"]), layout.inventory)
                with (plan / "shopping.csv").open(encoding="utf-8-sig") as file:
                    shopping = {row["tillig_code"]: int(row["quantity_pieces"]) for row in csv.DictReader(file)
                                if row["category"] == "track"}
                self.assertEqual(shopping, dict(layout.inventory))
                self.assertTrue((plan / "layout.jpg").is_file())

        original, optimized = results
        self.assertTrue(equivalent(original.centered_tracks(), optimized.centered_tracks()))
        self.assertAlmostEqual(original.total_track_length_mm, optimized.total_track_length_mm)
        self.assertLess(len(optimized.pieces), len(original.pieces))
        original_price, optimized_price = (
            sum(piece.part.data.get("unit_price_eur_cents", 0) for piece in layout.pieces)
            for layout in results
        )
        self.assertLess(optimized_price, original_price)
