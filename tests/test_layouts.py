import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from tt_layouts.config import Limit, read_config
from tt_layouts.export import write_run
from tt_layouts.generate import Stock, generate, specialty_modes
from tt_layouts.geometry import Pose, Step, joint_fits, load_catalog
from tt_layouts.model import Layout, equivalent, validate

ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(ROOT / "tillig_tt.json")
        cls.config = read_config(ROOT / "layout.json", cls.catalog)
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

    def test_sample_mode_mix_and_inventory(self):
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
        cfg = self.config_for(mode="double", count=1, parts={"83212": Limit(enabled=False)})
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

    def test_fixed_diamond_does_not_connect_its_routes(self):
        layout = Layout(self.catalog, "single", "diamond", "Fixed crossing", "")
        crossing, = layout.lay(Pose(), (Step("83161"),))
        buffer = (Step("83100", "a-stop"),)
        for move in (crossing, layout.branch(crossing, "c", "d")):
            before, = layout.lay(layout.port(move), buffer)
            after, = layout.lay(layout.port(move, True), buffer)
            layout.add(move.entry, (before.reversed(), move, after), "spur")
        self.assertEqual(validate(layout, self.config), "track network is disconnected")

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
        for strategy in ("longest", "fewest_parts", "shunting"):
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
            for plan in ("01-single", "02-passing"):
                original = path / plan / "layout.jpg"
                collected = drawings / f"0{plan}-layout.jpg"
                self.assertEqual(collected.read_bytes(), original.read_bytes())

    def test_shortfall_report(self):
        with TemporaryDirectory() as root:
            path = Path(root) / "partial"
            write_run([], Counter({"board size or edge margin": 1}), self.config, path)
            result = json.loads((path / "run.json").read_text())
            self.assertEqual(result["shortfall"], 10)
            self.assertIn("board size or edge margin", (path / "index.md").read_text())


if __name__ == "__main__":
    unittest.main()
