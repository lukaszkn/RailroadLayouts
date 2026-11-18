import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from subprocess import run
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from railroad_layouts.config import Limit, read_config
from railroad_layouts.export import write_run
from railroad_layouts.generate import Stock, generate
from railroad_layouts.geometry import Pose, Step, align, joint_fits, load_catalog, min_radius, spacing
from railroad_layouts.model import Layout, validate

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "roco_line_h0.json"


class RocoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog(CATALOG)
        cls.config = read_config(ROOT / "layout.roco_line_h0.json", cls.catalog)

    def test_stock_geometry_and_handedness(self):
        catalog = self.catalog
        self.assertEqual(catalog.info, dict(system="Roco Line H0 Code 83", scale="1:87",
                                            gauge_mm=16.5, rail_height_mm=2.1))
        self.assertEqual(catalog["42423"].radius, 419.6)
        self.assertEqual(catalog["42412"].routes["a-b"].length, 115)
        for left, right, angle, length in (("42442", "42443", 15, 230), ("42490", "42491", 10, 345)):
            with self.subTest(angle=angle):
                self.assertEqual(catalog[left].ports["c"], Pose(length, -30.8, 360-angle))
                self.assertEqual(catalog[right].ports["c"], Pose(length, 30.8, angle))
                first = align(catalog[right], "a", Pose())
                second = align(catalog[right], "a", Pose(2*length, 61.6, 180))
                self.assertTrue(joint_fits(first.ports["c"], second.ports["c"]))

    def test_scale_rules_and_spacers(self):
        self.assertEqual([min_radius(group, 16.5) for group in "ABC"], [354.75, 412.5, 495])
        self.assertEqual(spacing("B", 425, 16.5), 58)
        self.assertEqual(spacing("C", float("inf"), 16.5), 52)
        stock = Stock(self.catalog, self.config)
        self.assertNotIn(358, stock.curves)
        self.assertIn(358, Stock(self.catalog, replace(self.config, vehicle_group="A")).curves)
        self.assertIsNone(stock.straight(8))
        self.assertEqual(tuple(stock.fitted_link(8)), ((Step("98003"),),))

    def test_clip_stop_fits_full_rail_and_counts_both_components(self):
        layout = Layout(self.catalog, "single", "test", "Test", "")
        move, = layout.lay(Pose(), (Step("42608", "a-stop"),))
        self.assertEqual(layout.port(move, True).x, 75)
        self.assertEqual(layout.bounds()[2], 115)
        self.assertEqual(layout.total_track_length_mm, 115)
        self.assertEqual(layout.inventory, Counter({"42412": 1, "42608": 1}))
        for code in layout.inventory:
            with self.subTest(code=code):
                cfg = replace(self.config, mode="single", parts={code: Limit(max_count=0)})
                plans, reasons = generate(self.catalog, cfg)
                self.assertFalse(plans)
                self.assertIn("42608", next(iter(reasons)))

    def check_exports(self, plans, rejected, cfg):
        with TemporaryDirectory() as root:
            output = Path(root) / "plans"
            write_run(plans, rejected, cfg, output)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["catalog"], self.catalog.info)
            for row, plan in zip(report["layouts"], plans, strict=True):
                folder = output / row["plan"]
                placement = json.loads((folder / "placement.json").read_text())
                self.assertEqual(placement["catalog"], self.catalog.info)
                pieces = Counter(part["code"] for part in placement["parts"])
                pieces.update(part["buffer_stop"]["code"] for part in placement["parts"]
                              if "buffer_stop" in part)
                with (folder / "shopping.csv").open(encoding="utf-8-sig") as file:
                    shopping = {item["part_code"]: int(item["quantity_pieces"])
                                for item in csv.DictReader(file) if item["category"] != "point_drive"}
                self.assertEqual(pieces, shopping)
                self.assertEqual(pieces, plan.inventory)
                self.assertTrue(pieces.keys() <= self.catalog.keys())
                with Image.open(folder / "layout.jpg") as drawing:
                    drawing.verify()

    def test_generate_and_export_all_standard_modes(self):
        cfg = replace(self.config, count=3)
        plans, rejected = generate(self.catalog, cfg)
        self.assertEqual([plan.mode for plan in plans], ["single", "passing", "double"])
        single, passing, double = plans
        self.assertGreater(single.inventory["42608"], 0)
        self.assertGreaterEqual(passing.lengths["Passing track"], cfg.train_length)
        self.assertGreaterEqual(passing.lengths["station_main"], cfg.train_length)
        self.assertEqual(sum(track.kind == "crossover" for track in double.tracks), 2)
        for plan in plans:
            self.assertIsNone(validate(plan, cfg))
        self.check_exports(plans, rejected, cfg)

    def test_station_with_four_tracks_and_two_curved_yard_sidings(self):
        cfg = read_config(ROOT / "layout.roco_line_h0.station.json", self.catalog,
                          {"min_station_tracks": 4})
        plans, rejected = generate(self.catalog, cfg)
        plan, = plans
        self.assertIsNone(validate(plan, cfg))
        self.assertEqual(len(plan.station_tracks), 4)
        self.assertGreaterEqual(plan.lengths["station_usable"], cfg.train_length)
        self.assertEqual(len(plan.interior_tracks), 2)
        self.assertEqual(plan.inventory["42608"], 2)
        self.assertGreaterEqual(plan.inventory["42412"], 2)
        self.check_exports(plans, rejected, cfg)

    def test_cli_shortfall_uses_selected_catalog(self):
        with TemporaryDirectory() as root:
            output = Path(root) / "plans"
            result = run([sys.executable, "-m", "railroad_layouts", "--catalog", str(CATALOG),
                          "--height-cm", "80", "--output", str(output)], cwd=ROOT,
                         capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["catalog"], self.catalog.info)
            self.assertEqual(report["config"]["width_cm"], 240)
            self.assertEqual(report["generated"], 0)
            self.assertIn("minimum main-line diameter", (output / "index.md").read_text())
