"""Generate and export track plans from the command line."""

import argparse
import logging
from pathlib import Path
import sys

from .config import MODES, STRATEGIES, read_config
from .export import write_run
from .generate import generate
from .geometry import load_catalog

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--catalog", type=Path, default=root / "tillig_tt.json")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/layouts"))
    parser.add_argument("--width-cm", type=float)
    parser.add_argument("--height-cm", type=float)
    parser.add_argument("--count", type=int)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--train-length-cm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--vehicle-group", choices=("A", "B", "C"))
    parser.add_argument("--edge-margin-mm", type=float)
    args = vars(parser.parse_args(argv))
    catalog_path = args.pop("catalog")
    config_path = args.pop("config")
    output = args.pop("output")
    try:
        if output.exists():
            raise ValueError(f"Output already exists at {output}. Choose a new --output directory.")
        catalog = load_catalog(catalog_path)
        if config_path is None:
            config_path = {
                root / "tillig_tt.json": root / "layout.json",
                root / "roco_line_h0.json": root / "layout.roco_line_h0.json",
            }.get(catalog_path.resolve())
        cfg = read_config(config_path, catalog, {k:v for k,v in args.items() if v is not None})
        logger.info("Generating up to %d %s plans for a %g × %g cm board",
                    cfg.count, catalog.info["system"], cfg.width_cm, cfg.height_cm)
        layouts, rejected = generate(catalog, cfg)
        logger.info("Exporting %d of %d requested plans", len(layouts), cfg.count)
        write_run(layouts, rejected, cfg, output, catalog=catalog)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(f"Wrote {len(layouts)} of {cfg.count} plans to {output.resolve() / 'index.md'}")
    if len(layouts) < cfg.count:
        for reason, count in rejected.most_common(5):
            print(f"  {count} candidate(s): {reason}", file=sys.stderr)
        if not rejected:
            print("  No stock formation matched the board, inventory, and train length.", file=sys.stderr)
        return 1
    return 0


def run():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())


if __name__ == "__main__":
    run()
