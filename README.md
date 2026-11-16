# Tillig TT layout generator

Generate to-scale assembly drawings for a flat model railroad using fixed-length Tillig TT Modellgleis. The included [10 example plans](examples/index.md) fit a 160 × 120 cm board and use the 25 cm train-length setting, MOROP vehicle group B, and DCC.

The checked-in examples are from an earlier run. `05-double` and `10-double` have independent loops.

## Run it

From this directory, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync --locked --python 3.12
uv run --locked python -m tt_layouts --output output/my-layouts
uv run --locked python -m tt_layouts --mode passing --train-length-cm 30 --count 4 --output output/station
uv run --locked python -m tt_layouts --mode double --strategy longest --output output/double-main
uv run --locked python -m tt_layouts --strategy longest --output output/longest
uv run --locked python -m tt_layouts --strategy longest_tracks --output output/longest-tracks
uv run --locked python -m unittest discover -s tests -v
```

`uv sync` creates `.venv`. You can also run `.venv/bin/python -m tt_layouts` directly. After setup, generation runs locally with no network access. Use a fresh output directory for each run. Exit status 0 means the requested count was met, 1 means a partial or empty result with a shortfall report, and 2 means an input or export error.

## Choose the layout and inventory

Edit [layout.json](layout.json) or use the CLI flags above. CLI flags override the corresponding JSON settings. `--config` selects another JSON file.

| Setting | Default | Meaning |
|---|---:|---|
| `width_cm`, `height_cm` | 160, 120 | Baseboard dimensions. A plan can rotate 90° to fit. |
| `count` | 10 | Number of distinct alternatives requested |
| `ignore_similar_percentage` | 0 | Track length tolerance as a fraction from `0` to `1`. The included `layout.json` uses `0.05` (5%). `0` disables the filter. |
| `mode` | `mixed` | `single` for a main line with freight switching, `passing` for two trains to meet, or `double` for simultaneous running on two closed routes |
| `require_connected_tracks` | `true` | Requires a traversable connection between all tracks in each plan. Set to `false` to allow independent loops. |
| `min_turnouts` | 1 | Minimum number of turnouts in each plan. Set to `0` to disable the minimum. |
| `max_turnouts` | 10 | Maximum number of turnouts in each plan. Must be at least `min_turnouts`. |
| `strategy` | `balanced` | Diversifies operating formations. `longest` ranks by continuous lap, `longest_tracks` by total installed track length, `shunting` by total usable freight storage, and `fewest_parts` by track piece count. |
| `train_length_cm` | 25 | Complete train including locomotive. Both station tracks must hold this length in passing mode. Double mode requires at least this much parallel running track. |
| `vehicle_group` | B | MOROP vehicle group A, B, or C for radius and clearance |
| `edge_margin_mm` | 35 | Minimum distance from track centerline to board edge |
| `seed` | 42 | Repeatable ordering of otherwise tied layouts |

Set `"min_turnouts": 5` and `"max_turnouts": 8` in `layout.json` to require five to eight turnouts per plan, inclusive. The generator counts one turnout per drive listed in the catalog. A double crossover counts as four, a three-way turnout or double slip counts as two, and a fixed crossing counts as zero. If too few plans meet the turnout limits, the generator reports a shortfall.

`longest_tracks` counts both main lines, sidings, yard roads, passing tracks, and all routes through installed pointwork. It counts shared centerline sections once and includes the track up to each buffer stop. The search also tries long interior sidings and additional yard roads where stock and board dimensions allow.

With `"ignore_similar_percentage": 0.05`, the generator skips a plan when it has the same named tracks as an earlier selected plan of the same mode and every corresponding length is within 5% of that plan's track length. A double-track plan with a goods road and one with only the two laps can both be selected. If too few plans meet the requested diversity, the run reports a shortfall.

The `parts` object applies to each alternative independently. This example requires at least one EW2 right turnout, disables EW2 left, and allows up to five buffer stops.

```json
{
  "parts": {
    "83100": {"enabled": true, "max_count": 5},
    "83331": {"enabled": true, "min_count": 1},
    "83332": {"enabled": false}
  }
}
```

Absent rules allow any quantity. `max_count: null` means unlimited, and `max_count: 0` forbids use. The generator can substitute enabled sectional pieces that preserve a straight or curve's geometry to meet inventory limits. The included sample enables 83331 without requiring it in every plan. Each SKU retains its actual turnout handedness.

## Build from a plan

Browse the `all/` folder in a generated run to see every drawing in plan order. Filenames use three-digit numbering, such as `001-single-layout.jpg`.

Open the output `index.md` to compare operating moves, total track length, longest lap, usable track lengths, and motor counts. The total also appears as `total_track_length_mm` in `run.json` and `placement.json`. The 3000 × 2360 JPG shows a board outline, centimetre grid, scale bar, joints and one label per physical piece. The first number in a label is the placement ID. The second is the Tillig SKU. Each `shopping.csv` lists actual piece quantities and an under-baseboard motor option where needed. Buy against the list for the alternative you choose. `placement.json` gives origin, rotation, connector poses, traversable routes, and the measured joint residuals in millimetres and degrees.

Track plans assume DCC. On double-track designs with a crossover, simultaneous circulation uses the two through routes. Stop conflicting movements before lining the crossover diagonally. Pointwork manuals cover frog wiring, insulation and installation of drives. Add power feeds, wiring and other electrical supplies for your installation.

## Geometry and coverage

[tillig_tt.json](tillig_tt.json) stores the actual 12 mm gauge and 2.07 mm rail height, nominal dimensions, connector poses, route centerlines and manufacturer links. It includes straights, curves, buffer stops, EW1/EW2/EW3, IBW, ABW, three-way points, both fixed diamonds, DKW I and DGV. 83391 Baeseler is reference-only because Tillig publishes conflicting dimensions. The older Tillig dimensioned drawing shows the EW3 12° exit used here. The 83212 uses its published 251 × 43 mm connector footprint and a derived internal drawing.

Search combines stock-length oval and rounded-rectangle backbones with freight spurs, yard ladders, passing loops, and crossovers. It ranks the feasible designs found in those formations. The geometry uses rigid pieces and checks every joint. A closure can have up to 0.5 mm of longitudinal gap, 0.15 mm of lateral residual and 0.1° of tangent error. No sections are cut or stretched. Passing and siding figures use the clear side of sampled fouling points with 10 mm stopping room at each end. The radius and parallel-track checks use [MOROP NEM 111](https://morop.eu/images/NEM_register/NEM_E/nem111_en_2016_20240907.pdf) and [NEM 112](https://www.morop.eu/images/NEM_register/NEM_E/nem112_en_2024.pdf). Tillig's 267 mm curves are restricted to compatible industrial sidings.

The examples were checked against nominal geometry and have yet to be dry-fitted with physical track. Lay the pointwork and the loop's closing joints loose first to check production tolerances and the swept envelope of your actual vehicles. The catalog links to [Tillig Modellgleis](https://www.tillig.com/eng/Modellgleissysteme.html), the [Tillig dimensioned drawings](https://vasutmodell.com/pics/tt-tillig-modell-sinrendszer-geometriaja-338458.pdf), and the relevant installation manuals.
