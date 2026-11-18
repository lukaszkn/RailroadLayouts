"""Stock preferences for the bundled track systems."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackProfile:
    buffer_code: str
    right_turnout: str
    left_turnout: str
    grid_mm: float
    single_leads: tuple[float, ...]
    branch_leads: tuple[float, ...]
    freight_tails: tuple[float, ...]
    countercurve_radius: float
    siding_only_radius: float | None = None
    tillig_formations: bool = False

    @property
    def side_lengths(self):
        return (0, self.grid_mm, 2 * self.grid_mm, 3 * self.grid_mm)


TILLIG = TrackProfile("83100", "83331", "83332", 83, (83, 41.5, 0),
                      (41.5, 43, 83, 119.5, 36.5, 0), (290.5, 249, 207.5, 166), 353,
                      siding_only_radius=267, tillig_formations=True)
ROCO = TrackProfile("42608", "42443", "42442", 115, (115, 57.5, 0),
                    (0, 57.5, 115, 230), (460, 345, 230, 172.5), 888)
