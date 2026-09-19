"""Safecast cells with their coordinates, which the rate-field loader discards.

Reads the same cached download as ``experiments/experiment-rate-fields``: 115k drive-by
measurements in counts per minute within 60 km of Fukushima Daiichi, CC0. A cell is a 500 m
square; its rate is the mean of the passes a Geiger tube made through it, and cells visited
fewer than ``MIN_PASSES`` times are dropped, because a single drive-by measures a cell rather
than estimating it.

The survey study needs each cell's distance and bearing from the plant as well as its rate,
because those are what a survey planner knows before going out and are what the priors are
conditioned on.
"""

import collections
import os

import numpy as np

CENTRE = (37.4211, 141.0328)      # Fukushima Daiichi
CELL_M = 500.0
MIN_PASSES = 10
RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "experiment-rate-fields", "data", "safecast_raw.npy")

__all__ = ["load_cells", "STRATA", "stratum_of"]

#: Strata a planner can assign before measuring: distance band, and whether the cell lies in
#: the north-west corridor the 2011 plume ran along. Each holds at least 128 cells, which is
#: enough to split in half and still fit a prior to each half.
STRATA = ("inner_nw", "mid_nw", "outer_nw", "outer_other")


def stratum_of(d_km, bearing_deg):
    nw = (bearing_deg >= 270.0) | (bearing_deg < 20.0)
    if d_km < 30.0:
        return "inner_nw"
    if d_km < 45.0:
        return "mid_nw"
    return "outer_nw" if nw else "outer_other"


def load_cells():
    """``(rate_cpm, distance_km, bearing_deg)`` for every cell with enough passes."""
    a = np.load(os.path.normpath(RAW))
    lat, lon, cpm = a[:, 0], a[:, 1], a[:, 2]
    ok = (cpm > 0) & (cpm < 1e6) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, cpm = lat[ok], lon[ok], cpm[ok]

    deg = CELL_M / 111000.0
    lon_deg = deg / np.cos(np.radians(CENTRE[0]))
    kx = np.round(lat / deg).astype(int)
    ky = np.round(lon / lon_deg).astype(int)

    cells = collections.defaultdict(list)
    for a_, b_, v in zip(kx, ky, cpm):
        cells[(a_, b_)].append(v)
    keys = [k for k, v in cells.items() if len(v) >= MIN_PASSES]

    rate = np.array([float(np.mean(cells[k])) for k in keys])
    ks = np.array(keys, dtype=float)
    dy = (ks[:, 0] * deg - CENTRE[0]) * 111.0
    dx = (ks[:, 1] * lon_deg - CENTRE[1]) * 111.0 * np.cos(np.radians(CENTRE[0]))
    return rate, np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360.0
