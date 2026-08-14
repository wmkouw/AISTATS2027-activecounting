"""Three real fields of rates, from three instruments that count.

The model of Section 2 says a count is Poisson about a rate, and that the rates across
contexts follow a mixing law. Everything else in this paper takes the second half on trust or
generates it. This module goes and looks.

Each loader returns a vector of rates measured in the world, from a domain where the counting
model is the measurement process rather than an analogy:

    fermi_fluxes      photon fluxes of gamma-ray sources, Fermi-LAT 4FGL. A photon counter.
    safecast_cells    count rates per 500 m cell around Fukushima Daiichi, Safecast. A
                      Geiger tube, reporting counts per minute.
    newsgroup_terms   relative rates of terms across the 20 Newsgroups corpora. Tokens are
                      counted, so the process is the same shape, but the domain is not
                      physical and it is included precisely because it might disagree.

The point of putting three side by side is that a mixing law argued for on one field is an
anecdote. What the fields say about the third parameter, and whether they say the same thing,
is the question.
"""

import collections
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

__all__ = ["fermi_fluxes", "safecast_cells", "newsgroup_terms", "FIELDS"]


# --------------------------------------------------------------------------

def fermi_fluxes():
    """Integral photon fluxes above 1 GeV, in 1e-10 photons per cm^2 per second.

    Restricted to the 5th-95th percentile band, as in the scheduling study: the faint end is
    where a flux-limited catalogue is incomplete, and a handful of exceptionally bright
    sources would otherwise dominate any fit.
    """
    sys.path.insert(0, os.path.join(HERE, "..", "experiment-photon-counting"))
    from fetch import fetch
    flux = fetch() * 1e10
    lo, hi = np.percentile(flux, 5), np.percentile(flux, 95)
    return flux[(flux >= lo) & (flux <= hi)]


# --------------------------------------------------------------------------

SAFECAST_RAW = os.path.join(DATA, "safecast_raw.npy")
SAFECAST_API = "https://api.safecast.org/measurements.json"
#: Fukushima Daiichi, and the radius over which the contamination field varies by orders of
#: magnitude.
SAFECAST_CENTRE = (37.4211, 141.0328)
SAFECAST_RADIUS_M = 60000
CELL_M = 500.0
MIN_PASSES = 10


def _download_safecast(pages=120):
    recs = []
    socket.setdefaulttimeout(120)
    for page in range(1, pages + 1):
        q = urllib.parse.urlencode({
            "unit": "cpm", "latitude": SAFECAST_CENTRE[0],
            "longitude": SAFECAST_CENTRE[1], "distance": SAFECAST_RADIUS_M,
            "per_page": 1000, "page": page})
        payload = None
        for attempt in range(4):          # the API returns intermittent 502s
            try:
                payload = json.loads(urllib.request.urlopen(
                    SAFECAST_API + "?" + q).read())
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        if payload is None:
            continue
        if not payload:
            break
        recs += [(r["latitude"], r["longitude"], r["value"]) for r in payload
                 if r.get("value") is not None and r.get("latitude") is not None]
    return np.asarray(recs, dtype=float)


def safecast_cells(force=False):
    """Mean count rate per 500 m cell, in counts per minute.

    A cell's rate is the mean of the passes a Geiger tube made through it. Cells visited
    fewer than :data:`MIN_PASSES` times are dropped, because a single drive-by is a
    measurement of the cell rather than an estimate of it.
    """
    if force or not os.path.exists(SAFECAST_RAW):
        np.save(SAFECAST_RAW, _download_safecast())
    a = np.load(SAFECAST_RAW)
    lat, lon, cpm = a[:, 0], a[:, 1], a[:, 2]
    ok = (cpm > 0) & (cpm < 1e6) & np.isfinite(lat) & np.isfinite(lon)
    lat, lon, cpm = lat[ok], lon[ok], cpm[ok]
    deg = CELL_M / 111000.0
    kx = np.round(lat / deg).astype(int)
    ky = np.round(lon / (deg / np.cos(np.radians(SAFECAST_CENTRE[0])))).astype(int)
    cells = collections.defaultdict(list)
    for a_, b_, v in zip(kx, ky, cpm):
        cells[(a_, b_)].append(v)
    rates = np.array([np.mean(v) for v in cells.values() if len(v) >= MIN_PASSES])
    return rates


# --------------------------------------------------------------------------

NEWS_CACHE = os.path.join(DATA, "newsgroup_term_rates.npy")
MIN_TERM_COUNT = 200


def newsgroup_terms(force=False):
    """Relative rate of a term in a newsgroup, over terms present in all twenty.

    A term's rate in a group is its occurrences divided by that group's tokens, scaled by the
    term's mean rate so that terms of different overall frequency are comparable. Terms
    absent from any group are excluded: an observed zero there stands in for an unknown
    positive rate, and fitting it as if it were zero would be measuring the truncation rather
    than the field.
    """
    if os.path.exists(NEWS_CACHE) and not force:
        return np.load(NEWS_CACHE)
    import re
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.datasets import fetch_20newsgroups
    d = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    tok = re.compile(r"[a-z]+")
    per_group = collections.defaultdict(collections.Counter)
    n_tokens = collections.Counter()
    total = collections.Counter()
    for text, g in zip(d.data, d.target):
        words = tok.findall(text.lower())
        n_tokens[g] += len(words)
        total.update(words)
        per_group[g].update(words)
    groups = sorted(n_tokens)
    out = []
    for w, c in total.items():
        if c < MIN_TERM_COUNT:
            continue
        r = np.array([per_group[g].get(w, 0) / n_tokens[g] for g in groups])
        if (r > 0).all():
            out.append(r / r.mean())
    rates = np.concatenate(out)
    np.save(NEWS_CACHE, rates)
    return rates


#: Name, loader, and what one observation of the field is.
FIELDS = (
    ("Fermi-LAT sources", fermi_fluxes, "photon flux of a source"),
    ("Safecast cells", safecast_cells, "count rate of a 500 m cell"),
    ("20 Newsgroups terms", newsgroup_terms, "relative rate of a term in a group"),
)
