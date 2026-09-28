"""The linguistic field: a control on the mixing-law claim, not evidence for it.

The two physical fields of ``rate-fields`` are instruments that count, and both
are fitted at a negative order, which is outside the range a gamma mixing law can reach. This
field is the deliberate outsider. Tokens are counted, so the process has the same shape, but
nothing physical is being measured, and the question it answers is whether the argument for a
third parameter is about counting or about the particular fields chosen to make it.

It is kept apart from the two physical fields for a reason. Reported beside them it reads as
one result over three fields, which is the opposite of what it is: it is the case that
*disagrees*, and it earns its place by disagreeing. On this field the third parameter is
worth almost nothing per observation and the fitted order comes out positive, inside the
gamma's own range.

The laws, the fitting and the row format are imported from ``rate-fields`` rather
than restated, so the two cannot drift apart.

Writes ``results/newsgroup_fits.csv``. Figure from ``visualize.py``.

Run: python experiments/newsgroups/run.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RATE_FIELDS = os.path.join(HERE, "..", "rate-fields")
sys.path.insert(0, RATE_FIELDS)

from compare import fit_fields, write_rows                            # noqa: E402
from fields import LINGUISTIC_FIELDS                                  # noqa: E402


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    write_rows(os.path.join(HERE, "results", "newsgroup_fits.csv"),
               fit_fields(LINGUISTIC_FIELDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
