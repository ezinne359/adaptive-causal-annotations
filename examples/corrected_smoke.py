"""Small generated-data example; no private inputs or external downloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from corrected_annotation.pipeline import make_split, fit_weights, benchmark, draw
rng = np.random.RandomState(842)
x = rng.normal(size=(240, 3)); z = np.tile([0, 1], 120)
y = x[:, 0] + z + rng.normal(size=240)
proxy = (y + rng.normal(size=240))[:, None]
fold, pilot = make_split(len(z))
w, clipping, audit, failures = fit_weights(x, z, fold, propensity='forest')
refs = benchmark(x, z, y, proxy, fold, 'llm-eff', w)
rows = draw(x, z, y, proxy, fold, pilot, w, clipping, failures, refs,
            'llm-eff', .5, 0, propensity='forest')
assert len(rows) == 6
for row in rows:
    print(row['policy'], row['weighting'], row['comparison_status'])
assert all(row['comparison_status'] == 'completed' for row in rows)
