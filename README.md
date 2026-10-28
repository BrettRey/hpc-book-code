# Simulation scripts for *Words That Won't Hold Still*

Agent-based sketches from Brett Reynolds's book *Words That Won't Hold Still: How Linguistic Categories Work*. Both are didactic: they make a proposed mechanism visible and show what it can and can't produce. They aren't calibrated simulations, and they aren't evidence for the book's claims about English.

| Script | Where the book uses it | What it sketches |
|---|---|---|
| `code/volcano_abm.py` | Chapter 7, the volcanic-island interlude | Agents learning a category boundary from a semantic cue and a distributional cue, before and after the semantic cue becomes noise |
| `code/countability_abm.py` | The diagnostics appendix, where the book's ablation test is applied to it | Count/non-count packaging as thresholds on a latent individuation score, with switches for ablation checks |
| `code/countability_two_cluster_abm.py` | The diagnostics appendix, the two-cluster sketch | Separate referent and grammatical variables per noun, coupled by bidirectional inference; *book* returns after a drop, and *furniture* is held non-count by statistical preemption |
| `code/population_maps/` | The same section's formal check | Population-level C versions of the two-cluster sketch for *book* and *furniture*, with the property functions that were checked and a random-sample harness |

`code/plot_style.py` holds the shared colour palette.

## Running

Python 3.10 or later.

```bash
pip install -r requirements.txt
cd code
python volcano_abm.py --exp main          # the Chapter 7 figure; --exp all adds controls, sweep, seed sensitivity
python countability_abm.py --exp=drift    # data-drift run
python countability_abm.py --exp=ablation # mechanism ablations, with computed differences from baseline
```

Figures are written to `figures/` and `code/`, and run manifests (settings and summary numbers) are written alongside them.

The two-cluster sketch has no command-line entry point. A minimal run:

```python
import countability_two_cluster_abm as m
w = m.World(m.Config(coupling=0.01, learner_gram_prior="semantic", preemption=0.5), seed=1)
for _ in range(300):
    w.phase()
print(w.mean_gram("furniture"))
```

The C files need only a C11 compiler:

```bash
cd code/population_maps
cc -std=c11 -O2 -DBOOK_BASIN_MAIN -o book_basin book_basin.c && ./book_basin
cc -std=c11 -O2 -DFURNITURE_BASIN_MAIN -o furniture_basin furniture_basin.c && ./furniture_basin
cc -std=c11 -O2 -I. -o sample_book sample_book.c -lm && ./sample_book 100000
cc -std=c11 -O2 -I. -o sample_furniture sample_furniture.c -lm && ./sample_furniture 100000
```

Each file's header comment gives the exact update rules. The `property_*` functions are the claims the book's appendix reports as formally checked; the `sample_*` programs test them on random admissible starting states.

## Switches in the countability sketch

`World(...)` takes five switches. They are all off by default, and with the defaults the model reproduces the original exactly.

- `learner_prior`: `'fixed'` centres new learners on each noun's starting value; `'community'` centres them on the current population mean.
- `regularization_rate`: the probability that a speaker forces an exact-count frame it can't license (no singulative available).
- `hearer_response`: `'evidence'` or `'error'`, for how hearers treat a forced form.
- `regress_on_loose`: whether hearing a looser construction lowers the hearer's score toward that construction's threshold.
- `measure_phrase_evidence`: whether a measure phrase (*a piece of data*) counts as non-count evidence.

## Licence

MIT. See `LICENSE`.
