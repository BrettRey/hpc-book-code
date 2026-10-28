# Simulation scripts for *Words That Won't Hold Still*

Two agent-based sketches from Brett Reynolds's book *Words That Won't Hold Still: How Linguistic Categories Work*. Both are didactic: they make a proposed mechanism visible and show what it can and can't produce. They aren't calibrated simulations, and they aren't evidence for the book's claims about English.

| Script | Where the book uses it | What it sketches |
|---|---|---|
| `code/volcano_abm.py` | Chapter 7, the volcanic-island interlude | Agents learning a category boundary from a semantic cue and a distributional cue, before and after the semantic cue becomes noise |
| `code/countability_abm.py` | The diagnostics appendix, where the book's ablation test is applied to it | Count/non-count packaging as thresholds on a latent individuation score, with switches for ablation checks |

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

## Switches in the countability sketch

`World(...)` takes five switches. They are all off by default, and with the defaults the model reproduces the original exactly.

- `learner_prior`: `'fixed'` centres new learners on each noun's starting value; `'community'` centres them on the current population mean.
- `regularization_rate`: the probability that a speaker forces an exact-count frame it can't license (no singulative available).
- `hearer_response`: `'evidence'` or `'error'`, for how hearers treat a forced form.
- `regress_on_loose`: whether hearing a looser construction lowers the hearer's score toward that construction's threshold.
- `measure_phrase_evidence`: whether a measure phrase (*a piece of data*) counts as non-count evidence.

## Licence

MIT. See `LICENSE`.
