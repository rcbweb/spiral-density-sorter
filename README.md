# spiral-density-sorter
Simulation code, table generator, and interactive 3-D device render for the study "Inertial Microfluidic Density-Discriminating Separation of PTFE and PVDF Microparticles in Saline Brine Using a Spiral Channel Geometry: A Computational Feasibility Study" (Global Undergraduate Awards 2026 submission, Engineering).

The device is a passive five-turn Archimedean spiral microchannel that separates
equal-sized (8 um) PTFE and PVDF microparticles in 3.5 % NaCl brine using only
their 24 % density contrast. The discriminating force is the centrifugal-buoyancy
term, the spiral's constant curvature being what keeps it from cancelling.
Everything in the paper is reproducible from this repository.

## Contents

| Path | What it is |
|---|---|
| `make_figures.py` | Regenerates Figures 2-4 (force balance, exit distributions, Bhagat 2008 validation). Deterministic, seed 42. |
| `make_tables.py` | Regenerates Tables 1-3 as `.tex` and `.xlsx`. Usage: `python make_tables.py [outdir]` |
| `verify_numbers.py` | Recomputes every number quoted in the paper from the same constants and prints a full audit report. |
| `device_render/fig1_device_3d.html` | Interactive WebGL render of the chip (Figure 1). Open in any browser; drag to rotate, scroll to zoom. Animated particle streams illustrate the sorting concept schematically. |
| `figures/` | Pre-rendered figure files as they appear in the paper. |
| `requirements.txt` | Python dependencies (NumPy, SciPy, Matplotlib, openpyxl). |

## Quickstart

```bash
pip install -r requirements.txt
python make_figures.py      # writes fig2/fig3/fig4 (PDF + PNG), ~1 min
python make_tables.py .     # writes table1/2/3 (.tex + .xlsx)
python verify_numbers.py    # prints the full numerical audit
```

All randomness is seeded (NumPy seed 42), so every figure, table, and quoted
number regenerates exactly. No model parameter was fitted to the outputs; the
lift profile is fixed a priori and benchmarked against Bhagat et al. (2008)
with no parameter changed (predicted focusing position within 2.5 % of channel
width).

## Headline result

With a sheath-focused inlet, the Langevin trajectory model (n = 5000 particles
per species) predicts species equilibria 16.4 um apart on the inner branch,
exit streams centred 13.2 um apart, and a single splitter at y = -39.7 um
routing 100 % of the simulated PTFE ensemble and 98.5 % of PVDF to their
target outlets (outlet purities 98.5 % and 100 %). The paper states these as
upper bounds: no chip has been fabricated, and the model's limitations are
disclosed in full in Section 6.4.

## Note on the render

The animation in `device_render/` is a visual aid: particle lanes settle to
opposite walls so the concept reads clearly at device scale. The quantitative
physics lives in `make_figures.py`; see Figures 2-3 and Table 3 of the paper
for the predicted equilibria and distributions.

## License

MIT (see `LICENSE`).
