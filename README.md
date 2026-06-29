# Hydro Slugs & Hydrates — Subsea Tieback Flow-Assurance Study

**Author: Akosa Samuel Onyejekwe**

A flow-assurance study of a deepwater subsea oil **tieback**, focused on the two
phenomena that most threaten steady production in a long, undulating subsea line:
**multiphase slugging** (hydrodynamic, terrain, and severe/riser-base) and
**gas-hydrate formation**. The study marches a one-dimensional multiphase model
(`USMH-1`) along the pipeline, evaluates slug statistics and hydrate sub-cooling
against published correlations, screens **MEG** (mono-ethylene glycol) inhibition,
and renders the full set of engineering figures and drawings.

The complete write-up is included as **[`hydro_slugs_report.pdf`](hydro_slugs_report.pdf)** (55 pages).

---

## The case at a glance

| Parameter | Value |
|---|---|
| Line | 16 in (406.4 mm ID) × 16.0 km, API 5L X65, 18.3 mm wall |
| Water depth | 1480 m (mean), terrain undulation ±18 m |
| Dominant sag | 55 m below trend @ 7.8 km (uphill leg 126 m) |
| Production | 9000 stb/d oil, ṁ ≈ 22.2 kg/s |
| Inlet | 70 bar, 60 °C |
| Environment | 4 °C seabed, OHTC ≈ 9 W/m²·K |

### Headline results

| KPI | Value |
|---|---|
| Total pressure drop | −1.8 bar (gravity-dominated; arrival ≈ 71.8 bar) |
| Minimum / outlet temperature | 5.1 °C |
| Mean liquid holdup | 0.90 (slug flow over 100 % of line) |
| Mean slug frequency / length | 0.077 Hz / 13.0 m |
| Slug translational velocity | 1.49 m/s |
| Terrain-slug period / swing / surge | 10.1 min / 9.0 bar / 22× mean liquid rate |
| Bøe number @ design | 5.3 (stable at design; slugs at turndown) |
| Hydrate-risk zone | 9.4 km — **59 % of the line** while flowing |
| Max hydrate sub-cooling (flowing) | 10.1 °C |
| Shutdown no-touch time | ≈ 0 h |
| Recommended MEG dose | ~30 wt % (flowing) / ~50 wt % (shut-in) |

> **Bottom line:** the line is gravity-dominated and runs cold — the bulk of it
> sits inside the hydrate envelope during normal flow and the shut-in no-touch
> time is effectively zero, so continuous MEG injection and an aggressive
> shut-in dose are required. Terrain slugging at the 7.8 km sag drives large,
> low-frequency liquid surges that the host separator must absorb.

---

## Repository layout

The project is organised as a sequential pipeline; each stage has a `run.py`
driver that consumes the previous stage and writes CSV data plus figures.

| Stage | Contents |
|---|---|
| `01_geometry/` | Route, elevation profile, inclination, terrain undulation |
| `02_mesh/` | 1-D discretisation, cell sizing, mesh-independence study |
| `03_model_setup/` | Boundary/operating conditions, PVT tables, fluid properties, calibration constants |
| `04_solution/` | Solver run — spatial profiles, slug characteristics, terrain/severe slugging time series, MEG sweep, KPI summary |
| `05_postprocessing/` | ~45 result figures: profiles (A), slug dynamics (B), hydrate risk (C), route/3-D fields (D), cross-sections (E) |
| `06_validation/` | Validation against published data and correlations |
| `07_engineering_drawings/` | Cross-section, elevations, plan, isometric, orthographic drawings |
| `hydro_slugs_report.pdf` | Full engineering report |

Each stage folder mixes inputs/outputs as CSV with the figures (`.png`) they
produce, so the data behind every plot is inspectable directly.

---

## Validation

| Quantity | Reference | Result |
|---|---|---|
| Hydrate equilibrium T | Deaton & Frost (1946) | AAD 0.28 °C, RMSE 0.30 °C, R² 0.996 — **excellent** |
| Slug translational velocity | Nicklin (1962) / Bendiksen (1984) | Exact reproduction of accepted drift-flux coefficients |
| Slug frequency | Zabaras (2000) | Mean deviation 18 % (within ±30 % model scatter) |

---

## Methods & correlations

The model is assembled from established multiphase-flow and hydrate correlations:

- **Pressure drop / friction** — Beggs & Brill (1973); Haaland (1983)
- **Flow-pattern transitions** — Taitel & Dukler (1976); Barnea (1987)
- **Slug dynamics** — Nicklin et al. (1962); Bendiksen (1984); Gregory & Scott (1969); Zabaras (2000); Gregory, Nicholson & Aziz (1978)
- **Severe / terrain slugging** — Taitel (1986); Bøe (1981)
- **Hydrates** — Motiee (1991); Hammerschmidt (1934) MEG depression; validated on Deaton & Frost (1946), per Sloan & Koh (2008)
- **PVT** — Standing (1947) GOR/FVF; Beggs & Robinson (1975) live-oil viscosity; Beggs & Brill (1973) / Sutton (1985) gas Z-factor

---

## Reproducing the figures

Each `run.py` is a stage driver written in Python (NumPy, pandas, Matplotlib,
SciPy). The drivers import a shared solver package (`usmh_solver`, `style`,
`common`) that is **not distributed in this repository** — so the stage scripts
are included for transparency and provenance rather than as a turnkey runner.
All generated outputs (CSV data, figures, drawings) and the final report are
committed directly, so the results are fully inspectable without re-running the
solver.

---

## Disclaimer

This is an independent engineering study by **Akosa Samuel Onyejekwe**, built on
published correlations and representative tieback parameters. It is intended for
educational and illustrative flow-assurance analysis, not as a design basis for
any specific asset.
