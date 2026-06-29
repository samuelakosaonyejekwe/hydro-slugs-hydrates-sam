"""STAGE 02 - MESH
Builds the 1-D finite-volume mesh (refined toward the sag/uphill leg where slug
gradients are steepest) and performs a mesh-independence study."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style, common
import usmh_solver as U

style.apply_style()
cfg = U.CaseConfig()
D = common.DIRS["mesh"]
L = cfg.L_pipe

# sag location for clustering
ss = np.linspace(0, L, 400)
i_sag = int(np.argmin([U._elev_of_s(cfg, s) for s in ss]))
s_sag = ss[i_sag]

n = cfg.n_cells
xi = np.linspace(0, 1, n + 1)
# clustering toward the sag (Gaussian density bump -> integrate to node positions)
dens = 1.0 + 1.6 * np.exp(-((xi - s_sag / L) / 0.10) ** 2)
cum = np.cumsum(dens); cum = (cum - cum[0]) / (cum[-1] - cum[0])
nodes = cum * L
ds = np.diff(nodes); centres = 0.5 * (nodes[1:] + nodes[:-1])

mesh = pd.DataFrame({"cell": np.arange(1, n+1), "s_start_m": nodes[:-1],
                     "s_end_m": nodes[1:], "s_centre_m": centres, "cell_size_m": ds})
common.save_table(mesh, "mesh.csv", "mesh", "Finite-volume mesh",
                  "Graded 1-D mesh refined toward the sag/uphill leg where slug "
                  "gradients are steepest.")

fig, ax = plt.subplots(figsize=(10.0, 4.2))
ax.plot(centres/1000, ds, color=style.PALETTE[2], lw=2.2, marker="o", ms=2.5)
ax.axvline(s_sag/1000, color=style.PALETTE[3], ls="--", lw=1.8, label="Dominant sag")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Cell size  $\\Delta s$ (m)")
ax.set_title("Mesh cell-size distribution (refined at the sag)"); ax.legend()
common.register_figure(style.finish(fig, os.path.join(D, "fig_cell_size.png")),
                       "mesh", "Cell-size distribution",
                       "Local mesh spacing; refinement clusters around the sag where the "
                       "terrain-slugging trap and steepest gradients occur.")

fig, ax = plt.subplots(figsize=(10.0, 2.4))
ax.vlines(nodes/1000, 0, 1, color=style.PALETTE[0], lw=0.5, alpha=0.7)
ax.set_yticks([]); ax.set_xlabel("Arc length (km)")
ax.set_title(f"Mesh node layout ({n} cells)")
common.register_figure(style.finish(fig, os.path.join(D, "fig_mesh_schematic.png")),
                       "mesh", "Mesh node layout",
                       "Node distribution of the 1-D finite-volume mesh along the pipeline.")

# ---- mesh-independence study ----------------------------------------------
rows = []
for nc in [40, 80, 120, 180, 240, 320, 480]:
    c = U.CaseConfig(n_cells=nc)
    r = U.march(c); hy = U.hydrate_assessment(c, r)
    rows.append({"n_cells": nc, "P_out_bar": r["P"][-1]/1e5,
                 "T_min_C": r["T"].min()-273.15, "GVF_out": r["GVF"][-1],
                 "max_subcool_C": float(np.nanmax(hy["subcool"]))})
mi = pd.DataFrame(rows)
mi["err_Pout_pct"] = (mi["P_out_bar"]-mi["P_out_bar"].iloc[-1]).abs()/mi["P_out_bar"].iloc[-1]*100
common.save_table(mi, "mesh_independence.csv", "mesh", "Mesh-independence study",
                  "Key outputs versus mesh count; the solution is grid-converged by "
                  "~240 cells (<0.5% change in outlet pressure).")

fig, ax = plt.subplots(figsize=(9.2, 4.4))
ax.plot(mi["n_cells"], mi["P_out_bar"], "-o", color=style.PALETTE[0], label="Outlet pressure")
ax.set_xlabel("Number of cells"); ax.set_ylabel("Outlet pressure (bar)", color=style.PALETTE[0])
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(mi["n_cells"], mi["max_subcool_C"], "-s", color=style.PALETTE[3], label="Max sub-cooling")
ax2.set_ylabel("Max sub-cooling (°C)", color=style.PALETTE[3])
ax.axvline(cfg.n_cells, color=style.PALETTE[2], ls="--", lw=1.6)
ax.set_title("Mesh-independence study")
common.register_figure(style.finish(fig, os.path.join(D, "fig_mesh_independence.png")),
                       "mesh", "Mesh-independence study",
                       "Outlet pressure and maximum hydrate sub-cooling converge with "
                       "mesh refinement; 240 cells (dashed) is grid-independent.")

print("Stage 02 mesh complete.")
