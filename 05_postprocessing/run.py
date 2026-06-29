"""STAGE 05 - POST-PROCESSING
Generates every figure: spatial profiles, slug analysis, hydrate analysis,
2-D pressure / temperature contours, 3-D contours and velocity vectors.
All figures use bright colours (never black/dark) and non-overlapping text."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa
import style, common
import usmh_solver as U

style.apply_style()
P = common.DIRS["post"]; DD = common.DIRS["data"]
cfg = U.CaseConfig()
R = dict(np.load(os.path.join(DD, "results.npz"), allow_pickle=True))
SS = dict(np.load(os.path.join(DD, "severe_slug.npz")))
TR = dict(np.load(os.path.join(DD, "slug_train.npz")))
RM = dict(np.load(os.path.join(DD, "regime_map.npz"), allow_pickle=True))
regime = np.load(os.path.join(DD, "regime_labels.npy"), allow_pickle=True)

s = R["s"]; skm = s/1000.0
i_sag = int(np.argmin(R["elev"])); s_sag = skm[i_sag]
hyd = R["hy_subcool"] > 0
def reg(fig, name, title, cap):
    common.register_figure(style.finish(fig, os.path.join(P, name)), "post", title, cap)
def zones(ax):
    """Shade the hydrate-risk zone and mark the dominant sag (shared context)."""
    if hyd.any():
        ax.axvspan(skm[hyd].min(), skm[hyd].max(), color="#dbe7ff", alpha=0.45,
                   zorder=0, label="Hydrate-risk zone")
    ax.axvline(s_sag, color=style.PALETTE[3], ls=":", lw=1.4, alpha=0.7, zorder=1)

# ============================================================ A. PROFILES ===
# A1 pressure
fig, ax = plt.subplots(figsize=(9.8, 4.6)); zones(ax)
ax.plot(skm, R["P"]/1e5, color=style.PALETTE[0], lw=2.8)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Pressure (bar)")
ax.set_title("Pressure profile along the subsea pipeline")
_pk = R["P"][i_sag]/1e5
ax.set_ylim(top=_pk + 4.5)                      # headroom so the label stays inside the axes
ax.annotate("sag", xy=(s_sag, _pk), xytext=(s_sag, _pk + 3.0),
            ha="center", va="bottom", color=style.PALETTE[3], fontsize=9,
            arrowprops=dict(arrowstyle="->", color=style.PALETTE[3]))
ax.legend(loc="upper right")
reg(fig, "A1_pressure_profile.png", "Pressure profile",
    "Pressure along the low-pressure tieback; the profile is controlled by the seabed "
    "terrain (hydrostatic head) rather than friction, which is small for this large-bore "
    "line. Vertical dotted line marks the dominant sag.")

# A2 temperature + hydrate envelope (KEY)
fig, ax = plt.subplots(figsize=(9.8, 4.8)); zones(ax)
ax.plot(skm, R["T"]-273.15, color=style.PALETTE[3], lw=2.8, label="Fluid temperature")
ax.plot(skm, R["Tamb"]-273.15, color=style.PALETTE[5], lw=2.0, ls="--", label="Ambient (seabed, 4 °C)")
ax.plot(skm, R["hy_Teq_eff"]-273.15, color=style.PALETTE[0], lw=2.4, label="Hydrate equilibrium $T_{eq}$")
risk = R["hy_subcool"] > 0
ax.fill_between(skm, R["T"]-273.15, R["hy_Teq_eff"]-273.15, where=risk,
                color="#9bbcff", alpha=0.55, label="_nolegend_")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Temperature (°C)")
ax.set_title("Temperature profile and hydrate-stability envelope")
ax.legend(loc="upper right", ncol=2)
reg(fig, "A2_temperature_hydrate.png", "Temperature profile & hydrate envelope",
    "Fluid temperature falls below the hydrate equilibrium temperature over the cold "
    "downstream pipeline (shaded) - the predicted hydrate-formation zone during flow.")

# A3 subcooling & risk
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
ax.plot(skm, R["hy_subcool"], color=style.PALETTE[0], lw=2.6, label="Sub-cooling $\\Delta T_{sub}$")
ax.axhline(0, color=style.PALETTE[3], lw=1.6, ls="--")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Sub-cooling (°C)", color=style.PALETTE[0])
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(skm, R["hy_risk"], color=style.PALETTE[1], lw=2.2, label="Hydrate-risk index")
ax2.set_ylabel("Hydrate-risk index (0-1)", color=style.PALETTE[1]); ax2.set_ylim(0, 1.05)
ax.set_title("Hydrate sub-cooling and risk index")
reg(fig, "A3_subcooling_risk.png", "Sub-cooling & risk",
    "Driving force for hydrate formation (sub-cooling, positive = inside envelope) and "
    "the normalised risk index along the line.")

# A4 holdup & GVF
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
ax.plot(skm, R["HL"], color=style.PALETTE[2], lw=2.8, label="Liquid holdup $H_L$")
ax.plot(skm, R["GVF"], color=style.PALETTE[4], lw=2.4, label="Gas void fraction $\\alpha$")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Fraction (-)")
ax.set_title("Liquid holdup and gas void fraction")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9, frameon=True)
reg(fig, "A4_holdup_gvf.png", "Holdup & GVF",
    "Liquid holdup and gas void fraction; free gas expands and holdup falls downstream.")

# A5 velocities
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
for y, lab, c in [(R["Vsl"], "Superficial liquid $V_{sl}$", 0),
                  (R["Vsg"], "Superficial gas $V_{sg}$", 1),
                  (R["Vm"], "Mixture $V_m$", 2)]:
    ax.plot(skm, y, color=style.PALETTE[c], lw=2.4, label=lab)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Velocity (m/s)")
ax.set_title("Superficial and mixture velocities")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=True)
reg(fig, "A5_velocities.png", "Velocity profiles",
    "Superficial gas velocity rises downstream as the pressure falls and gas expands.")

# A6 densities
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
ax.plot(skm, R["rho_l"], color=style.PALETTE[0], lw=2.4, label="Liquid $\\rho_l$")
ax.plot(skm, R["rho_m"], color=style.PALETTE[2], lw=2.4, label="Mixture $\\rho_m$")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(skm, R["rho_g"], color=style.PALETTE[1], lw=2.4, label="Gas $\\rho_g$")
ax2.set_ylabel("Gas density (kg/m³)", color=style.PALETTE[1])
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Liquid/mixture density (kg/m³)")
ax.set_title("Phase and mixture densities")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4,
          fontsize=9, frameon=True)
reg(fig, "A6_densities.png", "Density profiles",
    "Liquid, gas and mixture densities; mixture density falls in the gassy downstream reach.")

# A7 viscosities
fig, ax = plt.subplots(figsize=(9.8, 4.2)); zones(ax)
ax.plot(skm, R["mu_l"]*1e3, color=style.PALETTE[5], lw=2.6, label="Liquid $\\mu_l$")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(skm, R["mu_g"]*1e3, color=style.PALETTE[1], lw=2.4, label="Gas $\\mu_g$")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Liquid viscosity (cP)")
ax2.set_ylabel("Gas viscosity (cP)", color=style.PALETTE[1])
ax.set_title("Phase viscosities")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
          fontsize=9, frameon=True)
reg(fig, "A7_viscosities.png", "Viscosity profiles",
    "Live-oil and gas viscosities along the line.")

# A8 Reynolds & friction factor
fig, ax = plt.subplots(figsize=(9.8, 4.2)); zones(ax)
ax.plot(skm, R["Re"]/1e3, color=style.PALETTE[0], lw=2.6, label="Reynolds /1000")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(skm, R["f"], color=style.PALETTE[3], lw=2.4, label="Friction factor")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Re ×10³", color=style.PALETTE[0])
ax2.set_ylabel("Darcy friction factor", color=style.PALETTE[3])
ax.set_title("Reynolds number and friction factor")
reg(fig, "A8_reynolds_friction.png", "Reynolds & friction",
    "Mixture Reynolds number and Haaland friction factor along the route.")

# A9 pressure-gradient components
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
ax.plot(skm, R["dP_fric"]/1e3, color=style.PALETTE[1], lw=2.4, label="Frictional dP/ds")
ax.plot(skm, R["dP_grav"]/1e3, color=style.PALETTE[0], lw=2.4, label="Gravitational dP/ds")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Pressure gradient (kPa/m)")
ax.set_title("Pressure-gradient components"); ax.legend()
reg(fig, "A9_dpds_components.png", "Pressure-gradient split",
    "Gravity (seabed terrain) dominates the pressure gradient; friction is very small for "
    "this large-bore, low-velocity pipeline, so the pressure tracks the terrain.")

# A10 PVT along pipe
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.9))
for ax, y, lab, c in zip(axs, [R["Rs"], R["Bo"], R["Z"]],
                         ["$R_s$ (scf/stb)", "$B_o$ (rb/stb)", "Z-factor"], [0, 2, 4]):
    zones(ax); ax.plot(skm, y, color=style.PALETTE[c], lw=2.5)
    ax.set_xlabel("Arc length (km)"); ax.set_title(lab)
fig.suptitle("In-situ PVT properties along the tieback", color=style.TITLE, fontweight="bold")
reg(fig, "A10_pvt_along_pipe.png", "PVT along the pipe",
    "Solution gas-oil ratio, oil FVF and gas Z-factor along the line.")

# ============================================================ B. SLUGS ======
slug_mask = np.array([("Slug" in r or "intermittent" in r or "Bubble" in r) for r in regime])
def slugzone(ax):
    if slug_mask.any():
        ax.axvspan(skm[slug_mask].min(), skm[-1], color="#fdeccd", alpha=0.6, zorder=0,
                   label="Gas-bearing / slug zone")

# B1 slug frequency
fig, ax = plt.subplots(figsize=(9.8, 4.2)); slugzone(ax)
ax.plot(skm, R["slug_fs"], color=style.PALETTE[3], lw=2.8, marker="o", ms=3)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Slug frequency (Hz)")
ax.set_title("Hydrodynamic slug frequency (Gregory & Scott)"); ax.legend()
reg(fig, "B1_slug_frequency.png", "Slug frequency",
    "Predicted hydrodynamic slug frequency along the gas-bearing pipeline.")

# B2 slug & unit-cell length
fig, ax = plt.subplots(figsize=(9.8, 4.2)); slugzone(ax)
ax.plot(skm, R["slug_Ls"], color=style.PALETTE[0], lw=2.6, label="Slug length $L_s$")
ax.plot(skm, R["slug_Lu"], color=style.PALETTE[2], lw=2.4, label="Unit-cell length $L_u$")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Length (m)")
ax.set_title("Slug and unit-cell lengths"); ax.legend(); ax.set_ylim(bottom=0)
reg(fig, "B2_slug_length.png", "Slug & unit-cell length",
    "Mean slug-body length and unit-cell length; the film/bubble region fills the rest "
    "of each unit cell.")

# B3 translational velocity & slug holdup
fig, ax = plt.subplots(figsize=(9.8, 4.2)); slugzone(ax)
ax.plot(skm, R["slug_Vt"], color=style.PALETTE[1], lw=2.6, label="Translational velocity $V_t$ (Bendiksen)")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(skm, R["slug_HLs"], color=style.PALETTE[4], lw=2.4, label="Slug-body holdup $H_{Ls}$")
ax2.set_ylabel("Slug-body holdup (-)", color=style.PALETTE[4])
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Velocity (m/s)", color=style.PALETTE[1])
ax.set_title("Slug translational velocity and slug-body holdup")
reg(fig, "B3_translational_velocity.png", "Translational velocity & slug holdup",
    "Slug translational velocity (Bendiksen 1984) and slug-body liquid holdup (Gregory).")

# B4 flow-regime strip along pipe
labels_all = list(RM["labels"])
code = {lab: i for i, lab in enumerate(labels_all)}
codes = np.array([code.get(r, 2) for r in regime])
cmap_r = ListedColormap(["#8fd3f4", "#4fb0e0", "#ff8c42", "#7ec8a9", "#2ca02c",
                         "#c084fc", "#bcd4e6"][:len(labels_all)])
fig, ax = plt.subplots(figsize=(9.8, 2.9)); ax.grid(False)
edges = np.concatenate([[skm[0]], 0.5*(skm[1:]+skm[:-1]), [skm[-1]]])
ax.pcolormesh(edges, [0, 1], codes[None, :], cmap=cmap_r,
              vmin=-0.5, vmax=len(labels_all)-0.5)
ax.set_yticks([]); ax.set_xlabel("Arc length (km)"); ax.set_xlim(skm[0], skm[-1])
ax.set_title("Predicted flow regime along the pipeline")
present = sorted(set(codes))
import matplotlib.patches as mp
handles = [mp.Patch(color=cmap_r(c/(len(labels_all)-1)), label=labels_all[c]) for c in present]
ax.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.30), fontsize=9)
reg(fig, "B4_regime_strip.png", "Flow-regime map (spatial)",
    "Flow pattern predicted in each cell: slug / intermittent flow along essentially the "
    "whole gas-bearing pipeline.")

# B5 flow-regime MAP (Vsg-Vsl) with operating path
fig, ax = plt.subplots(figsize=(8.6, 6.4))
norm = BoundaryNorm(np.arange(-0.5, len(labels_all)+0.5), cmap_r.N)
pc = ax.pcolormesh(RM["Vsg"], RM["Vsl"], RM["Z"], cmap=cmap_r, norm=norm, shading="auto")
ax.plot(np.clip(RM["path_Vsg"], 1e-2, None), np.clip(RM["path_Vsl"], 1e-2, None),
        color="#e6194B", lw=3.0, label="Operating path")
ax.scatter(RM["path_Vsg"][-1], RM["path_Vsl"][-1], color="#e6194B", s=70, zorder=5,
           edgecolor="#ffffff", label="Outlet")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Superficial gas velocity $V_{sg}$ (m/s)")
ax.set_ylabel("Superficial liquid velocity $V_{sl}$ (m/s)")
ax.set_title("Flow-pattern map (Taitel-Dukler/Barnea) with operating path")
cb = fig.colorbar(pc, ax=ax, ticks=range(len(labels_all)))
cb.ax.set_yticklabels(labels_all)
ax.legend(loc="lower left")
reg(fig, "B5_regime_map.png", "Flow-pattern map",
    "Mechanistic flow-pattern map over the superficial-velocity plane; the operating "
    "path crosses from single-phase liquid into the slug/intermittent region.")

# B6 terrain-slugging time series (multi-panel)
tmin = SS["t"]/60.0
fig, axs = plt.subplots(3, 1, figsize=(10.0, 7.2), sharex=True)
axs[0].plot(tmin, SS["P_base"]/1e5, color=style.PALETTE[0], lw=2.0)
axs[0].set_ylabel("Sag/trap\npressure (bar)")
axs[0].set_title("Terrain (severe) slugging limit cycle at the sag")
axs[1].plot(tmin, SS["HL_riser"], color=style.PALETTE[2], lw=2.0)
axs[1].set_ylabel("Trap liquid\nfill fraction (-)")
axs[2].plot(tmin, SS["Wl_out"], color=style.PALETTE[3], lw=2.0)
axs[2].axhline(SS["Wl"], color=style.PALETTE[1], ls="--", lw=1.6, label="Mean liquid rate")
axs[2].set_ylabel("Outlet liquid\nrate (kg/s)"); axs[2].set_xlabel("Time (min)"); axs[2].legend()
reg(fig, "B6_severe_slugging_timeseries.png", "Terrain-slugging transient",
    "Terrain-slugging limit cycle at the sag: cyclic trap-pressure swings, liquid "
    "accumulation/blow-through and intense outlet liquid surges.")

# B7 terrain-slugging phase portrait
fig, ax = plt.subplots(figsize=(6.6, 5.6))
ax.plot(SS["HL_riser"], SS["P_base"]/1e5, color=style.PALETTE[4], lw=1.6)
ax.set_xlabel("Trap liquid fill fraction (-)"); ax.set_ylabel("Sag pressure (bar)")
ax.set_title("Terrain-slugging phase portrait")
reg(fig, "B7_severe_slug_phase.png", "Terrain-slugging phase portrait",
    "Closed limit-cycle orbit in the fill-pressure plane confirms a self-sustained "
    "terrain-slugging cycle once triggered.")

# B8 hydrodynamic slug train space-time holdup contour (zoomed for clarity)
xkm = TR["x"]/1000.0
x0 = xkm[len(xkm)//2]                       # window centred mid-pipeline
win = (xkm >= x0) & (xkm <= x0 + 1.5)       # 1.5 km window
tw = TR["t"] <= 6.0/float(TR["fs"])          # ~6 slug periods
fig, ax = plt.subplots(figsize=(9.6, 5.0))
ax.grid(False)
ext = [xkm[win][0], xkm[win][-1], TR["t"][tw][0], TR["t"][tw][-1]]
im = ax.imshow(TR["HL"][np.ix_(tw, win)], aspect="auto", origin="lower", extent=ext,
               cmap=style.CMAP_HOLD, vmin=0, vmax=1)
ax.set_xlabel("Position along pipeline (km)"); ax.set_ylabel("Time (s)")
ax.set_title("Hydrodynamic slug train - space-time liquid holdup")
cb = fig.colorbar(im, ax=ax); cb.set_label("Liquid holdup $H_L$ (-)")
ax.annotate("liquid slugs travel\nat $V_t$ (slope)", (ext[0]+0.15, ext[3]*0.7),
            color="#ffffff", fontsize=9)
reg(fig, "B8_slug_train_spacetime.png", "Slug-train space-time map",
    "Kinematic slug tracking over a 1.5 km window: liquid slugs (yellow, high holdup) "
    "separated by gas films (blue) travel at the translational velocity - the space-time "
    "signature of hydrodynamic slugging.")

# B9 riser-base pressure spectrum
sig = TR["P_base"] - TR["P_base"].mean()
dt = TR["t"][1]-TR["t"][0]
freq = np.fft.rfftfreq(len(sig), dt); psd = np.abs(np.fft.rfft(sig))**2
fig, ax = plt.subplots(figsize=(8.8, 4.2))
ax.plot(freq, psd/psd.max(), color=style.PALETTE[5], lw=2.2)
ax.axvline(TR["fs"], color=style.PALETTE[1], ls="--", lw=1.8,
           label=f"Slug frequency ≈ {float(TR['fs']):.3f} Hz")
ax.set_xlim(0, max(0.5, 4*float(TR["fs"]))); ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Normalised PSD"); ax.set_title("In-line slug pressure spectrum")
ax.legend()
reg(fig, "B9_pressure_spectrum.png", "Slug pressure spectrum",
    "Power spectrum of the in-line pressure signal, peaking at the predicted slug "
    "frequency.")

# ============================================================ C. HYDRATE ====
# C1 hydrate phase envelope P-T with operating path (KEY)
Tk = np.linspace(274, 305, 120)
Peq = []
for t in Tk:                                    # invert Teq(P) for the envelope
    from scipy.optimize import brentq
    f = lambda P: U.hydrate_Teq(P, cfg.gas_grav, cfg.C_hy) - t
    try:
        Peq.append(brentq(f, 1e5, 400e5))
    except Exception:
        Peq.append(np.nan)
Peq = np.array(Peq)
fig, ax = plt.subplots(figsize=(8.8, 6.2))
ax.plot(Tk-273.15, Peq/1e5, color=style.PALETTE[0], lw=3.0, label="Hydrate envelope (no inhibitor)")
ax.fill_betweenx(Peq/1e5, Tk-273.15, 274-273.15, color="#cfe0ff", alpha=0.5)
pts = ax.scatter(R["T"]-273.15, R["P"]/1e5, c=R["hy_subcool"], cmap=style.CMAP_RISK,
                 s=26, label="Operating path", zorder=4)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Pressure (bar)")
ax.set_title("Hydrate phase envelope with the operating path")
cb = fig.colorbar(pts, ax=ax); cb.set_label("Sub-cooling (°C)")
ax.annotate("HYDRATE\nREGION", (276-273.15+0.5, 120), color=style.PALETTE[0], fontsize=11,
            fontweight="bold", ha="left")
ax.legend(loc="lower right")
reg(fig, "C1_hydrate_envelope.png", "Hydrate phase envelope",
    "Operating path on the pressure-temperature plane; points left of the envelope "
    "(warm colours) are inside the hydrate-stability region.")

# C2 MEG mitigation - envelope shift
fig, ax = plt.subplots(figsize=(8.8, 6.0))
for meg, c in zip([0, 20, 35, 50], [0, 1, 2, 3]):
    dT = U.hammerschmidt_depression(meg, cfg.MEG_MW)
    ax.plot(Tk-273.15-dT, Peq/1e5, color=style.PALETTE[c], lw=2.6,
            label=f"MEG {meg} wt%  (ΔT={dT:.1f}°C)")
ax.scatter(R["T"]-273.15, R["P"]/1e5, color=style.PALETTE[5], s=14, alpha=0.7,
           label="Operating path")
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Pressure (bar)")
ax.set_title("Hydrate envelope shift with MEG inhibition")
ax.legend(loc="lower right")
reg(fig, "C2_meg_envelope_shift.png", "MEG envelope shift",
    "MEG injection shifts the hydrate envelope to lower temperatures; at ~45-50 wt% the "
    "operating path clears the envelope.")

# C3 MEG sweep
meg = pd.read_csv(os.path.join(common.DIRS["solution"], "meg_sweep.csv"))
fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.plot(meg["MEG_wt_pct"], meg["max_subcooling_C"], "-o", color=style.PALETTE[3],
        label="Max sub-cooling")
ax.axhline(0, color=style.PALETTE[0], ls="--", lw=1.6)
ax.set_xlabel("MEG concentration (wt%)"); ax.set_ylabel("Max sub-cooling (°C)", color=style.PALETTE[3])
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(meg["MEG_wt_pct"], 100*meg["hydrate_zone_frac"], "-s", color=style.PALETTE[2],
         label="Hydrate zone (% line)")
ax2.set_ylabel("Hydrate zone (% of line)", color=style.PALETTE[2])
ax.set_title("Hydrate mitigation versus MEG dosage")
reg(fig, "C3_meg_sweep.png", "MEG dosage sweep",
    "Maximum sub-cooling and in-envelope line fraction fall with MEG dosage; ~45 wt% "
    "fully suppresses hydrate risk.")

# C4 hydrate risk heat-strip along pipe
fig, ax = plt.subplots(figsize=(9.8, 2.7)); ax.grid(False)
sc = ax.scatter(skm, np.ones_like(skm), c=R["hy_risk"], cmap=style.CMAP_RISK,
                vmin=0, vmax=1, marker="s", s=90)
ax.set_yticks([]); ax.set_xlabel("Arc length (km)")
ax.set_title("Hydrate-risk index along the pipeline")
cb = fig.colorbar(sc, ax=ax, orientation="vertical"); cb.set_label("Risk (0-1)")
reg(fig, "C4_risk_strip.png", "Hydrate-risk strip",
    "Spatial hydrate-risk index; the cold downstream reach is the high-risk segment.")

# B10 terrain-slug susceptibility vs turndown
td = pd.read_csv(os.path.join(common.DIRS["solution"], "terrain_slug_susceptibility.csv"))
fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.plot(td["turndown_pct"], td["Boe_number_PI"], "-o", color=style.PALETTE[0], lw=2.6)
ax.axhline(1.0, color=style.PALETTE[3], ls="--", lw=1.8, label="Severe-slugging threshold ($\\Pi_{ss}$ = 1)")
ax.fill_between(td["turndown_pct"], 0, 1, color="#ffd9c9", alpha=0.5)
ax.set_ylim(-0.4, td["Boe_number_PI"].max()*1.12)
ax.text(72, 1.9, "severe terrain-slugging region (below threshold)",
        color=style.PALETTE[1], fontsize=9.5, ha="center")
ax.annotate("", (72, 0.5), (72, 1.7), arrowprops=dict(arrowstyle="->", color=style.PALETTE[1]))
ax.set_xlabel("Throughput (% of design)"); ax.set_ylabel("Boe number  $\\Pi_{ss}$ (-)")
ax.set_title("Terrain-slugging susceptibility vs turndown"); ax.legend()
reg(fig, "B10_turndown_susceptibility.png", "Terrain-slug susceptibility",
    "Boe terrain-slugging number versus throughput: hydrodynamic slugging at design "
    "rate, crossing into the severe terrain-slugging regime (PI<1) at deep turndown.")

# ---- B11 multi-timescale slug pressure time histories ----------------------
fig, axs = plt.subplots(2, 1, figsize=(10.0, 6.6))
sig = TR["P_base"]/1e5 - TR["P_base"].mean()/1e5 + R["P"][i_sag]/1e5
axs[0].plot(TR["t"], sig, color=style.PALETTE[0], lw=1.8)
axs[0].set_ylabel("In-line pressure (bar)")
axs[0].set_xlabel("Time (s)  -  hydrodynamic slugging")
axs[0].set_xlim(0, min(TR["t"][-1], 8/float(TR["fs"])))
axs[0].set_title("Pressure time histories - hydrodynamic (fast) & terrain (slow) slugging")
axs[1].plot(SS["t"]/60.0, SS["P_base"]/1e5, color=style.PALETTE[3], lw=2.0)
axs[1].set_ylabel("Sag pressure (bar)"); axs[1].set_xlabel("Time (min)  -  terrain slugging")
reg(fig, "B11_time_histories.png", "Slug pressure time histories",
    "Pressure time histories at monitoring points: fast hydrodynamic slug pulses (top, "
    "seconds) and the slow, large-amplitude terrain-slugging cycle (bottom, minutes).")

# ---- C5 shutdown cool-down temperature field (time-distance contour) -------
CD = dict(np.load(os.path.join(DD, "cooldown.npz")))
fig, ax = plt.subplots(figsize=(10.0, 4.8))
ext = [CD["s"][0]/1000, CD["s"][-1]/1000, CD["t"][0]/3600, CD["t"][-1]/3600]
im = ax.imshow(CD["Tfield"]-273.15, aspect="auto", origin="lower", extent=ext, cmap=style.CMAP_TEMP)
cs = ax.contour(np.linspace(ext[0], ext[1], CD["Tfield"].shape[1]),
                np.linspace(ext[2], ext[3], CD["Tfield"].shape[0]),
                CD["subcool"], levels=[0], colors=[style.INK], linewidths=2.2)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Time after shutdown (h)")
ax.set_title("Shutdown cool-down: temperature field (navy line = hydrate onset)")
cb = fig.colorbar(im, ax=ax); cb.set_label("Temperature (°C)")
reg(fig, "C5_cooldown_field.png", "Shutdown cool-down field",
    "Pipeline temperature after an unplanned shutdown; the navy contour is the moment "
    "each section enters the hydrate region at settle-out pressure.")

# ---- C6 time-to-hydrate after shutdown -------------------------------------
fig, ax = plt.subplots(figsize=(9.8, 4.4))
ax.plot(CD["s"]/1000, CD["tth_h"], color=style.PALETTE[1], lw=2.8)
ax.fill_between(CD["s"]/1000, 0, CD["tth_h"], color="#ffe3c2", alpha=0.6)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Time-to-hydrate (h)")
ax.set_title("No-touch time after shutdown (time-to-hydrate)")
ax.annotate("sections already in the hydrate\nregion during flow -> 0 h",
            (s_sag, 0.2), color=style.PALETTE[3], fontsize=8.5)
reg(fig, "C6_time_to_hydrate.png", "No-touch time",
    "Time for each pipeline section to enter the hydrate region after shutdown - the "
    "available intervention window before remediation (MEG bullheading / depressurisation).")

# ---- C7 cool-down temperature time histories at monitoring points ----------
Teq_settle = U.hydrate_Teq(float(CD["Psettle"]), cfg.gas_grav, cfg.C_hy) - 273.15
th = CD["t"]/3600.0; ns = CD["Tfield"].shape[1]
mons = {"Inlet (0 km)": 0, "Mid-line": ns//2, "Outlet": ns-1}
fig, ax = plt.subplots(figsize=(9.8, 5.0))
for (lab, j), c in zip(mons.items(), [0, 2, 3]):
    ax.plot(th, CD["Tfield"][:, j]-273.15, color=style.PALETTE[c], lw=2.6, label=lab)
ax.axhline(Teq_settle, color=style.PALETTE[1], ls="--", lw=2.0,
           label=f"Hydrate $T_{{eq}}$ @ settle-out ({Teq_settle:.0f} °C)")
ax.fill_between(th, ax.get_ylim()[0], Teq_settle, color="#dbe7ff", alpha=0.4)
ax.set_xlabel("Time after shutdown (h)"); ax.set_ylabel("Temperature (°C)")
ax.set_title("Shutdown cool-down temperature histories (monitoring points)")
ax.legend(loc="upper right")
reg(fig, "C7_cooldown_histories.png", "Cool-down temperature histories",
    "Temperature time histories at three monitoring points after shutdown; once a curve "
    "crosses below the settle-out hydrate equilibrium temperature (shaded), that location "
    "is inside the hydrate region.")

# ---- C8 thermal time constant along the pipeline (cooldown CSV) ------------
cdf = pd.read_csv(os.path.join(common.DIRS["solution"], "shutdown_cooldown.csv"))
fig, ax = plt.subplots(figsize=(9.8, 4.4)); zones(ax)
ax.plot(cdf["s_km"], cdf["tau_h"], color=style.PALETTE[4], lw=2.8)
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Thermal time constant $\\tau$ (h)")
ax.set_title("Per-section thermal time constant (cool-down rate)")
ax.legend(loc="upper right")
reg(fig, "C8_time_constant.png", "Thermal time constant",
    "Per-section lumped thermal time constant tau = rho*cp*A/(U*pi*D); larger tau means "
    "slower cool-down and a longer no-touch time after shutdown.")

print("Stage 05 part 1 (profiles, slugs, hydrate) complete.")

# ============================================ D. CONTOURS / FIELDS / 3D ======
exec(open(os.path.join(os.path.dirname(__file__), "run_fields.py")).read())
print("Stage 05 post-processing complete.")
