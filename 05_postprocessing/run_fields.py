# -*- coding: utf-8 -*-
# Contours, 2-D / 3-D fields and velocity vectors for the SUBSEA PIPELINE.
# Executed within run.py (shares its namespace: R, SS, TR, cfg, style, common, P,
# reg, zones, skm, s, i_sag, s_sag, U, np, plt, LineCollection, Normalize, Axes3D).
import matplotlib.cm as cmx
from matplotlib.cm import ScalarMappable

# ---- route coordinates (seabed pipeline) ----------------------------------
elev = R["elev"]
xr = s.copy()                                   # near-horizontal seabed route
yr = 60.0*np.sin(np.pi*s/cfg.L_pipe)
depth = cfg.water_depth - elev                  # depth below sea surface
from matplotlib.ticker import MaxNLocator

def fix3d(ax, npad=16):
    """Keep 3-D axis labels clear of tick numbers (no text overlap), readable fonts."""
    ax.xaxis.labelpad = npad; ax.yaxis.labelpad = npad; ax.zaxis.labelpad = npad
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(5))
    ax.tick_params(pad=5, labelsize=9.5)

def color_line(xv, yv, cv, cmap, label, title, fname, cap, xlab, ylab, invert_y=False):
    pts = np.array([xv, yv]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(cv.min(), cv.max()), linewidth=7)
    lc.set_array(cv)
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.add_collection(lc); ax.autoscale()
    if invert_y: ax.invert_yaxis()
    cb = fig.colorbar(lc, ax=ax); cb.set_label(label)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    reg(fig, fname, title, cap)

# ---- D1 pressure field along the route ------------------------------------
color_line(xr/1000, depth, R["P"]/1e5, style.CMAP_PRES, "Pressure (bar)",
           "Pressure field mapped on the pipeline seabed profile", "D1_pressure_route.png",
           "Pressure mapped onto the physical seabed profile; the pressure tracks the "
           "seabed terrain (hydrostatic head). Vertical scale (depth) is exaggerated.",
           "Arc length (km)", "Depth below surface (m)", invert_y=True)

# ---- D2 temperature field along the route ----------------------------------
color_line(xr/1000, depth, R["T"]-273.15, style.CMAP_TEMP, "Temperature (°C)",
           "Temperature field mapped on the pipeline seabed profile", "D2_temperature_route.png",
           "Temperature mapped onto the seabed profile; the fluid cools to near-seabed "
           "temperature downstream (the hydrate-prone reach).",
           "Arc length (km)", "Depth below surface (m)", invert_y=True)

# ---- D3 radial temperature contour through pipe wall + insulation ----------
r_in = cfg.D/2; r_wall = r_in+0.018; r_ins = r_wall+0.06
rgrid = np.linspace(0, r_ins, 60)
Sgrid, Rgrid = np.meshgrid(skm, rgrid)
Tfield = np.zeros_like(Sgrid)
for j in range(len(skm)):
    Tf = R["T"][j]-273.15; Ta = R["Tamb"][j]-273.15
    for i, rr in enumerate(rgrid):
        if rr <= r_in: Tfield[i, j] = Tf
        elif rr <= r_ins: Tfield[i, j] = Tf + (Ta-Tf)*((rr-r_in)/(r_ins-r_in))**1.4
        else: Tfield[i, j] = Ta
fig, ax = plt.subplots(figsize=(10.0, 4.8)); ax.grid(False)
cf = ax.contourf(Sgrid, Rgrid*1000, Tfield, levels=24, cmap=style.CMAP_TEMP)
ax.contour(Sgrid, Rgrid*1000, Tfield, levels=10, colors="#ffffff", linewidths=0.5, alpha=0.6)
ax.axhline(r_in*1000, color=style.INK, lw=1.2, ls="--")
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Radius from pipe axis (mm)")
ax.set_title("Radial temperature field (fluid → wall → insulation)")
cb = fig.colorbar(cf, ax=ax); cb.set_label("Temperature (°C)")
reg(fig, "D3_temperature_radial.png", "Radial temperature contour",
    "Two-dimensional temperature field through the pipe wall and insulation along the "
    "route (dashed line = inner pipe wall).")

# ---- D4 space-time slug pressure-FLUCTUATION contour (zoomed, clear) -------
xz = TR["x"]; tz = TR["t"]; HLst = TR["HL"]
imid = i_sag
rho_l = float(R["rho_l"][imid]); Vm = float(R["Vm"][imid])
# slug body -> local pressure spike; show the fluctuation about the mean profile
dPfield = 1.2 * (HLst - HLst.mean(axis=0, keepdims=True))   # normalised slug pressure pulse, bar
xkm = xz/1000.0; x0 = xkm[len(xkm)//2]
win = (xkm >= x0) & (xkm <= x0 + 1.5)
tw = tz <= 6.0/float(TR["fs"])
fig, ax = plt.subplots(figsize=(9.8, 5.0)); ax.grid(False)
ext = [xkm[win][0], xkm[win][-1], tz[tw][0], tz[tw][-1]]
im = ax.imshow(dPfield[np.ix_(tw, win)], aspect="auto", origin="lower", extent=ext,
               cmap=style.CMAP_PRES)
ax.set_xlabel("Position along pipeline (km)"); ax.set_ylabel("Time (s)")
ax.set_title("Slug-induced pressure fluctuation (space-time)")
cb = fig.colorbar(im, ax=ax); cb.set_label("Pressure fluctuation $\\Delta P'$ (bar)")
ax.annotate("pressure pulses travel\nwith the slugs", (ext[0]+0.12, ext[3]*0.72),
            color="#ffffff", fontsize=9)
reg(fig, "D4_pressure_spacetime.png", "Space-time pressure contour",
    "Slug-induced pressure fluctuation (deviation from the mean profile) over a 1.5 km "
    "window: each passing liquid slug produces a travelling pressure pulse.")

# ---- D5-D7 3-D route coloured by field (seabed band, labelled) -------------
def route3d(cv, cmap, label, title, fname, cap):
    fig = plt.figure(figsize=(10.0, 6.6)); ax = fig.add_subplot(111, projection="3d")
    # seabed context plane at the deepest terrain
    xx, yy = np.meshgrid(np.linspace(xr.min()/1000, xr.max()/1000, 2),
                         np.linspace(yr.min(), yr.max(), 2))
    ax.plot_surface(xx, yy, np.full_like(xx, depth.max()+8), color="#e7d6b8",
                    alpha=0.30, shade=False)
    pts = np.array([xr/1000, yr, depth]).T.reshape(-1, 1, 3)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=cmap, norm=Normalize(cv.min(), cv.max()), lw=5)
    lc.set_array(cv); ax.add_collection3d(lc)
    ax.scatter([xr[0]/1000], [yr[0]], [depth[0]], color=style.PALETTE[2], s=55, depthshade=False)
    ax.scatter([xr[-1]/1000], [yr[-1]], [depth[-1]], color=style.PALETTE[1], s=55, depthshade=False)
    ax.scatter([xr[i_sag]/1000], [yr[i_sag]], [depth[i_sag]], color=style.PALETTE[3], s=50, depthshade=False)
    _bx = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.80)
    ax.text(xr[0]/1000-0.6, yr[0]-6, depth[0]-22, "inlet", color=style.PALETTE[2],
            fontsize=9.5, fontweight="bold", bbox=_bx)
    ax.text(xr[-1]/1000+0.9, yr[-1]+6, depth[-1]+4, "host", color=style.PALETTE[1],
            fontsize=9.5, fontweight="bold", bbox=_bx)
    ax.text(xr[i_sag]/1000-1.0, yr[i_sag]-8, depth[i_sag]-30, "sag", color=style.PALETTE[3],
            fontsize=9.5, fontweight="bold", bbox=_bx)
    ax.set_xlim(xr.min()/1000, xr.max()/1000); ax.set_ylim(yr.min(), yr.max())
    ax.set_zlim(depth.max()+10, depth.min()-10)
    ax.view_init(elev=22, azim=-62); fix3d(ax)
    ax.set_xlabel("Distance (km)"); ax.set_ylabel("Plan offset (m)"); ax.set_zlabel("Depth (m)")
    ax.set_title(title)
    cb = fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.12); cb.set_label(label)
    reg(fig, fname, title, cap)

route3d(R["P"]/1e5, style.CMAP_PRES, "Pressure (bar)",
        "3-D pipeline route coloured by pressure", "D5_route3d_pressure.png",
        "Seabed pipeline coloured by pressure (depth exaggerated to reveal the terrain).")
route3d(R["T"]-273.15, style.CMAP_TEMP, "Temperature (°C)",
        "3-D pipeline route coloured by temperature", "D6_route3d_temperature.png",
        "Seabed pipeline coloured by temperature; the cold downstream reach is the "
        "hydrate-prone segment.")
route3d(R["hy_subcool"], style.CMAP_RISK, "Sub-cooling (°C)",
        "3-D pipeline route coloured by hydrate sub-cooling", "D7_route3d_subcooling.png",
        "Seabed pipeline coloured by hydrate sub-cooling (warm = inside the envelope).")

# ---- D8 3-D surface: slug length over (Vsg, Vsl) ---------------------------
Vsg = np.logspace(-1, 1.2, 45); Vsl = np.logspace(-1, 0.6, 45)
G, L = np.meshgrid(Vsg, Vsl); Lsurf = np.zeros_like(G)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        sc = U.slug_characteristics(cfg, G[i, j], L[i, j], np.deg2rad(2.0),
                                    rho_l, float(R["rho_g"][imid]), float(R["mu_l"][imid]))
        Lsurf[i, j] = min(sc["Lu"] if sc["Lu"] == sc["Lu"] else 0, 500)
fig = plt.figure(figsize=(9.2, 6.4)); ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(np.log10(G), np.log10(L), Lsurf, cmap=style.CMAP_SEQ, linewidth=0)
ax.set_xlabel("log₁₀ $V_{sg}$"); ax.set_ylabel("log₁₀ $V_{sl}$"); ax.set_zlabel("Unit-cell length (m)")
ax.set_title("Slug unit-cell length over the operating envelope"); ax.view_init(26, -52); fix3d(ax)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1).set_label("Unit-cell length (m)")
reg(fig, "D8_sluglength_surface.png", "Slug-length response surface",
    "Predicted unit-cell length as a 3-D response surface over the superficial-velocity "
    "operating envelope.")

# ---- D9 3-D surface: hydrate sub-cooling over (P, T) -----------------------
Pp = np.linspace(20e5, 200e5, 50); Tt = np.linspace(276, 300, 50)
GP, GT = np.meshgrid(Pp, Tt)
Sub = np.array([[U.hydrate_Teq(p, cfg.gas_grav, cfg.C_hy)-t for p in Pp] for t in Tt])
fig = plt.figure(figsize=(9.2, 6.4)); ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(GP/1e5, GT-273.15, Sub, cmap=style.CMAP_RISK, linewidth=0)
ax.contour(GP/1e5, GT-273.15, Sub, levels=[0], colors=[style.PALETTE[0]], linewidths=3)
ax.set_xlabel("Pressure (bar)"); ax.set_ylabel("Temperature (°C)"); ax.set_zlabel("Sub-cooling (°C)")
ax.set_title("Hydrate sub-cooling surface ($\\Delta T_{sub}=T_{eq}-T$)"); ax.view_init(24, -130); fix3d(ax)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1).set_label("Sub-cooling (°C)")
reg(fig, "D9_subcooling_surface.png", "Hydrate sub-cooling surface",
    "Hydrate sub-cooling as a 3-D surface over pressure and temperature; the blue "
    "contour is the hydrate-formation boundary.")

# ---- D10 3-D terrain-slugging field (trap position x time) -----------------
hh = np.linspace(0, SS["z"].max() if "z" in SS else 100, 60)
tt = SS["t"][::40]
Pss = np.zeros((len(tt), len(hh)))
for a, ta in enumerate(tt):
    idx = np.argmin(np.abs(SS["t"]-ta)); zlv = SS["z"][idx]
    for b, h in enumerate(hh):
        Pss[a, b] = (R["P"][i_sag] + rho_l*U.g*max(zlv-h, 0))/1e5
Hh, Tt2 = np.meshgrid(hh, tt/60.0)
fig = plt.figure(figsize=(9.4, 6.4)); ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(Hh, Tt2, Pss, cmap=style.CMAP_PRES, linewidth=0)
ax.set_xlabel("Height in uphill leg (m)"); ax.set_ylabel("Time (min)"); ax.set_zlabel("Pressure (bar)")
ax.set_title("Terrain-slugging pressure field (uphill leg × time)"); ax.view_init(26, -58); fix3d(ax)
fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1).set_label("Pressure (bar)")
reg(fig, "D10_severe_slug_surface.png", "Terrain-slugging 3-D field",
    "Pressure in the sag uphill leg versus height and time over several terrain-slugging "
    "cycles (build-up then blow-through).")

# ===================================================== E. 3-D RESOLVED FIELD =
# Genuine 3-D fields from the resolved cross-section (r, phi) x axial backbone.

# ---- E1 resolved cross-section: axial velocity + secondary-flow vectors ----
fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.4))
cases = [(0.78, "Slug body  ($H_L$=0.78)"), (0.30, "Film/bubble  ($H_L$=0.30)")]
vmax = 0
fields = []
for HLloc, lab in cases:
    cs = U.cross_section_field(cfg, HLloc, R["Vsl"][imid], R["Vsg"][imid], np.deg2rad(2))
    fields.append((cs, lab)); vmax = max(vmax, cs["W"].max())
for ax, (cs, lab) in zip(axs, fields):
    a = cs["Xc"]; b = cs["Yc"]
    tcf = ax.tricontourf(a.ravel(), b.ravel(), cs["W"].ravel(), levels=22, cmap=style.CMAP_SEQ,
                         vmin=0, vmax=vmax)
    th = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(th), np.sin(th), color=style.INK, lw=2)
    ax.axhline(cs["h"], color=style.PALETTE[3], lw=1.6, ls="--")
    sk = (slice(None, None, 3), slice(None, None, 2))
    ax.quiver(a[sk], b[sk], cs["Us"][sk], cs["Vs"][sk], color=style.INK, scale=8, width=0.005)
    ax.set_aspect("equal"); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(lab)
sm = ScalarMappable(norm=Normalize(0, vmax), cmap=style.CMAP_SEQ)
cb = fig.colorbar(sm, ax=axs, shrink=0.85); cb.set_label("Axial velocity (m/s)")
fig.suptitle("Resolved pipe cross-section: axial velocity & secondary flow",
             color=style.TITLE, fontweight="bold")
reg(fig, "E1_cross_section_field.png", "Resolved cross-section field",
    "Resolved 2-D pipe cross-section (gas above the interface, liquid below): filled "
    "contours of axial velocity with in-plane secondary-flow vectors, for a slug body "
    "(left) and a film region (right). Dashed line = gas-liquid interface.")

# ---- E2 3-D slug structure over a pipe segment ----------------------------
Ls = float(TR["Ls"]); Lu = float(TR["Lu"]) if float(TR["Lu"]) == float(TR["Lu"]) else 6*Ls
HLs = float(TR["HLs"])
seg_len = min(4.0*Lu, 220.0); nax = 220
xseg = np.linspace(0, seg_len, nax)
# explicit slug train: slug bodies (high holdup) separated by gas films
HLx = np.where((xseg % Lu) < Ls, HLs, 0.15)
nph = 60; phi = np.linspace(0, 2*np.pi, nph)
Rp = cfg.D/2
Xg, Pg = np.meshgrid(xseg, phi)
Yg = Rp*np.cos(Pg); Zg = Rp*np.sin(Pg)
HLg = np.tile(HLx, (nph, 1))
fig = plt.figure(figsize=(11.5, 5.6)); ax = fig.add_subplot(111, projection="3d")
norm = Normalize(0.1, max(0.85, HLs)); colors = cmx.get_cmap(style.CMAP_HOLD)(norm(HLg))
ax.plot_surface(Xg, Yg, Zg, facecolors=colors, rstride=2, cstride=2, linewidth=0,
                antialiased=True, shade=False)
ax.set_box_aspect((3.2, 1, 1))
ax.set_xlabel("Distance along pipe (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title("3-D slug structure over a pipe segment (colour = liquid holdup)")
ax.view_init(elev=20, azim=-72); fix3d(ax, npad=18)
sm = ScalarMappable(norm=norm, cmap=style.CMAP_HOLD)
fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.10).set_label("Liquid holdup $H_L$ (-)")
reg(fig, "E2_slug_structure_3d.png", "3-D slug structure",
    f"Three-dimensional view of a {seg_len:.0f} m pipe segment: alternating liquid slugs "
    f"(high holdup, Ls≈{Ls:.0f} m) and gas films (low holdup) - the resolved slug train in 3-D.")

# ---- E3 resolved axial-velocity in successive cross-sections (unit cell) ----
fig = plt.figure(figsize=(10.8, 6.2)); ax = fig.add_subplot(111, projection="3d")
HLseq = [0.80, 0.72, 0.55, 0.35, 0.18, 0.35, 0.55, 0.72]   # through one slug unit cell
labels_stage = ["slug", "", "", "", "film", "", "", "slug"]
spacing = 2.6; vmax = 0
data = []
for HLk in HLseq:
    cs = U.cross_section_field(cfg, HLk, float(R["Vsl"][imid]), float(R["Vsg"][imid]), 0.0)
    data.append(cs); vmax = max(vmax, cs["W"].max())
norm = Normalize(0, vmax)
for k, cs in enumerate(data):
    xk = np.full_like(cs["W"], k*spacing)
    sc = ax.scatter(xk, cs["Yc"], cs["Xc"], c=cs["W"], cmap=style.CMAP_SEQ, norm=norm,
                    s=10, depthshade=False)
    if labels_stage[k]:
        ax.text(k*spacing, 0, 1.3, labels_stage[k], color=style.INK, fontsize=9, ha="center")
ax.set_box_aspect((2.4, 1, 1))
ax.set_xlabel("Cross-section station (through a slug unit cell)")
ax.set_ylabel("y / R (-)"); ax.set_zlabel("z / R (-)")
ax.set_title("Resolved axial-velocity in successive cross-sections")
ax.set_xticks([]); ax.view_init(elev=16, azim=-72); fix3d(ax)
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).set_label("Axial velocity (m/s)")
reg(fig, "E3_velocity_slices_3d.png", "3-D velocity slices",
    "Resolved pipe cross-sections at successive stations through one slug unit cell "
    "(slug body -> film -> slug body): each disk shows the axial-velocity field, with the "
    "fast gas occupying the upper part of the film sections.")

# ---- E4 3-D temperature field on the pipe tube (exaggerated, visible) -------
# The true 0.4 m pipe is invisible over a 16 km route, so the tube is drawn with
# an exaggerated radius (a fixed fraction of each axis) to make the temperature
# field a clear, appreciable 3-D surface from the hot inlet to the cold host.
elevr = R["elev"]
Tc = R["T"] - 273.15
nph = 44; phi = np.linspace(0, 2*np.pi, nph)
ry = 0.085*(yr.max() - yr.min() + 1e-9)          # visual tube radius (plan)
rz = 0.085*(elevr.max() - elevr.min() + 1e-9)    # visual tube radius (elevation)
Xt = np.tile((xr/1000)[:, None], (1, nph))
Yt = yr[:, None] + ry*np.cos(phi)[None, :]
Zt = elevr[:, None] + rz*np.sin(phi)[None, :]
norm = Normalize(Tc.min(), Tc.max())
colors = cmx.get_cmap(style.CMAP_TEMP)(norm(np.tile(Tc[:, None], (1, nph))))
fig = plt.figure(figsize=(10.8, 6.2)); ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xt, Yt, Zt, facecolors=colors, rstride=2, cstride=1, linewidth=0,
                shade=False, antialiased=True)
ax.text(xr[0]/1000, yr[0], elevr[0]+rz+10, "inlet (hot)", color=style.PALETTE[1],
        fontsize=9, ha="center")
ax.text(xr[-1]/1000, yr[-1], elevr[-1]+rz+10, "host (cold)", color=style.PALETTE[0],
        fontsize=9, ha="center")
ax.text(xr[i_sag]/1000, yr[i_sag], elevr[i_sag]-rz-12, "sag", color=style.PALETTE[3],
        fontsize=9, ha="center")
ax.set_box_aspect((3.2, 1, 1))
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Plan offset (m)"); ax.set_zlabel("Elevation (m)")
ax.set_title("3-D temperature field on the pipeline (tube radius exaggerated)")
ax.view_init(elev=20, azim=-62); fix3d(ax)
ax.yaxis.labelpad = 10                            # pull plan-offset label clear of colorbar
sm = ScalarMappable(norm=norm, cmap=style.CMAP_TEMP)
fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.16).set_label("Temperature (°C)")
reg(fig, "E4_temperature_3d.png", "3-D temperature field",
    "Three-dimensional temperature field rendered on the pipeline as a tube (radius "
    "exaggerated for visibility): the fluid enters hot at the inlet and cools to near "
    "the seabed temperature toward the host - the cold downstream reach is hydrate-prone.")

# ---- D12 velocity quiver along the route -----------------------------------
fig, ax = plt.subplots(figsize=(10.0, 5.0))
step = max(1, len(s)//40)
theta_s = np.array([U._theta_of_s(cfg, si) for si in s])
xi = xr[::step]/1000; zi = depth[::step]; ti = theta_s[::step]; Vmi = R["Vm"][::step]
u = np.cos(ti)*Vmi; w = -np.sin(ti)*Vmi
q = ax.quiver(xi, zi, u, w, Vmi, cmap=style.CMAP_SEQ, scale=18, width=0.004)
ax.plot(xr/1000, depth, color=style.INK_SOFT, lw=1.0, alpha=0.5)
ax.invert_yaxis()
ax.set_xlabel("Arc length (km)"); ax.set_ylabel("Depth below surface (m)")
ax.set_title("Mixture-velocity vector field along the pipeline")
cb = fig.colorbar(q, ax=ax); cb.set_label("Mixture velocity (m/s)")
reg(fig, "D12_velocity_quiver_route.png", "Velocity vector field (route)",
    "Mixture-velocity vectors tangent to the seabed route, coloured by magnitude; flow "
    "accelerates downstream as gas expands.")

# ---- D13 3-D velocity vectors along the route ------------------------------
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection
_turbo = cmx.get_cmap("turbo")
CMAP_VEL = LinearSegmentedColormap.from_list(                 # bright turbo, dark low end dropped
    "turbo_bright", _turbo(np.linspace(0.22, 1.0, 256)))
fig = plt.figure(figsize=(10.6, 6.8)); ax = fig.add_subplot(111, projection="3d")
# seabed context plane (same style as the D5-D7 route plots)
xx, yy = np.meshgrid(np.linspace(xr.min()/1000, xr.max()/1000, 2),
                     np.linspace(yr.min(), yr.max(), 2))
ax.plot_surface(xx, yy, np.full_like(xx, depth.max()+8), color="#e7d6b8",
                alpha=0.30, shade=False)
# route coloured by mixture velocity (bright colours only)
pts = np.array([xr/1000, yr, depth]).T.reshape(-1, 1, 3)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = Line3DCollection(segs, cmap=CMAP_VEL, norm=Normalize(R["Vm"].min(), R["Vm"].max()), lw=5)
lc.set_array(R["Vm"]); ax.add_collection3d(lc)
# velocity arrows tangent to the route: evenly spaced, point downstream, tilt with slope
step2 = max(1, len(s)//12)
xi = xr[::step2]/1000; yi = yr[::step2]; di = depth[::step2]
ti2 = np.clip(theta_s[::step2], -np.deg2rad(8), np.deg2rad(8))   # cap tilt so arrows stay tidy
aL = 0.85                                                     # arrow length along route (km units)
u = np.cos(ti2)*aL; v = np.zeros_like(u); w = -np.sin(ti2)*aL*40   # depth exaggerated to match z
ax.quiver(xi, yi, di, u, v, w, color=style.INK, lw=2.4, arrow_length_ratio=0.42)
# endpoints + sag markers
ax.scatter([xr[0]/1000], [yr[0]], [depth[0]], color=style.PALETTE[2], s=55, depthshade=False)
ax.scatter([xr[-1]/1000], [yr[-1]], [depth[-1]], color=style.PALETTE[1], s=55, depthshade=False)
ax.scatter([xr[i_sag]/1000], [yr[i_sag]], [depth[i_sag]], color=style.PALETTE[3], s=50, depthshade=False)
_lblbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.80)
ax.text(xr[0]/1000-0.6, yr[0]-6, depth[0]-22, "inlet", color=style.PALETTE[2],
        fontsize=10, fontweight="bold", bbox=_lblbox)
ax.text(xr[-1]/1000+0.9, yr[-1]+6, depth[-1]+4, "host", color=style.PALETTE[1],
        fontsize=10, fontweight="bold", bbox=_lblbox)
ax.text(xr[i_sag]/1000-1.0, yr[i_sag]-8, depth[i_sag]-30, "sag", color=style.PALETTE[3],
        fontsize=10, fontweight="bold", bbox=_lblbox)
ax.set_xlim(xr.min()/1000, xr.max()/1000); ax.set_ylim(yr.min(), yr.max())
ax.set_zlim(depth.max()+12, depth.min()-12)                  # zoom to route -> terrain visible
ax.view_init(elev=22, azim=-60)
ax.set_xlabel("Distance (km)"); ax.set_ylabel("Plan offset (m)"); ax.set_zlabel("Depth (m)")
fix3d(ax)
ax.set_title("3-D mixture-velocity vectors along the pipeline")
fig.colorbar(lc, ax=ax, shrink=0.6, pad=0.1).set_label("Mixture velocity (m/s)")
reg(fig, "D13_velocity_vectors_3d.png", "3-D velocity vectors",
    "Three-dimensional velocity vectors (navy arrows) tangent to the seabed route, with "
    "the route coloured by mixture velocity along the line. Depth axis zoomed to the "
    "route so the terrain is visible.")
