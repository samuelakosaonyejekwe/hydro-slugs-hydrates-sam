"""STAGE 01 - GEOMETRY
Builds the subsea export pipeline geometry: a ~16 km seabed line at ~1480 m water
depth with terrain undulations and a dominant sag (terrain-slugging trap)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import style, common
import usmh_solver as U

style.apply_style()
cfg = U.CaseConfig()
D = common.DIRS["geometry"]

s = np.linspace(0, cfg.L_pipe, 600)
elev = np.array([U._elev_of_s(cfg, si) for si in s])              # +up, rel. mean seabed
theta = np.array([U._theta_of_s(cfg, si) for si in s])
x = s.copy()                                                       # near-horizontal route
y = 60.0 * np.sin(np.pi * s / cfg.L_pipe)                         # gentle plan route corridor
depth = cfg.water_depth - elev                                     # depth below sea surface
i_sag = int(np.argmin(elev))

df = pd.DataFrame({"arc_length_m": s, "x_horizontal_m": x, "y_plan_m": y,
                   "seabed_elevation_m": elev, "depth_below_surface_m": depth,
                   "inclination_deg": np.rad2deg(theta)})
common.save_table(df, "pipeline_geometry.csv", "geometry",
                  "Subsea pipeline route geometry",
                  "Discretised seabed route: arc length, plan coordinates, terrain "
                  "elevation, water depth and local inclination.")

seg = pd.DataFrame({
    "Property": ["Internal diameter", "Outside diameter (with coatings)",
                 "Wall thickness", "Wall roughness", "Pipeline length",
                 "Mean water depth", "Terrain undulation amplitude",
                 "Dominant sag depth", "Sag location", "Max |inclination|"],
    "Value": [f"{cfg.D*1000:.1f} mm ({cfg.D/0.0254:.0f} in)", "515.7 mm",
              "18.3 mm (API 5L X65)", f"{cfg.eps*1e6:.0f} um",
              f"{cfg.L_pipe/1000:.1f} km", f"{cfg.water_depth:.0f} m",
              f"+/-{cfg.terrain_amp:.0f} m", f"{cfg.sag_extra:.0f} m below trend",
              f"{s[i_sag]/1000:.1f} km", f"{np.abs(np.rad2deg(theta)).max():.1f} deg"]})
common.save_table(seg, "geometry_summary.csv", "geometry", "Geometry summary",
                  "Key geometric parameters of the subsea crude-oil pipeline.")

# ---- Fig: seabed elevation / depth profile --------------------------------
fig, ax = plt.subplots(figsize=(10.0, 4.8))
ax.plot(s/1000, depth, color=style.PALETTE[0], lw=2.8)
ax.fill_between(s/1000, depth, depth.max()+20, color="#e7d6b8", alpha=0.5, zorder=0)
ax.scatter([s[i_sag]/1000], [depth[i_sag]], color=style.PALETTE[3], s=70, zorder=5)
ax.annotate("dominant sag\n(terrain-slugging trap)", (s[i_sag]/1000, depth[i_sag]),
            (s[i_sag]/1000+1.0, depth[i_sag]+30), color=style.PALETTE[3], fontsize=9,
            arrowprops=dict(arrowstyle="->", color=style.PALETTE[3]))
ax.invert_yaxis()
ax.set_xlabel("Arc length along pipeline (km)")
ax.set_ylabel("Depth below sea surface (m)")
ax.set_title("Subsea pipeline seabed profile")
ax.annotate("inlet (wellhead/manifold)", (0.2, depth[0]), color=style.PALETTE[2], fontsize=9)
ax.annotate("outlet (host)", (s[-1]/1000-0.2, depth[-1]), color=style.PALETTE[1],
            fontsize=9, ha="right")
common.register_figure(style.finish(fig, os.path.join(D, "fig_elevation_profile.png")),
                       "geometry", "Seabed depth profile",
                       "Depth of the pipeline below the sea surface along the seabed; the "
                       "terrain undulates with a dominant sag at ~8 km that traps liquid.")

# ---- Fig: terrain elevation + inclination ---------------------------------
fig, axs = plt.subplots(2, 1, figsize=(10.0, 6.0), sharex=True)
axs[0].plot(s/1000, elev, color=style.PALETTE[2], lw=2.6)
axs[0].axhline(0, color=style.GRID, lw=1)
axs[0].scatter([s[i_sag]/1000], [elev[i_sag]], color=style.PALETTE[3], s=55, zorder=5)
axs[0].set_ylabel("Seabed elevation (m)")
axs[0].set_title("Terrain elevation and local inclination")
axs[1].plot(s/1000, np.rad2deg(theta), color=style.PALETTE[1], lw=2.2)
axs[1].axhline(0, color=style.GRID, lw=1)
axs[1].set_xlabel("Arc length (km)"); axs[1].set_ylabel("Inclination (deg)")
common.register_figure(style.finish(fig, os.path.join(D, "fig_inclination.png")),
                       "geometry", "Terrain elevation & inclination",
                       "Seabed terrain relative to the mean and the resulting local pipe "
                       "inclination (uphill legs after sags promote slugging).")

# ---- Fig: 3D route (clear, labelled, with seabed context) -----------------
fig = plt.figure(figsize=(10.0, 6.8))
ax = fig.add_subplot(111, projection="3d")
ax.plot(x/1000, y, depth, color=style.PALETTE[0], lw=3.5, solid_capstyle="round")
ax.scatter([x[0]/1000], [y[0]], [depth[0]], color=style.PALETTE[2], s=70, depthshade=False)
ax.scatter([x[-1]/1000], [y[-1]], [depth[-1]], color=style.PALETTE[1], s=70, depthshade=False)
ax.scatter([x[i_sag]/1000], [y[i_sag]], [depth[i_sag]], color=style.PALETTE[3], s=60, depthshade=False)
ax.text(x[0]/1000, y[0], depth[0]-22, " inlet", color=style.PALETTE[2], fontsize=9)
ax.text(x[-1]/1000, y[-1], depth[-1]-22, " outlet (host)", color=style.PALETTE[1], fontsize=9)
ax.text(x[i_sag]/1000, y[i_sag], depth[i_sag]+22, " sag", color=style.PALETTE[3], fontsize=9)
ax.invert_zaxis(); ax.view_init(elev=24, azim=-62)
ax.set_xlabel("Horizontal distance (km)"); ax.set_ylabel("Plan offset (m)")
ax.set_zlabel("Depth below surface (m)")
ax.set_title("3-D subsea pipeline route on the seabed")
common.register_figure(style.finish(fig, os.path.join(D, "fig_route_3d.png")),
                       "geometry", "3-D pipeline route",
                       "Three-dimensional layout of the seabed pipeline (vertical scale "
                       "exaggerated): inlet, the dominant sag, and the outlet at the host "
                       "facility. The line runs along the seabed at ~1480 m water depth.")

print("Stage 01 geometry complete.")
