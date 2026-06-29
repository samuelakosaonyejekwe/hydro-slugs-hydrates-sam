"""STAGE 07 - ENGINEERING DRAWINGS
Standard, dimensioned engineering drawings of the SUBSEA PIPELINE carrying live
crude oil: pipe cross-section, side elevation (seabed profile), plan, end view,
isometric and a third-angle orthographic multi-view.  ISO-128 / ASME-Y14
conventions; clean, with text placed clear of the geometry."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib.pyplot as plt
import style, common, draw_utils as dw
import usmh_solver as U

style.apply_style()
cfg = U.CaseConfig()
DRW = os.path.join(common.ROOT, "07_engineering_drawings")
os.makedirs(DRW, exist_ok=True)
common.DIRS["drawings"] = DRW

def save(fig, name, title, cap):
    p = os.path.join(DRW, name)
    fig.savefig(p, dpi=170, facecolor=style.FACE, bbox_inches="tight")
    plt.close(fig)
    common.register_figure(p, "drawings", title, cap)

# ===========================================================================
# DWG-101  PIPE CROSS-SECTION (SECTIONAL VIEW) - to scale
# ===========================================================================
ID = cfg.D*1000; WT = 18.3; FBE = 3.0; INS = 60.0; OUT = 22.0
layers = [("Bore (crude oil)", ID/2, "#cfe8ff"),
          ("Carbon-steel pipe wall (API 5L X65)", ID/2+WT, None),
          ("Anti-corrosion coating (FBE)", ID/2+WT+FBE, "#ffd9a6"),
          ("Thermal insulation (syntactic PU)", ID/2+WT+FBE+INS, "#bfe3c0"),
          ("Outer HDPE shield", ID/2+WT+FBE+INS+OUT, "#d8c7f0")]
OD = 2*layers[-1][1]
fig, ax = dw.new_sheet()
cx, cy = 88, 116; sc = 145.0/OD
def rr(real): return real*sc
for name, ro, col in reversed(layers):
    if col: ax.add_patch(plt.Circle((cx, cy), rr(ro), fc=col, ec=dw.LINE, lw=1.1))
r_o = rr(ID/2+WT); r_i = rr(ID/2)
ax.add_patch(plt.Circle((cx, cy), r_o, fc="#9fb3d8", ec=dw.LINE, lw=1.2))
ax.add_patch(plt.Circle((cx, cy), r_i, fc="#cfe8ff", ec=dw.LINE, lw=1.2))
for a in np.arange(0, 2*np.pi, 0.16):
    ax.plot([cx+r_i*np.cos(a), cx+r_o*np.cos(a)], [cy+r_i*np.sin(a), cy+r_o*np.sin(a)],
            color=dw.HATCH, lw=0.4, alpha=0.7)
dw.centerline(ax, cx-rr(OD/2)-8, cy, cx+rr(OD/2)+8, cy)
dw.centerline(ax, cx, cy-rr(OD/2)-8, cx, cy+rr(OD/2)+8)
dw.dim_h(ax, cx-r_i, cx+r_i, cy-rr(OD/2)-6, f"ID {ID:.1f}", off=-6)
dw.dim_h(ax, cx-r_o, cx+r_o, cy-rr(OD/2)-6, f"steel OD {ID+2*WT:.1f}", off=-16)
dw.dim_h(ax, cx-rr(OD/2), cx+rr(OD/2), cy-rr(OD/2)-6, f"OD {OD:.1f}", off=-26)
# radial leader fan (upper-right) so the leaders never cross one another
tvals = [WT, WT, FBE, INS, OUT]
tx = cx + rr(OD/2) + 16
angles = np.deg2rad([72, 58, 44, 30, 16])
for i, (name, ro, col) in enumerate(layers):
    rprev = rr(layers[i-1][1]) if i else 0.0
    rmid = (rr(ro) + rprev) / 2
    a = angles[i]
    p_from = (cx + rmid*np.cos(a), cy + rmid*np.sin(a))
    ty = cy + (rr(OD/2) + 8) * np.sin(a)
    dw.leader(ax, p_from, (tx, ty), name + (f"  (t={tvals[i]:.1f})" if i else ""))
ax.text(20, 188, "PIPE CROSS-SECTION  (Section A-A)", fontsize=11, color=dw.DIMC, fontweight="bold")
ax.text(tx, 128, "NOTES:\n1. All dimensions in millimetres.\n"
        "2. Insulated subsea line; design OHTC ~9 W/m2K.\n3. Pipe grade API 5L X65.",
        fontsize=6.5, color=dw.LINE, va="top")
dw.title_block(ax, "Pipe cross-section (Section A-A)", "USMH-101", scale="1:3")
save(fig, "DWG101_cross_section.png", "Pipe cross-section (sectional view)",
     "Dimensioned sectional view of the insulated pipeline: bore, carbon-steel wall "
     "(hatched), anti-corrosion coating, thermal insulation and outer shield. To scale; "
     "all dimensions in millimetres.")

# ===========================================================================
# DWG-102  SIDE ELEVATION (SEABED LONGITUDINAL PROFILE) - NTS
# ===========================================================================
fig, ax = dw.new_sheet()
s = np.linspace(0, cfg.L_pipe, 240)
elev = np.array([U._elev_of_s(cfg, si) for si in s])
i_sag = int(np.argmin(elev))
xa, xb = 28, 250
X = xa + (xb-xa)*s/cfg.L_pipe
# map elevation to a band around y0
y0 = 120; yscale = 0.35
Y = y0 + yscale*elev
y_surf = 188
ax.add_patch(plt.Rectangle((20, Y.min()-14), 238, y_surf-(Y.min()-14), fc="#eaf4ff", ec="none", zorder=0))
xs = np.linspace(20, 258, 120); ax.plot(xs, y_surf+1.0*np.sin(xs/6), color=dw.DIMC, lw=1.2)
ax.text(24, y_surf+4, "sea surface", fontsize=7, color=dw.DIMC)
# seabed (hatched below the pipe)
ax.fill_between(X, Y-3, Y.min()-14, color="#e7d6b8", ec="#8a6d3b", lw=0.6, alpha=0.6)
ax.plot(X, Y, color=dw.LINE, lw=3.0)                       # the pipeline
ax.scatter([X[i_sag]], [Y[i_sag]], color=dw.CENT, s=45, zorder=5)
ax.annotate("dominant sag\n(terrain-slugging trap)", (X[i_sag], Y[i_sag]),
            (X[i_sag]+8, Y[i_sag]-16), color=dw.CENT, fontsize=7,
            arrowprops=dict(arrowstyle="->", color=dw.CENT))
ax.add_patch(plt.Rectangle((xa-9, Y[0]-3), 8, 7, fc="#bfe3c0", ec=dw.LINE, lw=1))
ax.text(xa-5, Y[0]+8, "Wellhead /\nmanifold", fontsize=7, color=dw.LINE, ha="center")
ax.add_patch(plt.Rectangle((xb+1, Y[-1]-3), 9, 7, fc="#ffd9a6", ec=dw.LINE, lw=1))
ax.text(xb+5, Y[-1]+9, "Host", fontsize=7, color=dw.LINE, ha="center")
for xx in np.linspace(xa+15, xb-15, 7):
    ax.annotate("", xy=(xx+6, y0-6), xytext=(xx, y0-6),
                arrowprops=dict(arrowstyle="-|>", color=dw.CENT, lw=0.9))
ax.text((xa+xb)/2, y0-12, "crude-oil multiphase flow", fontsize=7, color=dw.CENT, ha="center")
dw.dim_h(ax, xa, xb, Y.min()-16, "PIPELINE LENGTH = 16.0 km", off=-8)
dw.dim_v(ax, Y.min()-14, y_surf, 256, "WATER DEPTH ~ 1480 m", off=4)
ax.text(20, 196, "SIDE ELEVATION  (seabed longitudinal profile, vertical scale exaggerated)",
        fontsize=10, color=dw.DIMC, fontweight="bold")
ax.text(20, 28, "Inlet: ~70 bar, 60 °C    Operating pressure terrain-controlled    "
        "Seabed ambient: 4 °C", fontsize=7, color=dw.LINE)
dw.title_block(ax, "Side elevation - seabed profile", "USMH-102", scale="NTS")
save(fig, "DWG102_side_elevation.png", "Side elevation (seabed profile)",
     "Dimensioned side elevation of the seabed pipeline from the wellhead/manifold to the "
     "host, showing the terrain undulations and the dominant sag (terrain-slugging trap); "
     "vertical scale exaggerated.")

# ===========================================================================
# DWG-103  PLAN VIEW (TOP-DOWN)
# ===========================================================================
fig, ax = dw.new_sheet()
Yp = 110 + 0.02*300*np.sin(2*np.pi*s/7000.0)
Xp = 30 + 215*s/cfg.L_pipe
ax.plot(Xp, Yp, color=dw.LINE, lw=3.0)
ax.add_patch(plt.Rectangle((21, 107), 8, 7, fc="#bfe3c0", ec=dw.LINE, lw=1))
ax.text(25, 102, "Manifold", fontsize=7, color=dw.LINE, ha="center")
ax.scatter([Xp[-1]], [Yp[-1]], s=55, color=dw.CENT, zorder=5)
ax.text(Xp[-1]-2, Yp[-1]+5, "Host", fontsize=7, color=dw.LINE, ha="right")
for xx in np.linspace(45, 235, 7):
    ax.annotate("", xy=(xx+6, 110), xytext=(xx, 110),
                arrowprops=dict(arrowstyle="-|>", color=dw.CENT, lw=0.9))
dw.centerline(ax, 25, 110, 250, 110)
dw.dim_h(ax, Xp[0], Xp[-1], 96, "PIPELINE STEP-OUT = 16.0 km", off=-10)
dw.dim_v(ax, 110, 110+0.02*300, 252, "ROUTE\nCORRIDOR\n~300 m", off=4)
ax.text(20, 188, "PLAN VIEW  (looking down)", fontsize=11, color=dw.DIMC, fontweight="bold")
ax.text(20, 30, "NOTE: route corridor shown to scale laterally; length is NTS.",
        fontsize=7, color=dw.LINE)
dw.title_block(ax, "Plan view", "USMH-103", scale="NTS")
save(fig, "DWG103_plan_view.png", "Plan view (top-down)",
     "Plan view of the seabed route from the manifold to the host facility, showing the "
     "16 km step-out and the lateral route corridor.")

# ===========================================================================
# DWG-104  END VIEW (looking along the pipe axis, on the seabed)
# ===========================================================================
fig, ax = dw.new_sheet()
y_surf, y_bed = 185, 70
ax.add_patch(plt.Rectangle((60, y_bed), 150, y_surf-y_bed, fc="#eaf4ff", ec="none"))
xs = np.linspace(60, 210, 80); ax.plot(xs, y_surf+1.0*np.sin(xs/5), color=dw.DIMC, lw=1.2)
ax.text(64, y_surf+4, "sea surface", fontsize=7, color=dw.DIMC)
ax.fill_between(xs, y_bed, y_bed-8, color="#e7d6b8", ec="#8a6d3b", lw=0.8)
ax.text(150, y_bed-5, "seabed", fontsize=7, color="#8a6d3b")
# pipe section resting on the seabed
ax.add_patch(plt.Circle((135, y_bed+6), 9, fc="#d8c7f0", ec=dw.LINE, lw=1.2))
ax.add_patch(plt.Circle((135, y_bed+6), 6.5, fc="#9fb3d8", ec=dw.LINE, lw=1.0))
ax.add_patch(plt.Circle((135, y_bed+6), 4.3, fc="#cfe8ff", ec=dw.LINE, lw=1.0))
dw.leader(ax, (140, y_bed+9), (175, y_bed+30), "Pipe OD 515.7 mm (with coatings)")
dw.dim_v(ax, y_bed+6, y_surf, 205, "WATER DEPTH ~ 1480 m", off=4)
dw.centerline(ax, 120, y_bed+6, 150, y_bed+6)
ax.text(20, 188, "END VIEW  (looking along pipe axis)", fontsize=11, color=dw.DIMC, fontweight="bold")
dw.title_block(ax, "End view", "USMH-104", scale="NTS")
save(fig, "DWG104_end_view.png", "End view",
     "End view looking along the pipe axis: the coated pipe section resting on the seabed "
     "at ~1480 m water depth.")

# ===========================================================================
# DWG-105  ISOMETRIC VIEW
# ===========================================================================
def iso(x, y, z):
    c = np.cos(np.pi/6); sgn = np.sin(np.pi/6)
    return (x-y)*c, (x+y)*sgn + z
fig, ax = dw.new_sheet()

# --- auto-fit the whole isometric model inside the inner border ---------------
# Raw iso extents are computed from every model point so the drawing can never
# spill outside the frame (target box clears the bottom-right title block).
sx = np.linspace(0, 150, 120)
sz = 6*np.sin(2*np.pi*sx/40) - 12*np.exp(-((sx-75)/10)**2)
_raw = [iso(0,0,0), iso(150,0,0), iso(150,70,0), iso(0,70,0)]            # seabed plane
_raw += [iso(xx, 35, zz+10) for xx, zz in zip(sx, sz)]                   # pipeline
_raw += [iso(22,0,0), iso(0,22,0), iso(0,0,22)]                          # axis triad
_rx = [p[0] for p in _raw]; _ry = [p[1] for p in _raw]
rx0, rx1, ry0, ry1 = min(_rx), max(_rx), min(_ry), max(_ry)
# target rectangle inside the inner border, clear of the title block
TX0, TX1, TY0, TY1 = 24, 250, 58, 184
sc = min((TX1-TX0)/(rx1-rx0), (TY1-TY0)/(ry1-ry0))
ox = TX0 + ((TX1-TX0) - sc*(rx1-rx0))/2 - sc*rx0
oy = TY0 + ((TY1-TY0) - sc*(ry1-ry0))/2 - sc*ry0
def Pt(x, y, z):
    Xx, Yy = iso(x, y, z); return ox+sc*Xx, oy+sc*Yy
# seabed parallelogram
corners = [Pt(0,0,0), Pt(150,0,0), Pt(150,70,0), Pt(0,70,0)]
ax.add_patch(plt.Polygon(corners, closed=True, fc="#e7d6b8", ec="#8a6d3b", lw=0.8, alpha=0.55))
# pipeline along x at y=35 with terrain (z) undulation incl. sag
sx = np.linspace(0, 150, 120)
sz = 6*np.sin(2*np.pi*sx/40) - 12*np.exp(-((sx-75)/10)**2)
pl = [Pt(xx, 35, zz+10) for xx, zz in zip(sx, sz)]
ax.plot([p[0] for p in pl], [p[1] for p in pl], color=dw.LINE, lw=3)
mx, my = Pt(0, 35, 10); ax.add_patch(plt.Rectangle((mx-3, my-2), 6, 5, fc="#bfe3c0", ec=dw.LINE))
hx, hy = Pt(150, 35, 10); ax.add_patch(plt.Rectangle((hx-3, hy-2), 6, 5, fc="#ffd9a6", ec=dw.LINE))
ax.text(mx-6, my-6, "Manifold", fontsize=7, color=dw.LINE)
ax.text(hx-2, hy+8, "Host", fontsize=7, color=dw.LINE)
isag = int(np.argmin(sz)); sgx, sgy = Pt(sx[isag], 35, sz[isag]+10)
ax.scatter([sgx], [sgy], color=dw.CENT, s=40, zorder=5)
ax.text(sgx-4, sgy-7, "sag", fontsize=7, color=dw.CENT)
ax.text(*Pt(70,35,16), s="seabed pipeline, 16 km", fontsize=7, color=dw.LINE)
# iso axes triad
ax.annotate("", xy=Pt(22,0,0), xytext=Pt(0,0,0), arrowprops=dict(arrowstyle="-|>", color=dw.CENT))
ax.annotate("", xy=Pt(0,22,0), xytext=Pt(0,0,0), arrowprops=dict(arrowstyle="-|>", color=dw.CENT))
ax.annotate("", xy=Pt(0,0,22), xytext=Pt(0,0,0), arrowprops=dict(arrowstyle="-|>", color=dw.CENT))
ax.text(*Pt(24,0,0), s="X", color=dw.CENT, fontsize=7)
ax.text(*Pt(0,24,0), s="Y", color=dw.CENT, fontsize=7)
ax.text(*Pt(0,0,24), s="Z", color=dw.CENT, fontsize=7)
ax.text(20, 190, "ISOMETRIC VIEW", fontsize=11, color=dw.DIMC, fontweight="bold")
dw.title_block(ax, "Isometric general arrangement", "USMH-105", scale="NTS")
save(fig, "DWG105_isometric.png", "Isometric view",
     "Isometric general arrangement of the seabed pipeline from the manifold to the host, "
     "showing the terrain undulations and the dominant sag.")

# ===========================================================================
# DWG-106  ORTHOGRAPHIC MULTI-VIEW (third-angle)
# ===========================================================================
fig, ax = dw.new_sheet()
fx0, fy0 = 30, 55
sxv = np.linspace(0, 70, 80); szv = 4*np.sin(2*np.pi*sxv/22) - 7*np.exp(-((sxv-35)/5)**2)
# FRONT view (seabed profile)
ax.plot(fx0+sxv, fy0+18+szv, color=dw.LINE, lw=2.4)
ax.fill_between(fx0+sxv, fy0+15+szv, fy0, color="#e7d6b8", ec="#8a6d3b", lw=0.5, alpha=0.5)
ax.text(fx0+2, fy0-6, "FRONT VIEW (seabed profile)", fontsize=8, color=dw.DIMC, fontweight="bold")
dw.dim_h(ax, fx0, fx0+70, fy0-2, "16.0 km", off=-5)
# TOP view (plan) above front
ty0 = fy0+55
ax.plot([fx0, fx0+70], [ty0+8, ty0+8], color=dw.LINE, lw=2.4)
ax.scatter([fx0+70], [ty0+8], s=30, color=dw.CENT)
ax.text(fx0+2, ty0+20, "TOP VIEW (PLAN)", fontsize=8, color=dw.DIMC, fontweight="bold")
for xx in (fx0, fx0+70):
    ax.plot([xx, xx], [fy0+34, ty0+6], color=dw.DIMC, lw=0.4, ls=(0, (2, 2)))
# SIDE / END view (right of front)
sx0 = fx0+95
ax.add_patch(plt.Circle((sx0+18, fy0+15), 7, fc="#d8c7f0", ec=dw.LINE, lw=1.1))
ax.add_patch(plt.Circle((sx0+18, fy0+15), 4.2, fc="#cfe8ff", ec=dw.LINE, lw=1.0))
ax.text(sx0+2, fy0-6, "END VIEW", fontsize=8, color=dw.DIMC, fontweight="bold")
dw.leader(ax, (sx0+24, fy0+18), (sx0+34, fy0+34), "OD 515.7")
for yy in (fy0+8, fy0+22):
    ax.plot([fx0+72, sx0+8], [yy, yy], color=dw.DIMC, lw=0.4, ls=(0, (2, 2)))
ax.text(20, 192, "ORTHOGRAPHIC PROJECTION  (third angle)", fontsize=11, color=dw.DIMC, fontweight="bold")
ax.text(150, 150, "Three principal views (third-angle):\n - Front: seabed longitudinal profile\n"
        " - Top: route plan / step-out\n - End: pipe cross-section", fontsize=7.5,
        color=dw.LINE, va="top")
dw.title_block(ax, "Orthographic multi-view (3rd angle)", "USMH-106", scale="NTS")
save(fig, "DWG106_orthographic.png", "Orthographic multi-view (third-angle)",
     "Third-angle orthographic projection with front (seabed profile), top (plan) and end "
     "(cross-section) views in standard arrangement with projection lines.")

print("Stage 07 engineering drawings complete (6 drawings).")
